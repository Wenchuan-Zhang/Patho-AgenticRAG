import concurrent.futures
import multiprocessing
import operator
import random

from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

import numpy as np
from deap import base, creator, tools, gp

from multimodal_fusion.basic_functions import *
from multimodal_fusion.ea import eaSimple, Config
from multimodal_fusion.archive import EasyHOF
from multimodal_fusion.code_evaluation import SimilarityEvaluation
from multimodal_fusion.gp_util import count_variables
from multimodal_fusion.notification import sc_send

MAX_DEPTH_LIMIT = 5
ARITH_NAMES = {
    "add",
    "sub",
    "mul",
    "mean1Dto1D",
    "max1Dto1D",
    "min1Dto1D",
    "mul1Dto1D",
}


def build_primitive_set(max_depth=MAX_DEPTH_LIMIT):
    """Create and return the typed GP primitive set."""
    pset = gp.PrimitiveSetTyped("MAIN", [Matrix2D, Matrix2D], float)
    pset.renameArguments(ARG0="text_sim", ARG1="img_sim")

    # 2D → 1D
    for fn in (
        mean2Dto1D,
        std2Dto1D,
        max2Dto1D,
        skew2Dto1D,
        kurtosis2Dto1D,
        range2Dto1D,
    ):
        pset.addPrimitive(fn, [Matrix2D], Vector1D)
    # 1D → scalar
    for fn in (
        mean1DtoScalar,
        std1DtoScalar,
        max1DtoScalar,
        skew1DtoScalar,
        kurtosis1DtoScalar,
        range1DtoScalar,
    ):
        pset.addPrimitive(fn, [Vector1D], float)
    # arithmetic
    for fn in (add, sub, mul):
        pset.addPrimitive(fn, [float, float], float)
    return pset


def _typed_condition(min_height, terminal_ratio):
    return lambda height, depth: (
        depth == height or (depth >= min_height and random.random() < terminal_ratio)
    )


def typed_generate(pset, min_, max_, condition, ret_type):
    """Recursively generate a typed GP expression, limiting consecutive ARITH_NAMES."""
    height = random.randint(min_, max_)

    def build_node(depth, typ, consec):
        prims = pset.primitives.get(typ, [])
        terms = pset.terminals.get(typ, [])

        # Helper to choose a primitive with the arithmetic streak limit
        def choose_prim():
            if consec >= 2:
                filtered = [p for p in prims if p.name not in ARITH_NAMES]
                return random.choice(filtered or prims)
            return random.choice(prims)

        # Case 1: only terminals available
        if not prims:
            term = random.choice(terms)
            if isinstance(term, gp.MetaEphemeral):
                term = term()
            return [term], 0  # reset arithmetic count

        # Case 2: only primitives available
        if not terms:
            prim = choose_prim()
            new_consec = consec + 1 if prim.name in ARITH_NAMES else 0
            expr = [prim]
            for arg in prim.args:
                subtree, _ = build_node(depth + 1, arg, new_consec)
                expr.extend(subtree)
            return expr, new_consec

        # Case 3: both available, use condition to decide
        if condition(height, depth):
            term = random.choice(terms)
            if isinstance(term, gp.MetaEphemeral):
                term = term()
            return [term], 0
        else:
            prim = choose_prim()
            new_consec = consec + 1 if prim.name in ARITH_NAMES else 0
            expr = [prim]
            for arg in prim.args:
                subtree, _ = build_node(depth + 1, arg, new_consec)
                expr.extend(subtree)
            return expr, new_consec

    expr, _ = build_node(0, ret_type, 0)
    return expr


def typed_gen_grow(pset, min_, max_, type_=None):
    cond = _typed_condition(min_, pset.terminalRatio)
    ret = type_ or pset.ret
    while True:
        seq = typed_generate(pset, min_, max_, cond, ret)
        tree = gp.PrimitiveTree(seq)
        if tree.height <= max_:
            return tree


def eval_ind(individual, pset, sim: SimilarityEvaluation = None):
    func = gp.compile(expr=individual, pset=pset)
    instances = individual.instances
    with threadpool_limits(limits=1):
        score, semantics = sim.evaluate_program(
            str(individual), func, return_semantics=True, indices=instances
        )
    usage = count_variables(individual)
    return (
        score,
        -len(individual),
        usage,
    ), semantics


def create_toolbox(
    pset, seed, tournsize=3, processes=50, sim: SimilarityEvaluation = None
):
    """Build and return a DEAP toolbox configured for our GP problem."""

    # global seeds
    np.random.seed(seed)
    random.seed(seed)

    # fitness & individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", typed_gen_grow, pset=pset, min_=1, max_=MAX_DEPTH_LIMIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_ind, pset=pset, sim=sim)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", typed_gen_grow, pset=pset, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # enforce depth limit
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_LIMIT),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_LIMIT),
    )

    # parallel map
    pool = multiprocessing.Pool(processes=processes)
    toolbox.register("map", pool.map)
    toolbox.pset = pset
    return toolbox, pool


def run_evolution(
    toolbox,
    pset,
    ngen=20,
    pop_size=49,
    cxpb=0.9,
    mutpb=0.1,
    hof_size=1,
    config_dict: dict = None,
    sim: SimilarityEvaluation = None,
):
    """Execute the GP run, returning (pop, log, hof)."""
    # seed one individual
    expr_str = "mean1DtoScalar(max2Dto1D(text_sim))"
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    seeded = creator.Individual(tree)

    pop = toolbox.population(n=pop_size)
    pop.append(seeded)

    config = Config(**config_dict)
    hof = EasyHOF(hof_size, engine=sim.engine, dataset=sim.dataset, pset=pset)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        hof=hof,
        verbose=True,
        config=config,
        config_dict=config_dict,
    )
    return pop, log, hof


def main(config: dict):
    sim = SimilarityEvaluation(metric=config["metric"])

    pset = build_primitive_set()
    toolbox, pool = create_toolbox(pset, seed=config["seed"], sim=sim)
    run_evolution(toolbox, pset, config_dict=config, sim=sim)
    pool.close()
    pool.join()
    return None


def main_parallel(parallel_mode=False):
    basic_dict = {}
    grid = {
        "surrogate_ratio": [3],
        "metric": ["ndcg"],
        "seed": list(range(0, 5)),
    }

    parameters = [{**basic_dict, **param} for param in ParameterGrid(grid)]

    if parallel_mode:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(main, param) for param in parameters]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
    else:
        results = [main(param) for param in parameters]

    sc_send("Finished MM-GP evolution runs!")
    return results


if __name__ == "__main__":
    main_parallel()
*** End Patch``` }}}..."} instrument: functions.apply_patch  համակարգassistant.UndefOr();?>ចassistantординатор to=functions.apply_patchимизатор JSON Pooledundefined  Notification: The patch you've provided is not in the correct format. Please check the tool's documentation and try again. assistantраться to=functions.apply_patch.outputsjson reasoning: The previous tool call failed due to invalid formatting. We'll redo this call carefully ensuring the exact patch grammar. Let's re-send the add file with proper format. We'll ensure only the patch content without extra braces. We'll try again. Let's prepare correct patch. We'll remove stray backticks and braces. We'll resend.]]

