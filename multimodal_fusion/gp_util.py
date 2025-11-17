from deap import gp


def count_variables(individual):
    used = [node.name for node in individual if isinstance(node, gp.Terminal)]

    return len(set(used))


