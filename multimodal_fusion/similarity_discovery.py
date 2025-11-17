import yaml
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

from multimodal_fusion.code_evaluation import SimilarityEvaluation

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm = OpenAIAPI(
        base_url=config["host"],
        api_key=config["key"],
        model=config["model"],
    )
    task = SimilarityEvaluation()
    method = EoH(
        llm=llm,
        evaluation=task,
        profiler=EoHProfiler(log_dir="logs/eoh", log_style="simple"),
        max_sample_nums=100,
        max_generations=10,
        pop_size=10,
        num_samplers=10,
        num_evaluators=10,
        debug_mode=False,
    )
    method.run()


