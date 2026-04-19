import json
from typing import Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.ethusdt.experiment.factor_experiment import (
    ETHUSDTFactorExperiment,
    ETHUSDTFactorScenario,
)
from rdagent.utils.agent.tpl import T

ETHUSDTFactorHypothesis = Hypothesis


class ETHUSDTFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            T("scenarios.ethusdt.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T("scenarios.ethusdt.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Try the easiest and fastest factors to experiment with from various perspectives first."
                if len(trace.hist) < 15
                else "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)."
            ),
            "hypothesis_output_format": T("scenarios.ethusdt.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.ethusdt.prompts:factor_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        return ETHUSDTFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )


class ETHUSDTFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = T("scenarios.ethusdt.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if hasattr(trace.hist[i][0].hypothesis, "action"):
                    # skip model-only experiments
                    if getattr(trace.hist[i][0].hypothesis, "action", None) != "factor":
                        continue
                specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.ethusdt.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []
        for factor_name, d in response_dict.items():
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=d["description"],
                    factor_formulation=d["formulation"],
                    variables=d["variables"],
                )
            )

        exp = ETHUSDTFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [ETHUSDTFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp
