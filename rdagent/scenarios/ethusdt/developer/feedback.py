import json
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


IMPORTANT_METRICS = ["IC", "Rank IC", "RMSE"]


def process_results(current_result, sota_result):
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)
    current_df.index.name = "metric"
    sota_df.index.name = "metric"
    current_df.rename(columns={current_df.columns[0]: "Current Result"}, inplace=True)
    sota_df.rename(columns={sota_df.columns[0]: "SOTA Result"}, inplace=True)
    filtered_combined_df = pd.concat([current_df, sota_df], axis=1).reindex(IMPORTANT_METRICS)

    results = []
    for metric, row in filtered_combined_df.iterrows():
        current = row.get("Current Result")
        sota = row.get("SOTA Result")
        current_text = "N/A" if pd.isna(current) else f"{current:.6f}"
        sota_text = "N/A" if pd.isna(sota) else f"{sota:.6f}"
        results.append(f"{metric} of Current Result is {current_text}, of SOTA Result is {sota_text}")
    return "; ".join(results)


class ETHUSDTFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        combined_result = process_results(current_result, sota_result)

        sys_prompt = T("scenarios.ethusdt.prompts:factor_feedback_generation.system").r(
            scenario=self.scen.get_scenario_all_desc()
        )

        usr_prompt = T("scenarios.ethusdt.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        response_json = json.loads(response)

        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
