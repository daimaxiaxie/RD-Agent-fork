"""
ETH/USDT factor workflow with session control
"""

import asyncio
from pathlib import Path
from typing import Optional

import fire

from rdagent.app.ethusdt_rd_loop.conf import ETHUSDT_FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error("Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    **kwargs,
):
    if not kwargs.get("checkout_path") is None:
        checkout = Path(kwargs["checkout_path"])

    if path is None:
        factor_loop = FactorRDLoop(ETHUSDT_FACTOR_PROP_SETTING)
    else:
        factor_loop = FactorRDLoop.load(path, checkout=checkout)

    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
