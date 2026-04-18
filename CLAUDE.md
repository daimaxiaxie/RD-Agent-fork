# RD-Agent-fork — CLAUDE.md

## Project Overview
R&D-Agent is a multi-agent framework (from Microsoft) that automates research and development processes. This fork focuses on **quantitative finance** — automated factor discovery, model development, and backtesting via Qlib.

## Key Directories
| Path | Purpose |
|------|---------|
| `rdagent/core/` | Base framework classes (RDLoop, FBWorkspace, Trace, etc.) |
| `rdagent/components/` | Reusable components (coder, evaluator, runner, workflow) |
| `rdagent/scenarios/` | Domain-specific implementations |
| `rdagent/scenarios/qlib/` | Qlib quantitative finance scenario |
| `rdagent/app/` | Entry points and workflow orchestration |
| `rdagent/app/qlib_rd_loop/` | Qlib factor/model/quant loop implementations |
| `rdagent/oai/` | LLM integration |
| `rdagent/log/` | Logging system (streamlit UI, server) |
| `constraints/` | Factor definitions (JSON files used as constraint knowledge base) |

## CLI Commands (`rdagent/app/cli.py`)
```
rdagent fin_factor          # Factor evolution loop
rdagent fin_model           # Model evolution loop
rdagent fin_quant           # Combined factor+model evolution
rdagent fin_factor_report   # Extract factors from papers/reports
rdagent general_model       # Extract models from papers
rdagent data_science        # Kaggle/data science workflow
rdagent llm_finetune        # LLM fine-tuning loop
rdagent ui                   # Start Streamlit UI
rdagent server_ui           # Start Flask log server
rdagent health_check        # System validation
```

## Core Workflow: RDLoop
```
direct_exp_gen → coding → running → feedback → record
```
- **direct_exp_gen**: `_propose()` → generate hypothesis, `exp_gen()` → convert to experiment
- **coding**: `RDLoopDeveloper` (`rdagent/core/developer.py`) implements via CoSTEER
- **running**: `Runner` executes the experiment
- **feedback**: `Summarizer` evaluates results and generates feedback for next iteration
- **record**: Tracks evolution history in `Trace`

## Qlib Factor Coder
- **Task**: `FactorTask` (`rdagent/components/coder/factor_coder/factor.py`) — defines factor via name, description, formulation
- **Workspace**: `FactorFBWorkspace` — executes factor code in isolated workspace
- **Version 1**: Direct execution of `factor.py`
- **Version 2**: Uses `factor_execution_template.txt` wrapper that imports `feature_engineering_cls`
- **Input**: `daily_pv.h5` (price/volume data from `factor_data_template/`)
- **Output**: `result.h5` (generated factor values in workspace path)

## Qlib Model Coder
- **Workspace**: `QlibFBWorkspace` (`rdagent/scenarios/qlib/experiment/workspace.py`)
- **Execution**: `qrun conf.yaml` → `python read_exp_res.py`
- **Output**: `ret.pkl` (backtest chart), `qlib_res.csv` (metrics)
- **Environment**: Docker (`local_qlib:latest`) or Conda (`rdagent4qlib`), controlled by `MODEL_COSTEER_SETTINGS.env_type`

## Configuration System
All config uses Pydantic `ExtendedBaseSettings` with environment variable override support.

| Class | File | Key Settings |
|-------|------|--------------|
| `ModelBasePropSetting` | `rdagent/app/qlib_rd_loop/conf.py` | Train 2008-2014, Valid 2015-2016, Test 2017-2020 |
| `FactorBasePropSetting` | same | Same date ranges, max 6 factors per exp |
| `QuantBasePropSetting` | same | Action selection: bandit/llm/random |
| `RDAgentSettings` | `rdagent/core/conf.py` | Workspace: `git_ignore_folder/RD-Agent_workspace` |
| `FactorCoSTEERSettings` | `rdagent/components/coder/factor_coder/config.py` | Timeout: 3600s |
| `ModelCoSTEERSettings` | `rdagent/components/coder/model_coder/conf.py` | env_type: conda/docker |

**Env var override prefix**: `QLIB_` for model/factor settings

## Key Files for Common Tasks
- Add new factor definitions: `rdagent/scenarios/qlib/experiment/prompts.yaml`
- Modify workflow steps: `rdagent/app/qlib_rd_loop/` (`factor.py`, `model.py`, `quant.py`)
- Change prompts/templates: `rdagent/scenarios/qlib/experiment/prompts.yaml`, `factor_template/`
- Factor data: `rdagent/scenarios/qlib/experiment/factor_data_template/`
- Qlib factor expressions: `rdagent/utils/qlib.py` (ALPHA20, ALPHA158)
- User interaction: `rdagent/app/utils/user_interaction.py`

## Notes
- `git_ignore_folder/` contains runtime data (workspace, reports, etc.)
- Pickle caching enabled by default (`cache_with_pickle: True`)
- Constraints knowledge base lives in `constraints/` directory (JSON format)
- Multi-process controlled by `multi_proc_n` setting
