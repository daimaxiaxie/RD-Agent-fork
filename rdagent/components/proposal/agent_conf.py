from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class AgentToolSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_TOOL_")
    enable: bool = True


AGENT_TOOL_SETTINGS = AgentToolSettings()