from rdagent.core.conf import ExtendedBaseSettings


class AgentToolSettings(ExtendedBaseSettings):
    enable: bool = False


AGENT_TOOL_SETTINGS = AgentToolSettings()