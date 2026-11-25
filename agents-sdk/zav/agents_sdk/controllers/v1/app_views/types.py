from typing import Any, Dict, Optional

from zav.pydantic_compat import BaseModel, Field

# NOTE: model docstrings become the schema ``description`` in the generated
# OpenAPI spec, so implementation detail stays in comments.


# Names a tool call's view: the agent that ran the call and the tool, as the
# streamed tool step names it.
class AppViewForm(BaseModel):
    agent_identifier: str
    view_tool: str
    bot_params: Optional[Dict[str, Any]] = None


# A view's HTML and the MCP Apps settings to render it with (csp, permissions,
# domain, prefersBorder).
class AppViewItem(BaseModel):
    html: str
    ui_meta: Dict[str, Any] = Field(default_factory=dict)


# A call a view makes to a tool of its own origin, by the name it has there.
class AppToolCallForm(AppViewForm):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AppToolCallItem(BaseModel):
    result: Any = None
