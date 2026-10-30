from __future__ import annotations

from typing import Any, Dict, Optional, Set

# Fields that must always be supplied by the service layer / handler itself and
# therefore must not be overridden by user-supplied ``bot_params``.
_PROTECTED_HANDLER_PARAMS: Set[str] = {
    "tenant",
    "request_headers",
    "index_id",
    "session_id",
    "agent_depth",
    "sub_agent_count",
    "sub_agent_approvals",
}

# Spawn limits, tool rules, the agent's prompt and its tool-step cap belong to
# the agent setup and the tenant: only the setup sets them.
# Otherwise a request, or a delegable agent's stored ``bot_params``, could fill
# them in for every agent of the tree it starts. Which MCP servers receive the
# user's files is the admin's choice too, and so are the platform services'
# URLs: their clients send the caller's credentials there.
_SETUP_ONLY_PARAMS: Set[str] = {
    "sub_agent_limits_configuration",
    "tool_policy_configuration",
    "my_documents_argument_resolver_configuration",
    "system_prompt",
    "system_prompt_provider_configuration",
    "max_nesting_level",
    "chat_service_url",
    "zav_notes_api_host",
    "zav_pipeline_service_url",
    "zav_recommendation_service_url",
    "zav_registration_api_host",
    "zav_retriever_host",
    "zav_saved_results_url",
    "zav_user_documents_api_host",
}
# Each tool rule source's own block (``<name>_rule_source_configuration``) is
# tool policy too: a request filling in ``enabled: false`` would switch the rule
# off for its turn. A system prompt source's block is the agent's prompt.
_SETUP_ONLY_SUFFIXES = (
    "_rule_source_configuration",
    "_system_prompt_source_configuration",
)

# Fields of a configuration block only the agent setup sets. Which MCP servers
# an agent uses, mounted or defined inline, is the admin's choice; a request may
# only narrow it (``selected_servers``), and leaves its confirmation default
# as the setup set it. So are the paths a source reads or writes and the URLs
# the server fetches.
_SETUP_ONLY_FIELDS: Dict[str, Set[str]] = {
    "mcp_tools_provider_configuration": {"mounts", "servers", "confirmation_default"},
    "agent_delegation_provider_configuration": {"sub_agent_approvals"},
    "sub_agent_tool_source_configuration": {"sub_agent_approvals"},
    "disk_instruction_source_configuration": {"instructions_directories"},
    "disk_skills_source_configuration": {"skills_directories"},
    "disk_rubric_source_configuration": {"rubrics_directories"},
    "file_attempt_store_configuration": {"directory"},
    "file_memory_store_configuration": {"file_path"},
    "file_notification_channel_configuration": {"directory"},
    "local_file_mcp_oauth_integration_store_configuration": {"file_path"},
    "monty_repl_sandbox_configuration": {"workspace_directory"},
    "platform_docs_tools_source_configuration": {"docs_base_url"},
    "refiner_configuration": {"ledger_path"},
}

__all__ = [
    "_PROTECTED_HANDLER_PARAMS",
    "sanitize_bot_params",
]


def sanitize_bot_params(bot_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of *bot_params* with security-critical keys stripped.

    This prevents callers from overriding values like ``tenant`` or
    ``request_headers`` that are controlled by the upstream service layer.
    """

    if not bot_params:
        return {}

    # Remove any keys that clash with protected or setup-only parameters. A
    # block with setup-only fields is kept only as a dict: the resolver also
    # parses a JSON string, which would carry those fields past the strip.
    return {
        k: (
            {
                field: value
                for field, value in v.items()
                if field not in _SETUP_ONLY_FIELDS[k]
            }
            if k in _SETUP_ONLY_FIELDS
            else v
        )
        for k, v in bot_params.items()
        if k not in _PROTECTED_HANDLER_PARAMS
        and k not in _SETUP_ONLY_PARAMS
        and not k.endswith(_SETUP_ONLY_SUFFIXES)
        and (k not in _SETUP_ONLY_FIELDS or isinstance(v, dict))
    }
