import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type, get_args

from zav.llm_domain import LLMVendorConfiguration
from zav.pydantic_compat import BaseModel

SKILLS_DIR = "skills"
INSTRUCTIONS_DIR = "instructions"
MEMORIES_DIR = "memories"
SPECS_DIR = "specs"
ENV_DIR = "env"
CONFIG_FILE = "agent_setups.json"
MCP_CONFIG_FILE = "mcp_setups.json"
REALTIME_TRANSCRIPTION_CONFIG_FILE = "realtime_transcription_setup.json"
LLM_CONFIG_FILE = "llm_configurations.json"
MCP_CONFIGURATIONS_FILE = "mcp_configurations.json"
SECRET_CONFIG_FILE = os.path.join(ENV_DIR, CONFIG_FILE)
SECRET_LLM_CONFIG_FILE = os.path.join(ENV_DIR, LLM_CONFIG_FILE)
SECRETS_FILE = os.path.join(ENV_DIR, "secrets.json")


def get_local_tenant() -> str:
    return os.getenv("ZAV_TENANT", "zetaalpha")


def get_local_requester_uuid() -> str:
    return os.getenv("ZAV_REQUESTER_UUID", "local-user")


PLATFORM_CONFIG_FILE = os.path.join(ENV_DIR, "zav_config.json")
PROJECT_DIRS = [SKILLS_DIR, INSTRUCTIONS_DIR, MEMORIES_DIR, SPECS_DIR]


class ProjectConfig:
    def __init__(self, project_dir: str):
        self.project_dir = os.path.abspath(project_dir)
        self.__public_path = os.path.join(self.project_dir, CONFIG_FILE)
        self.__secret_path = os.path.join(self.project_dir, SECRET_CONFIG_FILE)
        self.__mcp_path = os.path.join(self.project_dir, MCP_CONFIG_FILE)
        self.__llm_path = os.path.join(self.project_dir, LLM_CONFIG_FILE)
        self.__mcp_configurations_path = os.path.join(
            self.project_dir, MCP_CONFIGURATIONS_FILE
        )
        self.__secret_llm_path = os.path.join(self.project_dir, SECRET_LLM_CONFIG_FILE)
        self.__secrets_path = os.path.join(self.project_dir, SECRETS_FILE)
        self.__public_setups: List[Dict[str, Any]] = []
        self.__secret_setups: List[Dict[str, Any]] = []
        self.__mcp_setups: List[Dict[str, Any]] = []
        self.__llm_configs: List[Dict[str, Any]] = []
        self.__mcp_configurations: List[Dict[str, Any]] = []
        self.__secret_llm_configs: List[Dict[str, Any]] = []
        self.__secrets: List[Dict[str, Any]] = []
        self.__load()

    def __load(self):
        if os.path.isfile(self.__public_path):
            with open(self.__public_path, "r") as f:
                self.__public_setups = json.load(f)
        if os.path.isfile(self.__secret_path):
            with open(self.__secret_path, "r") as f:
                self.__secret_setups = json.load(f)
        if os.path.isfile(self.__mcp_path):
            with open(self.__mcp_path, "r") as f:
                self.__mcp_setups = json.load(f)
        if os.path.isfile(self.__llm_path):
            with open(self.__llm_path, "r") as f:
                self.__llm_configs = json.load(f)
        if os.path.isfile(self.__mcp_configurations_path):
            with open(self.__mcp_configurations_path, "r") as f:
                self.__mcp_configurations = json.load(f)
        if os.path.isfile(self.__secret_llm_path):
            with open(self.__secret_llm_path, "r") as f:
                self.__secret_llm_configs = json.load(f)
        if os.path.isfile(self.__secrets_path):
            with open(self.__secrets_path, "r") as f:
                self.__secrets = json.load(f)

    def save(self):
        with open(self.__public_path, "w") as f:
            json.dump(self.__public_setups, f, indent=2)
            f.write("\n")
        if self.__secret_setups or os.path.isfile(self.__secret_path):
            os.makedirs(os.path.dirname(self.__secret_path), exist_ok=True)
            with open(self.__secret_path, "w") as f:
                json.dump(self.__secret_setups, f, indent=2)
                f.write("\n")

    def list_agents(self) -> List[Dict[str, Any]]:
        return list(self.__public_setups)

    def get_agent(self, identifier: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if identifier is None:
            return self.__public_setups[0] if self.__public_setups else None
        for setup in self.__public_setups:
            if setup.get("agent_identifier") == identifier:
                return setup
        return None

    def get_agent_or_default(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        agent = self.get_agent(identifier)
        if agent is None:
            raise ProjectConfigError(
                "No agent found"
                + (f" with identifier '{identifier}'" if identifier else "")
            )
        return agent

    def get_secret(self, identifier: str) -> Optional[Dict[str, Any]]:
        for setup in self.__secret_setups:
            if setup.get("agent_identifier") == identifier:
                return setup
        return None

    def add_agent(
        self,
        public_setup: Dict[str, Any],
        secret_setup: Optional[Dict[str, Any]] = None,
    ):
        identifier = public_setup["agent_identifier"]
        if self.get_agent(identifier) is not None:
            raise ProjectConfigError(f"Agent '{identifier}' already exists")
        self.__public_setups.append(public_setup)
        if secret_setup:
            self.__secret_setups.append(secret_setup)
        self.save()

    def update_agent(self, identifier: str, patch: Dict[str, Any]):
        for i, setup in enumerate(self.__public_setups):
            if setup.get("agent_identifier") == identifier:
                self.__public_setups[i] = _deep_merge(setup, patch)
                self.save()
                return
        raise ProjectConfigError(f"Agent '{identifier}' not found")

    def update_secret(self, identifier: str, patch: Dict[str, Any]):
        for i, setup in enumerate(self.__secret_setups):
            if setup.get("agent_identifier") == identifier:
                self.__secret_setups[i] = _deep_merge(setup, patch)
                self.save()
                return
        self.__secret_setups.append({"agent_identifier": identifier, **patch})
        self.save()

    def remove_secret_agent_configuration_key(self, identifier: str, key: str):
        for setup in self.__secret_setups:
            configuration = setup.get("agent_configuration") or {}
            if setup.get("agent_identifier") == identifier and key in configuration:
                del configuration[key]
                self.save()

    def remove_agent(self, identifier: str):
        self.__public_setups = [
            s for s in self.__public_setups if s.get("agent_identifier") != identifier
        ]
        self.__secret_setups = [
            s for s in self.__secret_setups if s.get("agent_identifier") != identifier
        ]
        self.save()

    def save_mcp_servers(self):
        with open(self.__mcp_path, "w") as f:
            json.dump(self.__mcp_setups, f, indent=2)
            f.write("\n")

    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        return list(self.__mcp_setups)

    def get_mcp_server(self, mcp_server_identifier: str) -> Optional[Dict[str, Any]]:
        for setup in self.__mcp_setups:
            if setup.get("mcp_server_identifier") == mcp_server_identifier:
                return setup
        return None

    def add_mcp_server(self, setup: Dict[str, Any]):
        identifier = setup["mcp_server_identifier"]
        if self.get_mcp_server(identifier) is not None:
            raise ProjectConfigError(f"MCP server '{identifier}' already exists")
        self.__mcp_setups.append(setup)
        self.save_mcp_servers()

    def remove_mcp_server(self, mcp_server_identifier: str):
        self.__mcp_setups = [
            s
            for s in self.__mcp_setups
            if s.get("mcp_server_identifier") != mcp_server_identifier
        ]
        self.save_mcp_servers()

    def list_mcp_configurations(self) -> List[Dict[str, Any]]:
        return list(self.__mcp_configurations)

    def get_mcp_configuration(self, name: str) -> Optional[Dict[str, Any]]:
        return next(
            (c for c in self.__mcp_configurations if c.get("name") == name), None
        )

    def set_mcp_configuration(self, configuration: Dict[str, Any]):
        self.__mcp_configurations = [
            c
            for c in self.__mcp_configurations
            if c.get("name") != configuration["name"]
        ] + [configuration]
        self.__save_mcp_configurations()

    def remove_mcp_configuration(self, name: str):
        self.__mcp_configurations = [
            c for c in self.__mcp_configurations if c.get("name") != name
        ]
        self.__save_mcp_configurations()

    def __save_mcp_configurations(self):
        with open(self.__mcp_configurations_path, "w") as f:
            json.dump(self.__mcp_configurations, f, indent=2)
            f.write("\n")

    def save_llm_configurations(self):
        with open(self.__llm_path, "w") as f:
            json.dump(self.__llm_configs, f, indent=2)
            f.write("\n")
        if self.__secret_llm_configs or os.path.isfile(self.__secret_llm_path):
            os.makedirs(os.path.dirname(self.__secret_llm_path), exist_ok=True)
            with open(self.__secret_llm_path, "w") as f:
                json.dump(self.__secret_llm_configs, f, indent=2)
                f.write("\n")

    def save_secrets(self):
        os.makedirs(os.path.dirname(self.__secrets_path), exist_ok=True)
        with open(self.__secrets_path, "w") as f:
            json.dump(self.__secrets, f, indent=2)
            f.write("\n")

    def secret_names(self) -> List[str]:
        return [secret["name"] for secret in self.__secrets if secret.get("name")]

    def get_secret_value(self, name: str) -> Optional[str]:
        return next(
            (
                secret.get("value")
                for secret in self.__secrets
                if secret.get("name") == name
            ),
            None,
        )

    def set_secret(self, name: str, value: str):
        for secret in self.__secrets:
            if secret.get("name") == name:
                secret["value"] = value
                break
        else:
            self.__secrets.append({"name": name, "value": value})
        self.save_secrets()

    def list_llm_configurations(self) -> List[Dict[str, Any]]:
        return list(self.__llm_configs)

    def get_llm_configuration(self, name: str) -> Optional[Dict[str, Any]]:
        return next((c for c in self.__llm_configs if c.get("name") == name), None)

    def __get_llm_secret(self, name: str) -> Optional[Dict[str, Any]]:
        return next(
            (c for c in self.__secret_llm_configs if c.get("name") == name), None
        )

    def find_llm_configuration(self, vendor: str, model_name: str) -> Optional[str]:
        for config in self.__llm_configs:
            if (
                config.get("vendor") == vendor
                and config.get("model_configuration", {}).get("name") == model_name
            ):
                return config.get("name")
        return None

    def set_llm_configuration(
        self,
        name: str,
        vendor: str,
        model_configuration: Dict[str, Any],
        vendor_config: Optional[Dict[str, Any]] = None,
    ):
        public = self.get_llm_configuration(name)
        if public is None:
            public = {"name": name}
            self.__llm_configs.append(public)
        public["vendor"] = vendor
        public["model_configuration"] = model_configuration
        if vendor_config:
            existing = public.get("vendor_configuration", {}).get(vendor, {})
            public["vendor_configuration"] = {
                vendor: self.__store_credentials(
                    _vendor_model(vendor), vendor_config, existing, _slug(name)
                )
            }
        self.save_llm_configurations()

    def __store_credentials(
        self,
        model: Optional[Type[BaseModel]],
        values: Dict[str, Any],
        existing: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """A credential field (one with a `<field>_ref` sibling on the model) is
        stored in env/secrets.json and referenced by name; typing the name of a
        stored secret references it; re-entering a value for a field that already
        has a ref rotates that secret. Everything else stays in the public file."""
        public: Dict[str, Any] = {}
        fields = model.model_fields if model is not None else {}
        for key, value in values.items():
            nested = _nested_model(fields.get(key))
            if nested is not None and isinstance(value, dict):
                public[key] = self.__store_credentials(
                    nested, value, existing.get(key, {}) or {}, f"{prefix}-{_slug(key)}"
                )
                continue
            if f"{key}_ref" not in fields:
                public[key] = value
                continue
            if value in (None, ""):
                if existing.get(f"{key}_ref"):
                    public[f"{key}_ref"] = existing[f"{key}_ref"]
                elif value == "":
                    public[key] = ""
                continue
            if value in self.secret_names():
                public[f"{key}_ref"] = value
                continue
            secret_name = existing.get(f"{key}_ref") or f"{prefix}-{_slug(key)}"
            self.set_secret(secret_name, value)
            public[f"{key}_ref"] = secret_name
        for key, value in existing.items():
            if (
                key.endswith("_ref")
                and key not in public
                and key[: -len("_ref")] not in values
            ):
                public[key] = value
        return public

    def get_llm_vendor_config(self, name: str, vendor: str) -> Dict[str, Any]:
        public = self.get_llm_configuration(name) or {}
        result = dict(public.get("vendor_configuration", {}).get(vendor, {}))
        secret = self.__get_llm_secret(name)
        if secret:
            for key, value in (
                secret.get("vendor_configuration", {}).get(vendor, {}).items()
            ):
                result.setdefault(key, value)
        return result

    def set_agent_llm_configuration(self, identifier: Optional[str], name: str):
        agent = self.get_agent_or_default(identifier)
        agent.pop("llm_client_configuration", None)
        agent["llm_configuration_name"] = name
        actual_id = agent.get("agent_identifier", identifier)
        secret = self.get_secret(actual_id) if actual_id else None
        if secret:
            secret.pop("llm_client_configuration", None)
        self.save()

    def agent_llm_configuration_name(
        self, identifier: Optional[str] = None
    ) -> Optional[str]:
        return self.get_agent_or_default(identifier).get("llm_configuration_name")

    def resolve_agent_llm(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        # Legacy inline config wins (read back-compat); otherwise resolve the agent's
        # named reference into a public {vendor, model_configuration} view.
        agent = self.get_agent_or_default(identifier)
        inline = agent.get("llm_client_configuration")
        if inline:
            return inline
        name = agent.get("llm_configuration_name")
        if name:
            config = self.get_llm_configuration(name)
            if config:
                return {
                    "vendor": config.get("vendor"),
                    "model_configuration": config.get("model_configuration", {}),
                }
        return {}

    def get_agent_config(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        agent = self.get_agent_or_default(identifier)
        return agent.get("agent_configuration", {})

    def set_agent_config(self, key: str, value: Any, identifier: Optional[str] = None):
        agent = self.get_agent_or_default(identifier)
        config = agent.setdefault("agent_configuration", {})
        config[key] = value
        self.save()

    def get_provider_config(
        self, provider_key: str, identifier: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        config = self.get_agent_config(identifier)
        return config.get(provider_key)

    def set_provider_config(
        self, provider_key: str, value: Dict[str, Any], identifier: Optional[str] = None
    ):
        self.set_agent_config(provider_key, value, identifier)

    def get_model_config(
        self, identifier: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        llm = self.resolve_agent_llm(identifier)
        model_cfg = llm.get("model_configuration", {})
        return llm.get("vendor"), model_cfg.get("name"), model_cfg

    def set_model(
        self,
        model_name: str,
        vendor: str,
        identifier: Optional[str] = None,
        vendor_config: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        # Edits the named LLM configuration the agent references; a legacy or
        # unconfigured agent is migrated onto a named config first.
        agent = self.get_agent_or_default(identifier)
        name = agent.get("llm_configuration_name")
        if name is None:
            name = self.find_llm_configuration(vendor, model_name) or agent.get(
                "agent_identifier", "default"
            )
            self.set_agent_llm_configuration(agent.get("agent_identifier"), name)
        existing = self.get_llm_configuration(name) or {}
        model_configuration = dict(existing.get("model_configuration", {}))
        model_configuration["name"] = model_name
        model_configuration.setdefault("type", "chat")
        model_configuration.setdefault("temperature", 0.0)
        if model_params:
            for key, value in model_params.items():
                if value is None:
                    model_configuration.pop(key, None)
                else:
                    model_configuration[key] = value
        self.set_llm_configuration(name, vendor, model_configuration, vendor_config)

    def get_vendor_config(
        self, vendor: str, identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        agent = self.get_agent_or_default(identifier)
        inline = agent.get("llm_client_configuration")
        if inline:
            result = dict(inline.get("vendor_configuration", {}).get(vendor, {}))
            actual_id = identifier or agent.get("agent_identifier", "agent")
            secret = self.get_secret(actual_id) or {}
            for key, value in (
                secret.get("llm_client_configuration", {})
                .get("vendor_configuration", {})
                .get(vendor, {})
                .items()
            ):
                result.setdefault(key, value)
            return result
        name = agent.get("llm_configuration_name")
        return self.get_llm_vendor_config(name, vendor) if name else {}

    def get_platform_config(self) -> Optional[Dict[str, Any]]:
        config_path = os.path.join(self.project_dir, PLATFORM_CONFIG_FILE)
        if not os.path.isfile(config_path):
            return None
        with open(config_path, "r") as f:
            return json.load(f)

    def platform_logins(self) -> Dict[str, Dict[str, Any]]:
        """Every login by tenant, the current one included."""
        platform = self.get_platform_config() or {}
        logins = dict(platform.get("logins") or {})
        if platform.get("tenant"):
            logins[platform["tenant"]] = {
                "base_url": platform.get("base_url", ""),
                "api_key": platform.get("api_key", ""),
            }
        return logins

    def platform_agent_configuration(
        self, tenant: Optional[str] = None
    ) -> Dict[str, Any]:
        """The agent_configuration keys a local agent gets from env/zav_config.json:
        the login of its tenant, else the current login. The developer's platform
        key never lives in a setup."""
        platform = self.get_platform_config() or {}
        login = self.platform_logins().get(tenant or "")
        if login is not None:
            platform = {**login, "tenant": tenant}
        return {
            key: platform[source]
            for key, source in (
                ("x_auth", "api_key"),
                ("base_url", "base_url"),
                ("tenant", "tenant"),
            )
            if platform.get(source)
        }

    def set_platform_config(self, base_url: str, api_key: str, tenant: str):
        """Makes ``tenant`` the current login; other tenants' logins stay under
        ``logins``. Empty ``tenant`` and ``api_key`` reset the file."""
        logins = self.platform_logins() if tenant or api_key else {}
        logins.pop(tenant, None)
        current = {"base_url": base_url, "api_key": api_key, "tenant": tenant}
        self.__write_platform_config(
            {**current, "logins": logins} if logins else current
        )

    def use_platform_login(self, tenant: str):
        logins = self.platform_logins()
        if tenant not in logins:
            raise ProjectConfigError(f"No platform login for tenant '{tenant}'")
        self.set_platform_config(tenant=tenant, **logins[tenant])

    def __write_platform_config(self, platform: Dict[str, Any]):
        config_path = os.path.join(self.project_dir, PLATFORM_CONFIG_FILE)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(platform, f, indent=4)

    def list_skills(self) -> List[str]:
        dir_path = os.path.join(self.project_dir, SKILLS_DIR)
        if not os.path.isdir(dir_path):
            return []
        return sorted(
            d
            for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))
            and os.path.isfile(os.path.join(dir_path, d, "Skill.md"))
            and not d.startswith(".")
        )

    def list_instructions(self) -> List[str]:
        return self.__list_files(INSTRUCTIONS_DIR, ".md")

    def list_specs(self) -> List[str]:
        return self.__list_files(SPECS_DIR, ".yaml")

    def __list_files(self, subdir: str, ext: str) -> List[str]:
        dir_path = os.path.join(self.project_dir, subdir)
        if not os.path.isdir(dir_path):
            return []
        return sorted(
            f for f in os.listdir(dir_path) if f.endswith(ext) and not f.startswith(".")
        )

    @staticmethod
    def is_valid_project(directory: str) -> bool:
        config_path = os.path.join(directory, CONFIG_FILE)
        if os.path.isfile(config_path):
            return True
        init_file = os.path.join(directory, "__init__.py")
        if os.path.isfile(init_file):
            with open(init_file, "r") as f:
                content = f.read()
                return "Zeta Alpha Agents SDK" in content or "zav.agents_sdk" in content
        return False


class ProjectConfigError(Exception):
    pass


def _slug(text: str) -> str:
    return "-".join(part for part in re.split(r"[^a-z0-9]+", text.lower()) if part)


def _unwrap_optional(annotation: Any) -> Any:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
    return args[0] if len(args) == 1 else annotation


def _nested_model(field: Any) -> Optional[Type[BaseModel]]:
    if field is None:
        return None
    annotation = _unwrap_optional(field.annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _vendor_model(vendor: str) -> Optional[Type[BaseModel]]:
    return _nested_model(LLMVendorConfiguration.model_fields.get(vendor))


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
