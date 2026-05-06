import json
import os
from typing import Any, Dict, List, Optional, Tuple

SKILLS_DIR = "skills"
INSTRUCTIONS_DIR = "instructions"
MEMORIES_DIR = "memories"
SPECS_DIR = "specs"
ENV_DIR = "env"
CONFIG_FILE = "agent_setups.json"
SECRET_CONFIG_FILE = os.path.join(ENV_DIR, CONFIG_FILE)
PLATFORM_CONFIG_FILE = os.path.join(ENV_DIR, "zav_config.json")
PROJECT_DIRS = [SKILLS_DIR, MEMORIES_DIR, SPECS_DIR]


class ProjectConfig:
    def __init__(self, project_dir: str):
        self.project_dir = os.path.abspath(project_dir)
        self.__public_path = os.path.join(self.project_dir, CONFIG_FILE)
        self.__secret_path = os.path.join(self.project_dir, SECRET_CONFIG_FILE)
        self.__public_setups: List[Dict[str, Any]] = []
        self.__secret_setups: List[Dict[str, Any]] = []
        self.__load()

    def __load(self):
        if os.path.isfile(self.__public_path):
            with open(self.__public_path, "r") as f:
                self.__public_setups = json.load(f)
        if os.path.isfile(self.__secret_path):
            with open(self.__secret_path, "r") as f:
                self.__secret_setups = json.load(f)

    def save(self):
        with open(self.__public_path, "w") as f:
            json.dump(self.__public_setups, f, indent=2)
            f.write("\n")
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

    def remove_agent(self, identifier: str):
        self.__public_setups = [
            s for s in self.__public_setups if s.get("agent_identifier") != identifier
        ]
        self.__secret_setups = [
            s for s in self.__secret_setups if s.get("agent_identifier") != identifier
        ]
        self.save()

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
        agent = self.get_agent_or_default(identifier)
        llm = agent.get("llm_client_configuration", {})
        vendor = llm.get("vendor")
        model_cfg = llm.get("model_configuration", {})
        model_name = model_cfg.get("name")
        return vendor, model_name, model_cfg

    def set_model(
        self,
        model_name: str,
        vendor: str,
        identifier: Optional[str] = None,
        vendor_config: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        agent = self.get_agent_or_default(identifier)
        llm = agent.setdefault("llm_client_configuration", {})
        llm["vendor"] = vendor
        llm.pop("vendor_configuration", None)
        model_cfg = llm.setdefault("model_configuration", {})
        model_cfg["name"] = model_name
        model_cfg.setdefault("type", "chat")
        model_cfg.setdefault("temperature", 0.0)
        if model_params:
            for key, value in model_params.items():
                if value is None:
                    model_cfg.pop(key, None)
                else:
                    model_cfg[key] = value
        if vendor_config is not None:
            actual_id = identifier or agent.get("agent_identifier", "agent")
            secret = self.get_secret(actual_id)
            if secret:
                secret.setdefault("llm_client_configuration", {})[
                    "vendor_configuration"
                ] = {vendor: vendor_config}
            else:
                self.__secret_setups.append(
                    {
                        "agent_identifier": actual_id,
                        "llm_client_configuration": {
                            "vendor_configuration": {vendor: vendor_config}
                        },
                    }
                )
            self.save()
        else:
            self.save()

    def get_vendor_config(
        self, vendor: str, identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        agent = self.get_agent_or_default(identifier)
        actual_id = identifier or agent.get("agent_identifier", "agent")
        secret = self.get_secret(actual_id)
        if secret:
            return (
                secret.get("llm_client_configuration", {})
                .get("vendor_configuration", {})
                .get(vendor, {})
            )
        return {}

    def get_platform_config(self) -> Optional[Dict[str, Any]]:
        config_path = os.path.join(self.project_dir, PLATFORM_CONFIG_FILE)
        if not os.path.isfile(config_path):
            return None
        with open(config_path, "r") as f:
            return json.load(f)

    def set_platform_config(self, base_url: str, api_key: str, tenant: str):
        config_path = os.path.join(self.project_dir, PLATFORM_CONFIG_FILE)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(
                {"base_url": base_url, "api_key": api_key, "tenant": tenant},
                f,
                indent=4,
            )

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


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
