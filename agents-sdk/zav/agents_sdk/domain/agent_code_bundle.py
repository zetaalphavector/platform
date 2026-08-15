import hashlib
import importlib
import io
import os
import sys
import zipfile
from typing import List, Optional

from zav.pydantic_compat import BaseModel


def __evict_module_from_cache(module_name: str):
    stale = [
        key
        for key in sys.modules
        if key == module_name or key.startswith(module_name + ".")
    ]
    for key in stale:
        del sys.modules[key]


def _load_python_module(project_dir: str):
    # This function imports the python module from the project directory.
    # It is assumed that the files under project dir were created by the CLI,
    # therefore by doing this operation we're importing ChatAgentClassRegistry and
    # AgentDependencyRegistry classes.
    import_str = project_dir.lstrip("/").replace("/", ".")
    module_str, _, _ = import_str.partition(":")
    if not module_str:
        raise Exception(
            f"Import string {import_str} must be in format <module> "
            "or <module>:<attribute>."
        )
    try:
        if project_dir == os.getcwd():
            target_module = os.path.basename(project_dir)
            __evict_module_from_cache(target_module)
            sys.path.append("..")
            importlib.import_module(target_module)
        else:
            __evict_module_from_cache(module_str)
            if os.path.isabs(project_dir):
                sys.path.append("/")
            else:
                sys.path.append(".")
            importlib.import_module(module_str)
    except ImportError as exc:
        if exc.name != module_str:
            raise exc from None
        raise Exception(f"Could not import module {module_str}.")


class AgentCodeBundle(BaseModel):
    project: str
    agent_names: List[str]
    agent_bundle: bytes

    def store_on_disk(
        self, base_path: str = "dynamic_agents", load_registries: bool = False
    ):
        bundle_hash = hashlib.sha256(self.agent_bundle).hexdigest()
        os.makedirs(base_path, exist_ok=True)
        # Unzip the agent bundle. The files will be in {base_path}/{self.project}
        with io.BytesIO(self.agent_bundle) as bytes_io:
            with zipfile.ZipFile(bytes_io) as zip_file:
                for zip_info in zip_file.filelist:
                    if zip_info.filename.startswith(f"{self.project}/"):
                        zip_info.filename = zip_info.filename.replace(
                            f"{self.project}/", f"{self.project}-{bundle_hash}/"
                        )
                    zip_file.extract(zip_info, base_path)
        if load_registries:
            self.load_agent_registries_from(
                project=f"{self.project}-{bundle_hash}", base_path=base_path
            )

    @classmethod
    def load_agent_registries_from(
        cls,
        project_dir: Optional[str] = None,
        project: Optional[str] = None,
        base_path: str = "dynamic_agents",
    ):
        if project_dir is None:
            if project is None:
                raise ValueError("Either project or project_dir must be provided.")
            project_dir = os.path.join(base_path, project)

        import zav.agents_sdk.agents  # noqa: F401

        if os.path.isfile(os.path.join(project_dir, "__init__.py")):
            _load_python_module(project_dir)

    @classmethod
    def from_project_dir(cls, project: str, agent_names: List[str], project_dir: str):
        filestream = io.BytesIO()
        with zipfile.ZipFile(
            filestream, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zip_ref:
            excluded_dirs = {"__pycache__", "env", "memories", "build"}
            for root, dirs, files in os.walk(project_dir):
                # Prune in-place so os.walk does not descend into excluded
                # trees — a basename-only `continue` still recurses into their
                # subdirs, which is how local run state (the default
                # `.agent-state`/`.perf-storage` storage paths and their nested
                # agent-traces/checkpoints) leaked into uploaded bundles. Also
                # drop any dot-dir (.git, .venv, .agent-state, …) so on-disk
                # run state never ships with the agent code.
                dirs[:] = [
                    d for d in dirs if d not in excluded_dirs and not d.startswith(".")
                ]

                for filename in files:
                    if filename.endswith(".pyc"):
                        continue
                    if filename.endswith(".pyo"):
                        continue
                    if filename == ".gitignore":
                        continue
                    if filename == "agent_setups.json":
                        continue
                    file_path = os.path.join(root, filename)
                    # When unzipping, the project directory with be the project name
                    relative_path = os.path.join(
                        project, os.path.relpath(file_path, project_dir)
                    )
                    zip_ref.write(file_path, relative_path)

        return cls(
            project=project,
            agent_names=agent_names,
            agent_bundle=filestream.getvalue(),
        )
