import hashlib
import importlib
import io
import os
import zipfile
from typing import List, Optional

from pydantic import BaseModel


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
            import sys

            sys.path.append("..")
            importlib.import_module(os.path.basename(project_dir))
        else:
            import sys

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
        _load_python_module(project_dir)

    @classmethod
    def from_project_dir(cls, project: str, agent_names: List[str], project_dir: str):
        filestream = io.BytesIO()
        with zipfile.ZipFile(
            filestream, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zip_ref:
            for root, _, files in os.walk(project_dir):
                if os.path.basename(root) == "__pycache__":
                    continue
                if os.path.basename(root) == "env":
                    continue

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
