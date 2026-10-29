from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.rubric import Rubric, RubricSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_FRONTMATTER_DELIMITER = "---"


class DiskRubricSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Load rubrics from markdown files.")
    rubrics_directories: List[str] = Field(
        default_factory=lambda: ["rubrics"],
        description=(
            "Directories of `<name>.md` files: YAML frontmatter with `match` and "
            "optional `task`, `criteria`, `pass_threshold`; the body is the "
            "reference answer."
        ),
    )


def parse_rubric_file(path: Path) -> Rubric:
    frontmatter, body = split_frontmatter(path.read_text())
    return Rubric(name=path.stem, reference=body.strip(), **frontmatter)


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        raise ValueError("rubric files start with a `---` frontmatter block")
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONTMATTER_DELIMITER:
            frontmatter = yaml.safe_load("\n".join(lines[1:index])) or {}
            if not isinstance(frontmatter, dict):
                raise ValueError("rubric frontmatter must be a mapping")
            return frontmatter, "\n".join(lines[index + 1 :])
    raise ValueError("unterminated rubric frontmatter")


class DiskRubricSource(RubricSource):
    source_name = "disk"

    def __init__(self, directories: List[str], enabled: bool) -> None:
        self.enabled = enabled
        self.__directories = [Path(d) for d in directories]

    async def list_rubrics(self) -> List[Rubric]:
        rubrics: List[Rubric] = []
        for directory in self.__directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                try:
                    rubrics.append(parse_rubric_file(path))
                except Exception as e:
                    logger.error(f"Skipping rubric {path}: {e}")
        return rubrics


class DiskRubricSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        disk_rubric_source_configuration: DiskRubricSourceConfiguration = (
            DiskRubricSourceConfiguration()
        ),
    ) -> DiskRubricSource:
        return DiskRubricSource(
            directories=disk_rubric_source_configuration.rubrics_directories,
            enabled=disk_rubric_source_configuration.enabled,
        )
