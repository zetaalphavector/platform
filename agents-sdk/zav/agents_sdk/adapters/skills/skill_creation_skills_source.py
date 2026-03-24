from typing import ClassVar, Dict, List

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_SKILL_NAME = "create-skill"

_SKILL_DESCRIPTION = (
    "Instructions for creating new agent skills (SKILL.md files). "
    "Use when asked to create a skill, add a skill, scaffold a skill, "
    "package domain knowledge, or save a workflow as a reusable skill."
)

_SKILL_BODY = """\
## Skill Creation

You have access to the `create_skill` tool for persisting \
new reusable skills. Once created, skills appear in the \
skill catalog and can be activated via `use_skill`.

### When to create a skill

Create a skill when:
- A workflow has **multiple steps** that need to be \
executed in a specific order.
- Domain knowledge is needed that the agent wouldn't \
have by default (conventions, templates, checklists).
- The same task is likely to be **repeated** — by the \
same user or across a team.

Do NOT create a skill when:
- The task is a one-off that won't recur.
- A simple instruction ("always use X pattern") would \
suffice — that belongs in workspace instructions, \
not a skill.
- The body would just be a single sentence — a skill \
should carry enough guidance to meaningfully change \
agent behavior.

### `create_skill`

Use this tool to save a new skill. Provide:
- `name`: a unique, lowercase, hyphen-separated identifier \
(e.g. `generate-api-endpoint`). Must match the `name` field \
in the frontmatter.
- `content`: the full SKILL.md document (frontmatter + body).

### SKILL.md format

The content must follow this structure:

```
---
name: <skill-name>
description: '<one-line summary with trigger phrases and keywords>'
---

<markdown instruction body>
```

#### Frontmatter

- `name` (required): must match the `name` parameter.
- `description` (required): the **discovery surface** — \
this is how the agent decides whether to activate the skill. \
If the user's trigger phrases aren't in the description, \
the skill won't be found. Follow this pattern: \
`'Brief summary. Use when asked to "phrase A", "phrase B". \
Keywords: kw1, kw2.'`
- Always quote descriptions containing colons.

#### Body

Write as **instructions to the agent**, not documentation. \
Use imperative style ("Do X", "Generate Y"). Structure \
with step-by-step workflows, decision points, constraints, \
and anti-patterns to avoid. Reference companion files \
with `read_skill_resource` if the skill bundles templates.

Good bodies include:
- A **workflow** section with numbered steps.
- **Decision guidance** — "if X, do Y; if Z, do W".
- **Edge cases** — situations that are easy to get wrong.
- **Pitfalls** — common mistakes and how to avoid them.
- **Examples** — concrete before/after or input/output.

Avoid bodies that just restate what the description says, \
or that read like API documentation. The body should make \
the agent *better at the task* than it would be without it.

### Common pitfalls

**Description is the discovery surface.** If trigger \
phrases aren't in the description, the agent won't find \
the skill. Use the "Use when asked to..." pattern with \
specific verbs and nouns the user would actually type.

**Vague descriptions waste the skill.** \
Bad: `description: Handles code generation` — too broad, \
matches everything, helps nothing. \
Good: `description: 'Generates FastAPI CRUD endpoints \
with tests. Use when asked to "create an endpoint", \
"add a REST resource", or "scaffold an API". Keywords: \
REST, CRUD, endpoint.'`

**YAML frontmatter fails silently.** Unescaped colons in \
description values, tabs instead of spaces, or `name` \
that doesn't match the parameter — all cause silent \
failures with no error message. Always quote descriptions.

**Overly generic body.** A body that says "follow best \
practices" or "write clean code" adds nothing. Include \
the *specific* knowledge the agent needs: exact file \
paths, naming conventions, structural patterns, decision \
criteria.

### Example

```
---
name: generate-api-endpoint
description: 'Generates FastAPI CRUD endpoints with tests. \
Use when asked to "create an endpoint", "add a REST resource", \
or "scaffold an API". Keywords: REST, CRUD, endpoint, FastAPI.'
---

# Generate API Endpoint

## Workflow

1. Ask the user for the resource name and fields.
2. Read `assets/controller_template.py` via `read_skill_resource`.
3. Adapt the template to the resource.
4. Create the controller, handler, and test files.
5. Register the new route in the app router.

## Edge cases

- If the resource name conflicts with an existing route, \
ask the user before overwriting.
- If the user requests nested resources (e.g. `/users/:id/posts`), \
create a separate sub-router.

## Constraints

- Follow existing project conventions for file placement.
- Use the project's established error handling pattern.
- Do not add dependencies without asking.
```\
"""


class SkillCreationSkillsSource(SkillsSource):
    """Provides instructions for creating new skills as a discoverable skill."""

    source_name: ClassVar[str] = "skill_creation_skills"

    async def discover(self) -> Dict[str, SkillProperties]:
        return {
            _SKILL_NAME: SkillProperties(
                name=_SKILL_NAME,
                description=_SKILL_DESCRIPTION,
            ),
        }

    async def get_skill_body(self, skill_name: str) -> str:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        return _SKILL_BODY

    async def get_resources(self, skill_name: str) -> List[str]:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        return []

    async def read_resource(self, skill_name: str, path: str) -> str:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        available_resources = await self.get_resources(skill_name)
        raise SkillResourceNotFoundError(skill_name, path, available_resources)


class SkillCreationSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(cls) -> SkillCreationSkillsSource:
        return SkillCreationSkillsSource()
