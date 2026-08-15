from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field, validator

from zav.agents_sdk.adapters.registration.registration_service import (
    RegistrationService,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, streamable


class OnboardingStep(BaseModel):
    title: str = Field(
        description="Short label shown for the step (e.g. 'Create a tag')."
    )
    content: str = Field(
        description="Behavioral instructions the agent follows for this step."
    )
    status_label: Optional[str] = Field(
        None,
        description=(
            "Progress-bar label shown while the user is on this step — a short, "
            "friendly action phrase for what they're doing now (e.g. 'Create a "
            "tag'), not a past-tense completion. Falls back to ``title`` when "
            "unset. The first step's label is never shown (it's the warm-up "
            "where onboarding begins)."
        ),
    )


DEFAULT_ONBOARDING_STEPS: List[OnboardingStep] = [
    OnboardingStep(
        title="Role",
        content=(
            "Ask their role/job title in a warm, conversational way — e.g. "
            '"Before we dive in — what\'s your role or job title?" Save to '
            "memory, do NOT create tag yet. Move on."
        ),
    ),
    OnboardingStep(
        title="Create a tag",
        content=(
            "- Conversationally tell the user you'll remember their role and "
            "interests to personalize the platform experience for them.\n"
            '- Explain what tags are: "Tags are collections of documents that '
            "you can organize by project, topic, or anything you like. You can "
            "search within them, add notes, chat with documents inside a tag, "
            'and share them with teammates."\n'
            "- Ask them what topic or project they're working on.\n"
            "- Create a tag based on the topic or project (don't include the "
            "user's name in the tag name).\n"
            "- Do ONE search on the platform index. If there are 0 relevant "
            "documents, do a web search. Present 3-5 relevant documents to the "
            "user and ask if they want to add them to the tag. If yes, add "
            "them. If no, move on to the next step.\n"
            "- Tell them how to do it themselves: create tags from the **Tags** "
            "section in the left sidebar; add documents via the tag/bookmark "
            "icon on search results.\n"
            "- Mention they'll receive recommendations based on this tag, and "
            "they can modify/remove it anytime.\n"
            "- Update their topic/interests to memory."
        ),
    ),
    OnboardingStep(
        title="Create a skill",
        content=(
            "- Tell the user creating a skill is the last step of onboarding.\n"
            '- Explain what skills are: "Skills are reusable instruction sets '
            "that any agent can use on demand. They're great for automating "
            "repetitive tasks. You can create them from any conversation (just "
            "tell any agent to create one!), and call them by name in future "
            'chats. You can also edit or delete them later."\n'
            "- Ask the user to describe a repeating workflow in one or two "
            'sentences. Give some examples ("e.g. ...") relevant to their role '
            'and current topic, but don\'t mention "weekly" or "daily" tasks. '
            "Those belong to scheduled agents, not skills. Don't ask follow-up "
            "questions. Use your best judgment to fill in details. Immediately "
            "create the skill — never ask permission.\n"
            "- Don't do searches and don't look into documents or tags! This "
            "slows down the process. The skill should be based on the skill "
            "workflow description, not on specific documents or tags.\n"
            "- Summarize the skill — show its name and a brief description of "
            "what it does."
        ),
    ),
    OnboardingStep(
        title="Summarize and handoff",
        status_label="Wrap Up",
        content=(
            "Announce completion:\n"
            '- Say something like "This concludes the onboarding! You now have '
            "a tag with relevant documents, a skill to automate a workflow, and "
            "a profile in memory that helps me help you better in future chats. "
            "Here's a quick summary:\"\n"
            "Explain:\n"
            "- **Discover mode**: search the platform's index for internal "
            "data.\n"
            "- **Chat with documents or tags**: open any document or tag and "
            "ask questions about it.\n"
            "- **Private documents**: upload PDFs or reports to make them "
            "searchable alongside the main index.\n"
            "- **Deep research**: run background multi-source investigations "
            "that produce structured reports — for broad or high-stakes "
            "questions.\n"
            "Have a section 'What's next?' with suggestions for skills or "
            "agents:\n"
            "- Suggest 2–3 specific skills or custom agents that would "
            "realistically help the user based on what you know about them, "
            "their company, and their role. Showcase how the platform's "
            "capabilities (skills, memories, custom/scheduled agents, tags) can "
            "work together.\n"
            "- Note that scheduled tasks always require scheduled agents, not "
            "skills.\n"
            "- If you suggest to create a custom or scheduled agent, explain "
            'how the user should set it up, e.g. "To set up this agent, go to '
            "the Agent Settings section in the left sidebar, click 'Create Your "
            "Agent', and follow the instructions. You can choose which data it "
            "has access to, what skills it has, set up a custom schedule, and "
            'share it with your team."\n'
            "Handoff:\n"
            '- Start new conversations with the **"+" button** at the top of '
            "the chat list.\n"
            "- They can return to this chat for platform questions anytime.\n"
            '- End with the slogan "**Enjoy discovery!**"'
        ),
    ),
]

_ONBOARDING_PREAMBLE = """You are the onboarding guide for a knowledge management platform. Your goal is for the user to quickly experience value — not to be lectured about features. These instructions should override your default behavior.

### Platform features (reference)

- **Tags**: curated document collections by project/topic. Users can search within them, attach notes, and share with colleagues.
- **Memory**: persistent across sessions. Save the user's name, role, domain, team, interests — anything that makes future interactions faster.
- **Private documents**: upload PDFs, reports, etc. to make them searchable alongside the main index.
- **Deep research**: background multi-source investigation producing structured reports. For broad/high-stakes questions, not quick lookups.
- **Skills**: reusable instruction sets that change agent behavior for specific tasks. Created from conversation.

### Onboarding rules

1. **One question per message.** Never combine questions. Each must be answerable in a short sentence.
2. **Pattern: ask → do → explain.** Ask for input, immediately perform the action (never ask permission), then explain what you did and how the user can do it themselves. Before doing something, explain what it is and how it fits into the platform — always tie it back to how the user can use it themselves in the future.
3. **Don't ask too many questions.** The user is new and doesn't know what they want to do on the platform. Demonstrate features by doing them.
4. **Don't put multiple steps in one message.** Don't present all steps at once or use numbered lists. Flow conversationally. One step may consist of multiple messages, but there cannot be multiple steps in one message.
5. **Be brief and friendly.** Use emojis sparingly, markdown formatting. Don't over-explain. Don't use too many tool calls, keep it simple. Speed over perfection, the examples here are just demo flows.
6. Throughout, save or update relevant user facts to memory. Only store user profile-related info (name, role, interests, domain). IMPORTANT: don't store duplicate information! Don't overdo the memory updates, call this max 2 times during onboarding.
7. **Report progress.** After completing a step, call the `update_onboarding_status` tool exactly as that step instructs — passing the number of the step the user is **now moving to** and a short label naming **that step** (what they're about to do). The number tells the UI which step the user is on, so do NOT skip this call. The label is **shown to the user** in the progress bar as the current step, so it must be a short, friendly action phrase (e.g. 'Create a tag', 'Create a skill') — never a past-tense completion like 'Tag created', and never a tool or function name like `create_tag`. The first step is the warm-up and is not reported; reaching step {total_steps} marks onboarding complete."""  # noqa: E501

_ONBOARDING_POSTAMBLE = """IMPORTANT: The final step's completion message MUST also be the onboarding wrap-up message. Do NOT split this into multiple turns — deliver the final step summary and hand-off all in one response.

### Boundaries

Do NOT answer domain/subject-matter questions. Redirect: "Great question! Start a new chat with the main assistant using the + button."

After all onboarding steps are complete, serve as platform support assistant."""  # noqa: E501


def _build_base_prompt(
    steps: List[OnboardingStep], instructions: Optional[str] = None
) -> str:
    total_steps = len(steps)
    preamble = _ONBOARDING_PREAMBLE.format(total_steps=total_steps)
    rendered = []
    for index, step in enumerate(steps, start=1):
        block = f"**Step {index} — {step.title}.**\n\n{step.content}"
        # On completing a step, report the *next* step's number and label — the user's
        # current action, not the step just finished. The last step reports nothing.
        if index < total_steps:
            next_step = steps[index]
            label = next_step.status_label or next_step.title
            block += (
                "\n\nWhen this step is done, call "
                f'`update_onboarding_status(step={index + 1}, label="{label}")`.'
            )
        rendered.append(block)
    rendered_steps = "\n\n".join(rendered)
    steps_section = f"### Steps\n\n{rendered_steps}"
    if instructions:
        steps_section = (
            f"### Additional instructions\n\n{instructions}\n\n{steps_section}"
        )
    return f"{preamble}\n\n{steps_section}\n\n{_ONBOARDING_POSTAMBLE}\n"


class OnboardingToolsSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Enable onboarding progress tracking and guidance. Meant to be "
            "switched on per conversation via runtime bot params, not in the "
            "agent configuration: an explicit value in the agent configuration "
            "takes precedence over the runtime value, enabling onboarding for "
            "every conversation (true) or disabling it entirely (false). Set "
            "steps/instructions in the agent configuration and leave this unset."
        ),
    )
    steps: List[OnboardingStep] = Field(
        default_factory=lambda: list(DEFAULT_ONBOARDING_STEPS),
        description=(
            "Ordered onboarding steps. Override to customize the flow per "
            "tenant. The total number of steps is derived from this list."
        ),
    )
    instructions: Optional[str] = Field(
        None,
        description=(
            "Tenant-specific behavioral instructions inserted before the "
            "steps. Use for cross-cutting rules that apply throughout the "
            "whole flow (e.g. language, terminology overrides)."
        ),
    )

    @validator("steps")
    @classmethod
    def _steps_not_empty(cls, v: List[OnboardingStep]) -> List[OnboardingStep]:
        if not v:
            raise ValueError("steps must contain at least one step")
        return v


class OnboardingToolsSource(ToolsSource):
    """Provides the progress-tracking tool and injects onboarding guidance into
    the prompt. ``steps`` is the single source of truth: the prompt renders them
    and the tool derives its step count from the list length."""

    source_name = "onboarding"

    def __init__(
        self,
        enabled: bool,
        registration_service: RegistrationService,
        steps: Optional[List[OnboardingStep]] = None,
        instructions: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.__steps = steps if steps is not None else list(DEFAULT_ONBOARDING_STEPS)
        self.__total_steps = len(self.__steps)
        self.__base_prompt = _build_base_prompt(self.__steps, instructions)
        self.__registration_service = registration_service

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []

        return [
            Tool.from_callable(
                name="update_onboarding_status",
                executable=self.__update_onboarding_status,
                description=(
                    "Report onboarding step completion. Call this tool "
                    "AFTER completing each onboarding step so the UI "
                    "can track progress.\n\n"
                    "Args:\n"
                    "    step: The step the user is now on (1-based). After "
                    "finishing a step, pass the next step's number; reaching "
                    "the final step number marks onboarding complete.\n"
                    "    label: A short, friendly action phrase shown to the "
                    "user in the progress bar, naming the step they are now on "
                    "(e.g. 'Create a tag', 'Create a skill') — not a past-tense "
                    "completion like 'Tag created', and never a tool or "
                    "function name like 'create_tag'."
                ),
            ),
        ]

    @streamable(
        running_text="Updating onboarding progress...",
        completed_text="Onboarding step {{ step }}/{{ total_steps }}: {{ label }}",
    )
    async def __update_onboarding_status(self, step: int, label: str) -> Dict[str, Any]:
        clamped = max(1, min(step, self.__total_steps))
        if clamped != step:
            logger.warning(
                f"update_onboarding_status: step={step} is out of range "
                f"[1, {self.__total_steps}], clamped to {clamped}."
            )
        return {
            "step": clamped,
            "total_steps": self.__total_steps,
            "label": label,
            "completed": clamped >= self.__total_steps,
        }

    async def __fetch_user_profile(self) -> str:
        try:
            user_info = await self.__registration_service.get_user_info()
        except Exception as e:
            logger.warning(f"Failed to fetch user profile for onboarding: {e}")
            return ""

        parts = []
        if user_info.first_name or user_info.last_name:
            name = f"{user_info.first_name} {user_info.last_name}".strip()
            parts.append(f"Name: {name}")
        if user_info.email:
            parts.append(f"Email: {user_info.email}")
        if user_info.affiliation:
            parts.append(f"Affiliation: {user_info.affiliation}")

        if not parts:
            return ""
        return "## User profile\n\n" + "\n".join(parts)

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""

        sections = [f"## Onboarding guidance\n\n{self.__base_prompt}"]

        profile = await self.__fetch_user_profile()
        if profile:
            sections.append(profile)

        return "\n\n".join(sections)


class OnboardingToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        registration_service: RegistrationService,
        onboarding_tools_source_configuration: OnboardingToolsSourceConfiguration = (
            OnboardingToolsSourceConfiguration()
        ),
    ) -> OnboardingToolsSource:
        return OnboardingToolsSource(
            enabled=onboarding_tools_source_configuration.enabled,
            registration_service=registration_service,
            steps=onboarding_tools_source_configuration.steps,
            instructions=onboarding_tools_source_configuration.instructions,
        )
