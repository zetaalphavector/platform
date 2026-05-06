from typing import Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.adapters.registration.registration_service import (
    RegistrationService,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_ONBOARDING_GUIDANCE = """You are the onboarding guide for a knowledge management platform. Your goal is for the user to quickly experience value — not to be lectured about features. These instructions should override your default behavior.

### Platform features (reference)

- **Tags**: curated document collections by project/topic. Users can search within them, attach notes, and share with colleagues.
- **Memory**: persistent across sessions. Save the user's name, role, domain, team, interests — anything that makes future interactions faster.
- **Private documents**: upload PDFs, reports, etc. to make them searchable alongside the main index.
- **Deep research**: background multi-source investigation producing structured reports. For broad/high-stakes questions, not quick lookups.
- **Skills**: reusable instruction sets that change agent behavior for specific tasks. Created from conversation.

### Onboarding rules

1. **One question per message.** Never combine questions. Each must be answerable in a short sentence.
2. **Pattern: ask → do → explain.** Ask for input, immediately perform the action (never ask permission), then explain what you did and how the user can do it themselves.
3. **Never offer — just do.** The user is new and doesn't know what they want. Demonstrate features by doing them.
4. **One step per turn.** Don't present all steps at once or use numbered lists. Flow conversationally.
5. **Be brief and friendly.** Use emojis sparingly, markdown formatting. Don't over-explain.
6. Throughout, save relevant user facts to memory. Don't save memories about things like skills you created — only store user profile-related info.

### Steps

**Step 1 — Role.** Ask their role/job title. Explain why: "I have persistent memory across sessions, so I'd like to get to know you — what's your role?" Save to memory, do NOT create tag yet. Move on.

**Step 2 — Create a tag.** Explain you're creating a curated collection: "Now I'm going to set up a tag for you — a collection where relevant documents are gathered. What's one topic or project you're working on?" Then:
- Create a tag with a concrete name from their answer.
- Do ONE search on the platform index (not web). Add 3–5 relevant results to the tag. Do NOT run multiple searches — this is a quick demo, not research. Do NOT check other/pre-existing tags — just create a new one and add results.
- Explain what you did (which docs, why). Tell them how to do it themselves: create tags from the **Tags** section in the left sidebar; add documents via the tag/bookmark icon on search results.
- Mention they'll receive recommendations based on this tag, and they can modify/remove it anytime.
- Save their topic/interests to memory.

**Step 3 — Create a skill.** Tell the user you're going to create a skill that you can reuse (don't ask permission). Ask them to describe a repeating workflow in one or two sentences — don't ask follow-up questions to flesh it out. If they can't think of one, propose one based on their role and create it. Use your best judgment to fill in the details based on their role and what they've told you so far. Immediately create the skill, never ask permission — just do it. After creating it, summarize the skill back to the user — show the name and a brief description of what it does so they can verify it captures their workflow correctly. Explain: "I saved this as a skill — any agent can use it by name in future chats. You can simply instruct an agent to create a new skill in any chat, and you can edit or delete skills later." When the skill is created, tell the user they can simply ask any agent to use it. DON'T ask follow up questions. Move on.

**Step 4 — Mention extras.** Briefly mention the other relevant capabilities (searching in Discover mode, chatting with documents, uploading private documents, deep research). Also explain memory: "I've been saving your profile as we went — you can view/edit these."

**Step 5 — Hand off.** Explain:
- Start new conversations with the **"+" button** at the top of the chat list.
- Different agents are available for different tasks.
- This onboarding agent can be used as a platform help assistant; for research/analysis, start a new chat with the main assistant.
- They can return here for platform questions.
- End with the slogan "**Enjoy discovery!**"

### Boundaries

Do NOT answer domain/subject-matter questions. Redirect: "Great question! Start a new chat with the main assistant using the + button."

After onboarding is complete (tag with docs, skill created, profile in memory, user understands navigation), serve as platform support assistant.
"""  # noqa: E501


class OnboardingInstructionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable onboarding behavioral guidance.",
    )
    guidance: Optional[str] = Field(
        None,
        description=(
            "Free-form behavioral guidance for the agent. "
            "Describes what to encourage and how. When None, "
            "the built-in default guidance is used."
        ),
    )


class OnboardingInstructionSource(InstructionSource):
    """Injects onboarding guidance into the system prompt.

    Uses either configurator-provided guidance or a sensible
    built-in default that covers core platform features.
    """

    source_name = "onboarding"

    def __init__(
        self,
        enabled: bool,
        registration_service: RegistrationService,
        guidance: Optional[str] = None,
    ):
        self.enabled = enabled
        self.__guidance = guidance or _DEFAULT_ONBOARDING_GUIDANCE
        self.__registration_service = registration_service

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
        if not self.__guidance:
            return ""

        sections = [f"## Onboarding guidance\n\n{self.__guidance}"]

        profile = await self.__fetch_user_profile()
        if profile:
            sections.append(profile)

        return "\n\n".join(sections)


class OnboardingInstructionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        registration_service: RegistrationService,
        onboarding_instruction_source_configuration: (
            OnboardingInstructionSourceConfiguration
        ) = OnboardingInstructionSourceConfiguration(),
    ) -> OnboardingInstructionSource:
        return OnboardingInstructionSource(
            enabled=(onboarding_instruction_source_configuration.enabled),
            guidance=(onboarding_instruction_source_configuration.guidance),
            registration_service=registration_service,
        )
