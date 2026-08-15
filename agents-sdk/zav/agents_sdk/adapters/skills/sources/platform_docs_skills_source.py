from typing import ClassVar, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.adapters.skills.sources.in_memory_skills_source import (
    InMemorySkillDefinition,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_SKILL_DEFINITIONS: List[InMemorySkillDefinition] = [
    InMemorySkillDefinition(
        name="platform-search",
        description=(
            "How to search and discover documents — keyword search, "
            "filters, search tabs, QA widget, federated web search."
        ),
        body="""\
# Searching and Discovering Documents

The **Discover** page is the main search interface.

## Basic search
Type your query in the search bar and press Enter. Results come \
from your organization's indexed documents and any federated \
search sources configured by your admin.

## Search tabs
The Discover page can show multiple search tabs:
- **Internal** — your organization's indexed documents.
- **All the Web** — federated web search via external engines.
- Additional tabs may be configured per tenant (e.g. academic \
sources, partner databases).

Switch between tabs to search different sources with the same query.

## Filters
Narrow results using the filter panel:
- Date range, document type, content source, authors
- Custom fields configured for your index (e.g. product line, \
department, classification level)
- Use the `get_filter_options` tool to discover available filters.

## QA widget
When enabled, a QA widget appears above search results and gives \
a direct AI-generated answer with citations. Click a citation to \
jump to the source document.

## Actions on results
For each search result you can:
- **Bookmark** it (adds to Favorites)
- **Add to a tag** (organize into collections)
- **Open in PDF viewer** (for PDFs)
- **Chat** about it (opens the QA sidebar)
- **Locate** a document by title, URL, or filename using fuzzy \
matching via the `locate_document` tool.
""",
    ),
    InMemorySkillDefinition(
        name="platform-tags",
        description=(
            "How to create, organize, and share document collections using tags."
        ),
        body="""\
# Tags — Organizing and Sharing Document Collections

Tags are the primary way to organize documents into collections \
— project folders, deal rooms, compliance dossiers, etc.

## Creating a tag
1. Go to the **Tags** page from the left sidebar.
2. Click **Create Tag** (or the + button).
3. Give it a name and optional description.

## Adding documents to a tag
- From **search results**: click the tag icon and select tag(s).
- From the **document page** or **PDF viewer**: use the tag action.
- **Drag and drop**: drag a document card onto a tag in the sidebar.
- The agent can also tag documents programmatically using tag tools.

## Browsing a tag
Click any tag to see:
- All documents in the tag, with sorting and filtering.
- A **Chat sidebar** — ask questions scoped to tagged documents.
- **Notes** tab — view and create notes within the tag.
- **Search within** the tag to find specific content.

## Sharing tags
- Click **Share** on any tag.
- Share with specific colleagues or the whole organization.
- Recipients can **follow** shared tags for sidebar access.

## Tag types
- **My Tags** — tags you created.
- **Shared** — tags shared with you.
- **Following** — shared tags you follow.
- **Favorites** — a built-in tag for bookmarked documents.
""",
    ),
    InMemorySkillDefinition(
        name="platform-documents",
        description=(
            "How to upload, manage, read, and chat with private "
            "documents — including the PDF viewer."
        ),
        body="""\
# Documents — Upload, Manage, Read, and Chat

## Uploading private documents
1. Go to **My Documents** from the left sidebar.
2. Click **Upload** and select files.
3. Supported formats: PDF, Word (.docx), Excel (.xlsx/.csv), \
PowerPoint (.pptx), plain text, and more.
4. Documents are automatically processed: text extracted, chunked, \
embedded, and indexed. Only you can see your private documents.

## After upload
- Documents appear in My Documents and are searchable from Discover.
- A **Chat** button lets you ask questions across all private docs.
- The agent can search your private documents with \
`search_my_documents` and list them with `list_my_documents`.

## Reading documents
- Open any document in the **PDF Viewer** for inline reading.
- The chat sidebar lets you ask questions about that document.
- The agent can read full document content page-by-page using \
`read_document`, including extracted text and bounding boxes.

## Managing documents
- Delete documents from My Documents by selecting and deleting.
- Re-upload updated versions to replace indexed content.
""",
    ),
    InMemorySkillDefinition(
        name="platform-chat",
        description=(
            "How to use the Chat page, select agents, attach context, "
            "resume conversations, use deep research, and schedule "
            "agents."
        ),
        body="""\
# Chat — Conversational AI

The **Chat** page provides the full conversational AI experience.

## Starting a conversation
Go to **Chat** from the sidebar, type your question, and press \
Enter. The agent responds with answers, optionally citing documents.

## Selecting agents
- Click the agent selector to switch between available agents.
- Different agents handle different tasks: general QA, document \
chat, deep research, data analysis, chemistry, etc.
- Go to **Settings > Agents** to configure which agents you see.

## Attaching context
- Click **Attach documents** to scope conversation to specific \
documents.
- You can also chat from within a **tag** (scoped to tagged docs), \
the **PDF viewer** (scoped to that PDF), or the **QA widget** on \
the Discover page (scoped to search results).

## Conversations and notes
- Every conversation is automatically saved as a **Note**.
- Resume by going to Notes and opening the saved chat note.
- All context and attached documents are restored.

## Deep research
Some agents support deep research — a multi-step investigation \
across many sources that produces a detailed report:
1. Ask a broad, high-stakes research question.
2. The agent proposes a research plan.
3. Confirm the plan, and the agent executes autonomously.
4. The report is saved as a note you can revisit.

Use deep research for competitive landscape analysis, regulatory \
evidence gathering, prior art searches, due diligence reviews, \
and any question that requires thorough multi-source analysis.

## Scheduled agents
- Create a schedule for any agent to run a recurring task \
(e.g. weekly competitor monitoring, daily regulatory digest).
- Results are delivered to you automatically.
- Configure schedules in **Settings > Agents**.

## Citations
- Agents cite sources as numbered references or superscripts.
- Click a citation to view the source passage.

## Memory
- The agent remembers facts about you across sessions: your name, \
role, domain, preferences, recurring projects.
- This context carries forward so you don't repeat yourself.
""",
    ),
    InMemorySkillDefinition(
        name="platform-data-analysis",
        description=(
            "How to analyze spreadsheets, CSV/Excel files, and SQL "
            "databases — statistics, filtering, aggregation, and "
            "chart visualization."
        ),
        body="""\
# Data Analysis — Spreadsheets, CSV/Excel, and SQL

The platform can analyze tabular data from uploaded files and \
connected databases.

## Loading spreadsheet data
- Upload a CSV or Excel file to My Documents.
- In chat, ask the agent to analyze it. The agent uses \
`load_dataframe` to load the file into a named DataFrame.
- Excel files with multiple sheets are loaded as separate \
DataFrames automatically.

## Exploring data
- `get_dataframe_schema` — see column names, types, and samples.
- `preview_dataframe` — see the first few rows.
- `count_dataframe_rows` — get the row count.
- `get_column_values` — list distinct values in a column.
- `get_column_statistics` — mean, median, std, min, max.
- `get_missing_value_counts` — find missing/null values.

## Filtering and transforming
- `filter_dataframe_rows` — filter by conditions (equals, \
greater than, contains, in list, etc.).
- `sort_dataframe_rows` — sort by any column.
- `aggregate_column` — compute sum, mean, min, max, count with \
optional group-by.
- `merge_dataframes` — SQL-style joins (inner, left, right, outer).
- `concatenate_dataframes` — stack DataFrames by rows or columns.

## Visualization
The agent can create charts from your data:
- `create_basic_plot` — bar, line, scatter charts.
- `create_multi_series_plot` — compare multiple data series.
- `create_distribution_plot` — histograms and distributions.
- `create_time_series_plot` — time-based trends.
- `create_correlation_heatmap` — show correlations between columns.

Charts are rendered inline in the conversation.

## SQL databases
If a SQL database is connected, the agent can:
- List tables and inspect schemas.
- Query, filter, sort, aggregate, and join tables.
- Insert new rows.
- All operations use safe, parameterized queries.
""",
    ),
    InMemorySkillDefinition(
        name="platform-skills-and-memory",
        description=(
            "How the agent learns from you — persistent memory, "
            "custom skills, and notes-based personalization."
        ),
        body="""\
# Skills and Memory — Teaching the Agent Your Workflows

## Memory
The agent has persistent memory that carries across sessions. \
It learns about you: your name, role, domain expertise, team, \
preferred answer style, and recurring projects.

- The agent saves relevant facts automatically when you share them.
- Next time you return, the agent already knows the context.
- Memory is private to you.

## Skills
Skills are reusable instruction sets that change how the agent \
behaves for a specific task. Think of them as saved workflows.

### Using skills
- The agent has a catalog of available skills.
- When your task matches a skill, it loads the instructions \
automatically (or you can ask it to use a specific skill).
- Skills provide step-by-step guidance, checklists, and templates.

### Creating skills
You can teach the agent new skills directly from conversation:
- Walk through a multi-step workflow (a review checklist, report \
template, triage procedure).
- Ask the agent to save it as a skill.
- The agent creates a structured skill definition it can follow \
perfectly next time.
- Skills can be stored as tag notes (shared with your team) or \
on disk for self-hosted deployments.

### Examples
- Patent review checklist that the agent follows consistently.
- Regulatory submission template with required sections.
- Due diligence procedure for evaluating acquisition targets.
- Triage procedure for categorizing incoming support tickets.
""",
    ),
    InMemorySkillDefinition(
        name="platform-notes",
        description=(
            "How to create, manage, and find notes and annotations on documents."
        ),
        body="""\
# Notes and Annotations

Notes let you capture thoughts, annotations, and AI conversations.

## Creating notes
- **Free notes**: Go to **Notes** from the sidebar and create a \
new standalone note.
- **Document annotations**: While viewing a PDF, highlight text \
and click **Add Note** to create an annotation linked to that \
passage.
- **Chat notes**: Every conversation is auto-saved as a note.

## Finding your notes
- The **Notes** page shows all your notes.
- Notes are searchable from the Discover page.
- Notes within a tag are visible on that tag's Notes tab.

## Sharing
Share notes with colleagues, just like tags.

## Resuming conversations
Open a saved chat note to resume the conversation where you left \
off. The full history and attached documents are restored.
""",
    ),
]

_SKILLS_BY_NAME: Dict[str, InMemorySkillDefinition] = {
    s.name: s for s in _SKILL_DEFINITIONS
}


class PlatformDocsSkillsSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable curated platform documentation skills.",
    )
    skill_overrides: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Override the body of specific skills by name. "
            "Keys are skill names, values are the replacement body."
        ),
    )


class PlatformDocsSkillsSource(SkillsSource):

    source_name: ClassVar[str] = "platform_docs_skills"

    def __init__(
        self,
        enabled: bool,
        skill_overrides: Optional[Dict[str, str]] = None,
    ):
        self.enabled = enabled
        self.__overrides = skill_overrides or {}

    async def discover(self) -> Dict[str, SkillProperties]:
        return {
            s.name: SkillProperties(name=s.name, description=s.description)
            for s in _SKILL_DEFINITIONS
        }

    async def get_skill_body(self, skill_name: str) -> str:
        if skill_name in self.__overrides:
            return self.__overrides[skill_name]
        skill = _SKILLS_BY_NAME.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(_SKILLS_BY_NAME.keys()))
        return skill.body

    async def get_resources(self, skill_name: str) -> List[str]:
        if skill_name not in _SKILLS_BY_NAME:
            raise SkillNotFoundError(skill_name, list(_SKILLS_BY_NAME.keys()))
        return []

    async def read_resource(self, skill_name: str, path: str) -> str:
        if skill_name not in _SKILLS_BY_NAME:
            raise SkillNotFoundError(skill_name, list(_SKILLS_BY_NAME.keys()))
        raise SkillResourceNotFoundError(skill_name, path, [])


class PlatformDocsSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        platform_docs_skills_source_configuration: PlatformDocsSkillsSourceConfiguration = PlatformDocsSkillsSourceConfiguration(),  # noqa: E501
    ) -> PlatformDocsSkillsSource:
        cfg = platform_docs_skills_source_configuration
        return PlatformDocsSkillsSource(
            enabled=cfg.enabled,
            skill_overrides=cfg.skill_overrides,
        )
