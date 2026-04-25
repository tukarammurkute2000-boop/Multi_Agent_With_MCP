# Venture Pilot — Multi-Agent MCP System

An AI-powered venture capital automation platform built with **LangGraph**, **FastMCP**, and **Claude**. It orchestrates a pipeline of specialized agents that handle every stage of fund operations — from onboarding through portfolio management.

## Architecture

```
START
  → Onboarding       (thesis · API keys · forms · social profiles · scoring)
  → Social Media     (post content · manage inbound · schedule events)
  → Scouting         (web scraping · AI discovery · mail outreach · diligence · scoring)
  → Phase 3 Parallel (pipeline + mentoring + reporting — concurrent)
  → [gate: demo_day_complete?]
  → Phase 4 Parallel (portfolio + investor — concurrent)
END
```

Each agent delegates all external API calls to a single **MCP server** (`mcp_server.py`). The server enforces a per-phase tool allowlist so agents can only call the integrations that are appropriate for their stage.

## Agents

| Agent | Sub-agents |
|---|---|
| **Onboarding** | thesis, api_keys, forms, social, scoring |
| **Social Media** | post_content, manage_inbound, schedule_events |
| **Scouting** | web_scraping, ai_discovery, mail_outreach, scoring, tech_diligence, business_diligence |
| **Pipeline** | pipeline_mgmt, scheduling, diligence_update, scoring_update, mentoring_needs |
| **Mentoring** | mentor_match, personalized_mentoring, investor_intro, demo_day, compliance |
| **Reporting** | platform_report, investor_report, fund_manager_report |
| **Portfolio** | ongoing_mentoring, investor_intro, update_reports, followon_fundraising |
| **Investor** | portco_update, lp_report, compliance_check, followon_fundraising |

## MCP Tools

| Tool | Integration | Used in phases |
|---|---|---|
| `scrape_startups` | Apify | scouting |
| `send_email` | SendGrid | scouting, mentoring, reporting, portfolio, investor |
| `post_twitter` | Twitter/X API | onboarding, social_media |
| `post_linkedin` | LinkedIn API | onboarding, social_media |
| `create_calendar_event` | Google Calendar | social_media, pipeline, mentoring, portfolio |
| `supabase_query` | Supabase | onboarding, pipeline, reporting, portfolio, investor |
| `supabase_upsert` | Supabase | onboarding, scouting, pipeline, mentoring, portfolio, investor |
| `store_embedding` | Pinecone | scouting |
| `retrieve_context` | Pinecone (RAG) | scouting, pipeline, mentoring, reporting, portfolio, investor |
| `claude_generate` | Anthropic API | all phases |
| `generate_pdf` | ReportLab | reporting, portfolio, investor |

## Persistence

- **Checkpointer** — saves full LangGraph state after every node using SQLite (dev) or Postgres (prod). Pass a previous `thread_id` to `run_orchestrator()` to resume a crashed or paused run.
- **MemoryStore** — cross-thread storage for fund config, user preferences, and deal notes.

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd Multi_agent_Mcp
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Fill in `.env` with your API keys:

| Variable | Service |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI (optional fallback) |
| `SUPABASE_URL` / `SUPABASE_KEY` | Supabase |
| `DATABASE_URL` | Postgres (prod checkpointer) |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | Pinecone vector store |
| `SENDGRID_API_KEY` | Email via SendGrid |
| `APIFY_API_TOKEN` | Web scraping via Apify |
| `TWITTER_API_KEY` / `TWITTER_API_SECRET` / `TWITTER_ACCESS_TOKEN` / `TWITTER_ACCESS_SECRET` | Twitter/X |
| `LINKEDIN_CLIENT_ID` / `LINKEDIN_CLIENT_SECRET` | LinkedIn |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Google Calendar |

## Usage

**Run the full pipeline**

```bash
python main.py
```

**Resume a previous run** (pass the `thread_id` printed on the prior run)

```python
from orchestrator.orchestrator import run_orchestrator
import asyncio

asyncio.run(run_orchestrator(
    user_id="demo_user",
    user_role="fund_manager",
    thread_id="<previous-thread-id>",
))
```

**Run the MCP server standalone**

```bash
python mcp_server.py
```

**Call a tool directly from a sub-agent**

```python
from mcp_server import call_tool

result = await call_tool("send_email", phase="scouting",
                          to="founder@example.com",
                          subject="Due diligence follow-up",
                          body="<p>Hi, ...</p>")
```

## Project Structure

```
├── main.py                  # Entry point
├── mcp_server.py            # FastMCP server + phase-based tool allowlist
├── orchestrator/
│   └── orchestrator.py      # LangGraph StateGraph definition
├── agents/
│   ├── onboarding/
│   ├── social_media/
│   ├── scouting/
│   ├── pipeline/
│   ├── mentoring/
│   ├── reporting/
│   ├── portfolio/
│   └── investor/
├── memory/
│   ├── checkpointer.py      # SQLite / Postgres checkpointer
│   ├── store.py             # Cross-thread MemoryStore
│   ├── thread.py            # Thread ID helpers
│   └── cache.py             # Prompt cache helpers
├── state/
│   └── schema.py            # VenturePilotState TypedDict
├── config/
│   └── settings.py          # Environment variable loader
├── requirements.txt
└── .env.example
```

## Requirements

- Python 3.11+
- See `requirements.txt` for the full dependency list (LangGraph, LangChain, FastMCP, Anthropic SDK, Supabase, Pinecone, etc.)
