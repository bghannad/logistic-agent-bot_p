# Logistic Agent Bot

**Status: Work in Progress.** The multi-agent system, retrieval pipeline, evaluation harness, tests, and CI are built and verified. Azure deployment has not happened yet (Docker is built and tested locally; the actual cloud deployment is the next step).

A multi-agent chatbot for delivery and customs support questions ("my package is stuck in customs," "can I import this item," "my order never arrived"). A supervisor agent routes each message to the specialist(s) it actually needs, rather than following one fixed sequence of steps.

## Architecture

```
customer message
      |
      v
   SUPERVISOR  --  decides which specialist(s) to call (zero, one, or more)
      |
  --------------------------------------------------
  |                  |                              |
TRACKING         COMPLIANCE                     ESCALATION
mocked tool      hybrid search (BM25 +          human-handoff
(shipment         embeddings) + reranking        judgment call,
 status lookup)   over a real CBP/FTC            full conversation
                  knowledge base                  as context
  |                  |                              |
  --------------------------------------------------
                      |
                      v
                 SUPERVISOR
         merges specialist findings into
         one reply, or asks a clarifying
          question if it needs more info
```

The model itself decides which specialists to call (real tool-calling, not a hardcoded `if`), and the supervisor-to-specialist handoff passes the customer's message via LangGraph's injected state rather than a model-generated argument, specifically so a compound question can't get silently split or paraphrased on its way to a specialist.

## Tech stack

- **Orchestration**: LangGraph (supervisor + specialist agents, tool-calling, conversation memory via checkpointing)
- **LLM**: OpenAI `gpt-4o-mini`
- **Retrieval**: hybrid search — BM25 (keyword) + `text-embedding-3-small` (semantic), merged by Reciprocal Rank Fusion, then reranked by a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Knowledge base**: real, public-domain CBP and FTC documents (not self-authored), chunked with a heading-aware heuristic
- **Web**: FastAPI backend, a plain HTML/JS chat page
- **Observability** (optional): Langfuse — tracing and versioned prompt management
- **Testing**: pytest, GitHub Actions CI
- **Deployment**: Docker (built and tested locally); Azure Container Instances deployment is planned, not yet done

## Results

All three evaluations use small, hand-labeled test sets written against the real knowledge base and the real agent behavior — not a self-authored knowledge base paired with self-authored questions, which was the honest caveat on this same project's predecessor.

**Retrieval** (18 hand-labeled questions, each with a known-correct chunk): Hit@5 17/18 (94%), Recall@5 0.92, Precision@5 0.26. (Precision@5 is low by construction — each question has only 1-2 correct chunks out of 5 returned slots, so it is not the meaningful number here; Hit@5 and Recall@5 are.)

**Routing** (18 hand-labeled customer messages, each with the specialist(s) that should be called): 17/18 (94%) exact match.

**Groundedness / hallucination check** (12 hand-labeled messages, LLM-as-judge): raw judge score 10/12 (83%). Hand-auditing the 2 flagged failures found both were judge errors, not real hallucinations — one flagged a fact that was literally present in the source text given to the judge, one penalized the system for honestly saying "not in my knowledge base" as if that were an unsupported claim. The audited result is 12/12. The judge's own prompt needed one round of tuning (explicit exemptions for "honestly says it doesn't know" and for generic advice) before its raw score was trustworthy at all — a cheap LLM judge is not reliable out of the box, even after tuning, which is why this was hand-audited rather than taken at face value.

Every test case in all three evaluations is self-authored (by the author, with Claude), not drawn from real customer messages.

## Known limitations

- **Tracking data is mocked.** `check_shipment_status` returns deterministic fake data for four tracking numbers, not a real carrier API.
- **Eval sets are small** (12-18 cases each) and self-authored, as noted above.
- **Conversation memory is in-memory only.** It is lost on process restart; no persistent checkpointer is wired up yet.
- **No access control.** Anyone who can reach the running app can use it. An optional access-code gate is designed but not built, since it only matters once this has a public URL.
- **Knowledge base is US-import-focused** (CBP and FTC documents only) and does not cover other countries' import rules.

## Data sources

The knowledge base is built from real U.S. federal government publications (CBP and FTC), in the public domain under 17 U.S.C. SS 105, not self-authored synthetic documents. Full source list, URLs, and retrieval dates: [`data/kb/sources/SOURCES.md`](data/kb/sources/SOURCES.md).

## Running locally

```
git clone git@github.com:bghannad/logistic-agent-bot_p.git
cd logistic-agent-bot_p
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
```

Single-turn CLI:
```
PYTHONPATH=src python3 -m agents.chat "my tracking number is TRK1001, where is it?"
```

Multi-turn chat in the terminal:
```
PYTHONPATH=src python3 -m agents.converse
```

Web UI:
```
PYTHONPATH=src uvicorn web.main:app --reload --app-dir src
```
then open `http://localhost:8000`.

## Tests

```
pip install -r requirements-dev.txt
pytest -q
```

No network calls, no API key spend — a dummy OpenAI key is set automatically for tests that only need one to construct, and any real Langfuse keys in `.env` are stripped for the test session.

## Langfuse (optional)

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (free tier at [cloud.langfuse.com](https://cloud.langfuse.com)) to get:
- full traces of every conversation (which specialist was called, why, what it returned)
- versioned prompt management — prompts live in `prompts/*.txt`, sync them to Langfuse with:
  ```
  python scripts/sync_prompts.py          # sync
  python scripts/sync_prompts.py --dry-run  # preview only
  ```

Without these keys, everything works identically, just untraced, reading prompts from the local `.txt` files directly.

## Docker

```
docker build -t logistic-agent-bot .
docker run --env-file .env -p 8000:8000 logistic-agent-bot
```
Image is approximately 2.2 GB (torch, transformers, and the baked-in reranker model account for most of it). The cross-encoder model is downloaded once at build time, so the running container needs no Hugging Face network access.

## Deployment

Azure Container Instances deployment is planned (see `docs/ROADMAP.md`) but not yet implemented — this section will be filled in once that's built.
