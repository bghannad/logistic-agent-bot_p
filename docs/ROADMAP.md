# Logistic-agent-bot — Project Roadmap & Design Doc

(Project 2 in the portfolio series. Project 1 is `support-triage-agent` — finished, deployed 2026-09-10.
This doc is the equivalent of that project's `LEARNING_GUIDE.md`: read it whenever you're not sure why a
piece exists, and update it as decisions change. It was written before any code, on purpose, so we agreed
on the shape first. **Status (2026-10-20):** everything through evaluation is built; Docker + Azure
deployment is next (Sec. 8 step 8). Per-phase learning notes (what happened, why, options, outcome,
caveats) are in `notes/01`–`06`, with `about.md` as the one-page index.)

---

## 1. Why this project exists

Project 1 proved you can build *an* agent with real RAG and a real framework (LangGraph). It closed three
gaps: RAG (no vector DB/embeddings/retrieval eval in any prior project), agent frameworks (prior agent work
was custom orchestration, not framework-based), and Azure deployment depth.

But Project 1 is still, underneath, one linear-ish graph with a hardcoded tool trigger
(`if category in {"billing", "account_access"}`) and a synthetic, self-authored knowledge base. Three
things it does NOT prove:

1. **Multi-agent coordination** — multiple agents with distinct jobs and a coordination layer between
   them, as opposed to one agent following a sequence of steps.
2. **Genuine model-driven tool use** — the model itself deciding whether and when to call a tool, not a
   Python `if` statement deciding for it.
3. **Retrieval that's actually hard** — Project 1's own eval caveat was "I wrote both the KB and the eval
   questions, so the perfect retrieval score is suspect." Real, messier documents fix that.

This project is built specifically to close those three gaps — not to repeat "RAG again" or "deployment
again" for their own sake. Multi-agent coordination and real tool-calling are the actual thesis; harder
retrieval and a more production-grade chatbot shape are supporting cast.

## 2. The real-world problem (not just "what does this prove about me")

A customer's delivery goes wrong, and someone has to figure out why and what to do about it — without
making the customer wait for a human every time. Concretely, these are the kinds of things a real person
would type into a delivery-company support chat:

- *"My package has been stuck in customs for 5 days, what's going on?"*
- *"Tracking says delivered but I never got it."*
- *"It says failed delivery attempt but nobody came."*
- *"Can I redirect this to a different address?"*

Each one needs a genuinely different kind of information to answer well — not a canned response. The
customs one needs live tracking data (where is it, what customs stage). The redirect one needs
policy/rules knowledge (is redirection even allowed for this shipment type/country). The "never got it"
one might need both, plus a judgment call on whether to escalate to a human (possible lost package,
possible fraud). The real operational question: **can one system look at an arbitrary delivery complaint,
figure out what kind of information it actually needs, go get that specific information from the right
place, and only bother a human when it genuinely can't resolve things itself?** That's a real problem — it's
what a delivery company's customer service org spends money on humans to triage today.

## 3. The finished picture

```
   customer message (turn N of an ongoing conversation)
            |
            v
     [SUPERVISOR AGENT]  --  reads the message + conversation so far,
            |                decides which specialist(s) it needs
            |
   -----------------------------------------------------------
   |                    |                                    |
   v                    v                                    v
[TRACKING AGENT]  [CUSTOMS/COMPLIANCE       [ESCALATION AGENT]
calls a tool       RAG AGENT]                decides human handoff,
(shipment status,  hybrid search + rerank     only if genuinely warranted,
at runtime, if      over the real CBP/FTC     with the full conversation
the model decides   KB (Sec. 7)                as context
it needs to)
   |                    |                                    |
   -----------------------------------------------------------
                         |
                         v
                [SUPERVISOR AGENT]
           merges specialist findings into
           one coherent, grounded reply —
           or asks a clarifying question if
           it doesn't have enough yet (Sec. 5)
                         |
                         v
              reply, sent back into the
           same ongoing conversation thread
```

Each box is its own agent with its own job and, where relevant, its own tools — this is the actual
"multi-agent system" pattern (a coordination layer routing to specialists), not one agent working through
a fixed sequence of steps.

## 4. Architecture, piece by piece

- **Supervisor agent.** A router and synthesizer, not a worker. Reads the incoming message plus
  conversation history, decides which specialist(s) to call (possibly more than one, possibly none if it
  can answer directly or needs to ask a clarifying question), and merges their outputs into one reply.
- **Tracking agent.** Has a tool — `check_shipment_status(tracking_ref)` — mocked the same honest way
  Project 1 mocked `check_account_status` (deterministic fake data, not a real carrier API; this is a
  named simplification, see Sec. 7). The model decides *at runtime*, via real function-calling, whether it
  needs to call the tool at all — this is the fix for Project 1's hardcoded trigger.
- **Customs/compliance RAG agent.** Retrieves from the real KB (CBP + FTC documents, see Sec. 6) using
  hybrid search (keyword + embedding similarity) plus a reranking step, not similarity search alone —
  the real documents are long enough and dense enough that plain top-k similarity search is expected to
  need this.
- **Escalation agent.** Decides whether to hand off to a human, and only if genuinely warranted — mirrors
  Project 1's `decide_node` reasoning (security-flavored issue, low confidence, something the other two
  agents couldn't resolve) but as its own agent with the full conversation as context, so a human handoff
  comes with everything the customer already said, not a blank slate.
- **Conversation memory.** Built as a real multi-turn chatbot using LangGraph's checkpointed state tied to
  a conversation thread — not single-shot ticket-in/answer-out like Project 1. This means the supervisor
  can ask a clarifying question ("what's your tracking number?") and pick the conversation back up once
  the customer answers, which is a genuinely different and more advanced agent behavior than "always
  produce an answer."

## 5. Why a chatbot, specifically

A ticket-in/answer-out pipeline, even a multi-agent one, is still fundamentally "retrieve and respond
once." A real chatbot has to handle things a one-shot system never does:

- **It doesn't have all the information yet.** Someone opens with "my package hasn't arrived" — no
  tracking number, no context. Knowing when to ask a clarifying question instead of answering (or
  guessing) is a more advanced, more honest agent behavior than always producing an answer.
- **It has to remember the conversation.** Turn 3 needs to know what was said in turn 1.
- **It can hand off mid-conversation, with context** — escalating "here's the full conversation so far" to
  a human is a real, valuable behavior and a much better demo than a static escalation message.

## 6. The knowledge base — real documents, not synthetic

Sourced from real, public-domain U.S. federal government publications (CBP + FTC) instead of self-authored
synthetic docs — see `data/kb/sources/SOURCES.md` for the full source list, URLs, and the public-domain
rationale (17 U.S.C. § 105). As of this doc, six raw documents are pulled and saved to
`data/kb/raw/` (verbatim, headers noting source/URL/retrieval date):

- `cbp_internet_purchases.txt` — the strongest single match for the core scenario (buying from abroad,
  duty, informal vs. formal entry thresholds, lost packages, prohibited/restricted merchandise).
- `cbp_prohibited_restricted_items.txt` — long, categorized, genuinely messy — good stress test for
  chunking and retrieval.
- `cbp_protecting_americans_shopping_online.txt` — short, counterfeit-goods focused.
- `ftc_online_order_never_arrives.txt`, `ftc_prompt_delivery_rules.txt` (the actual Mail/Telephone/Internet
  Order Merchandise Rule — has real regulatory teeth on refund timing), `ftc_scams_orders_dont_arrive.txt`.

Not yet pulled but identified as worth adding (see `SOURCES.md` "Not yet pulled"): the June 2026 Federal
Register notice suspending duty-free de minimis treatment for mail shipments — ~21 dense pages of brand-new
regulatory text, arguably the best "messy and real" stress test available, and a good excuse to talk about
*why* a low-value package might suddenly be held under a rule that didn't used to apply to it.

This real-document choice is also what makes hybrid search + reranking worth doing at all — Project 1's
16 short synthetic docs were basically all one chunk each; these are long enough and dense enough that
retrieval quality is a real, non-trivial question.

## 7. Known simplifications & design decisions (the honest list — start filling this in as we build)

Carried over from Project 1's own list as a reminder of the *shape* this section should take once we
start building — DHL wants trade-offs named out loud, unprompted, before they find them:

1. **Tracking tool will be mocked**, same as Project 1's account-status tool — deterministic fake data, not
   a real carrier/USPS API. Named honestly, not hidden.
2. **Chunking is heuristic, not markdown-header-based** — the raw docs have no headers (extracted via
   `innerText`, see `data/kb/sources/SOURCES.md`). The two long docs (`cbp_internet_purchases.txt`,
   `cbp_prohibited_restricted_items.txt`) get heading-aware chunking via a line-scanning heuristic
   (`src/ingestion/chunking.py`) that looks for standalone short title-like lines and "Label: rest of
   sentence" run-in headings; the four short docs are chunked whole. Known limitation: a handful of short
   mailing-address lines inside `cbp_prohibited_restricted_items.txt` (e.g. "Falls Church, VA 22041") get
   misread as headings, slightly over-segmenting those topics with a nonsense section label — the content
   itself stays attached to the right topic, so this is a labeling artifact, not a retrieval-correctness
   one. Verified by hand against the raw text. The retrieval eval (Sec. 8 step 7) didn't test it
   specifically; its one miss, `ret_13`, has no diagnosed cause yet.
3. **Embedding model: OpenAI `text-embedding-3-small`.** Hybrid search = BM25 (keyword) + cosine similarity
   (embeddings), merged by Reciprocal Rank Fusion (rank-based merge, no hand-tuned weight between the two
   scoring systems). Reranker: a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, free, no
   per-query API cost) re-scores the top 15 hybrid candidates down to the final top 5. Vectors are stored as
   a plain local `.npy` file, not a vector DB — 63 chunks doesn't need one. Cost: embedding the whole KB was
   ~$0.0003; each future query embeds the query text too, also a fraction of a cent.
4. **Supervisor-to-specialist handoff uses LangGraph's `InjectedState`, not a model-filled argument.**
   Originally `call_tracking_agent`/`call_compliance_agent` took a `user_message: str` arg the supervisor LLM
   filled in — on compound questions ("package X is stuck AND I think it's banned") it silently split/
   paraphrased the message per specialist, which also degraded that specialist's retrieval since it ran on
   the trimmed text. A docstring instruction not to trim didn't fix it (small models don't reliably honor
   that over their default tool-arg-extraction behavior). Fix: the tools now take no model-visible argument
   at all — they pull the customer's actual last message out of graph state via `InjectedState`, so there's
   nothing left for the model to paraphrase. Known remaining limitation (not fixed): if the customer's
   message is itself vague about what item they mean, the compliance specialist still gives a generic
   answer — that's a retrieval/information-availability gap, not a context-loss bug, and arguably needs a
   clarifying question from the supervisor rather than a retrieval fix. Left as-is for now.
5. **All eval questions (retrieval, routing, groundedness) are self-authored**, same caveat Project 1 had
   ("I wrote both the KB and the eval questions"). Here the *KB* is real, but the test *questions* in
   `data/eval/` are still written by Claude and reviewed by Bahar, not drawn from real customer messages.
6. **Groundedness/hallucination eval uses an LLM-as-judge (`src/eval/groundedness_eval.py`), and the judge
   itself turned out to be unreliable** — worth knowing going in, not just in hindsight. First run: the judge
   flagged 5 of 12 replies as "ungrounded" (7/12 grounded). Hand-checking against the source text showed
   most were judge mistakes: it penalized the system for honestly saying "not in my knowledge base" (a true
   statement about the source, not an unsupported claim) and for generic advice ("contact CBP") despite an
   explicit prompt exemption for both, and it missed a date that was in the tracking data. I fixed the judge
   prompt (explicit exemptions, told to read every field) *after seeing the results*, and the re-run gave
   **11/12**. The one remaining flag (Cuba/gold, `ground_05`) is borderline: the reply says "gold items" where
   the source chunk says "gold coins, medals and bullion" — a mild broadening, not an invention. Net finding:
   in the final run no reply was clearly hallucinated (one borderline), but treat it as one sample: replies
   change between runs (an earlier reply to `ground_12` did claim there was "no indication" the contents were
   prohibited, which the bot can't know), the judge was tuned on the same 12 cases, and a planted-error test
   (add a false claim, check the judge catches it) was not done. The real lesson: a cheap LLM judge
   (`gpt-4o-mini`, same model used everywhere else for cost reasons) can misjudge even after prompt tuning,
   so its raw score isn't trustworthy without hand-verifying a sample — exactly why that step wasn't
   skipped here.
7. **Routing eval: 17/18 exact match; the one mismatch (`route_10`, "Can I redirect my shipment…") is an
   out-of-scope action request.** The supervisor called no specialist and asked for a tracking number — no
   hallucination, but the wording implied the bot could redirect shipments, which it can't (the KB has no
   redirect policy, so the label "compliance" was debatable too). Left as a known limitation: label, prompt
   and code unchanged. Possible later fix: a supervisor rule for out-of-scope action requests, plus new
   out-of-scope routing cases.
8. **The local web app is a demo, not production.** Memory is in RAM (a restart wipes every conversation, and
   several server copies wouldn't share it), there is no login or rate limit (anyone with the address can
   spend API money), and it shows only the final reply (no routing or chunks). Fine locally; to be addressed
   before or while deploying (Sec. 8 step 8).

## 8. What's planned to build, in order

| # | Task | Status |
|---|---|---|
| 1 | Pull real CBP + FTC source documents, write `SOURCES.md` | **Done** — 6 docs pulled, `SOURCES.md` written |
| 2 | This design doc | **Done** — you're reading it |
| 3 | Project scaffold (accounts, repo, `.env`) | **Done** — new OpenAI key obtained and in use since Sec. 8 step 4 |
| 4 | Chunk + embed the real KB; hybrid search + reranking retrieval | **Done** — `src/ingestion/` (chunking) + `src/retrieval/` (embed, hybrid search, rerank). Spot-checked with real queries, then formally evaluated in step 7. |
| 5 | Supervisor + specialist agents in LangGraph, with LLM-decided tool-calling | **Done** — `src/agents/`. Built single-turn first (memory came in step 6). Spot-tested 3 scenarios (tool call, clarifying question, honest "not in KB"), then formally evaluated in step 7. |
| 6 | Conversation memory / multi-turn chatbot behavior | **Done** — `agents/converse.py`, LangGraph `InMemorySaver` checkpointer keyed by thread_id. Tested: clarifying-question-then-answer flow works across turns. In-memory only (lost on quit); a persistent checkpointer is a later swap if needed. |
| 7 | Evaluation harness (classification/routing + retrieval quality, adapted for multi-agent + hybrid search) | **Done** — three separate evals in `src/eval/` against hand-labeled cases in `data/eval/`: `retrieval_eval.py` (Hit@5 17/18 = 94%, Precision@5 0.26 vs a ceiling of 0.29, Recall@5 0.92), `routing_eval.py` (exact match 17/18 = 94%), and `groundedness_eval.py`, an LLM-as-judge hallucination check (11/12 after fixing the judge; see Sec. 7 items 5-7). Not done: MRR in code, ablation, hard/robustness cases, a planted-error test of the judge, end-to-end / escalation / multi-turn / cost evals. |
| 8 | Containerize + deploy to Azure | In progress — local web UI done first (`src/web/`, FastAPI + a static chat page), so there's something to actually see before infra work. Tested: single-turn and multi-turn memory both work through the API. Docker + Azure not started yet (`docker/` is still empty; next: Dockerfile and `.dockerignore`, run it locally, then Azure). |
| 9 | README + portfolio packaging | Not started |
| — | Learning notes per phase | **Done** — `notes/01`–`06` plus `about.md` as the index |

## 9. Concepts you should be able to explain and defend (self-test as we go)

- Why multi-agent (supervisor + specialists) over one bigger graph — what does the coordination layer
  actually buy you that a single agent with more `if` branches wouldn't.
- LLM-decided tool-calling vs. a hardcoded trigger — mechanically, how does the model "decide" to call a
  tool, and what's the failure mode if it decides wrong.
- Hybrid search — what keyword search catches that embedding similarity alone misses (and vice versa),
  and what a reranker adds on top of either.
- Conversation memory / checkpointing in LangGraph — what state actually persists between turns and how.
- Why real (messy) documents make retrieval evaluation more meaningful than Project 1's synthetic KB did.
- The honest account of what's still mocked (the tracking tool) vs. what's real (the KB, the routing
  logic, the multi-agent coordination).
- What `InjectedState` fixed in the supervisor-to-specialist hand-off, and why a prompt instruction didn't.
- How hallucination is attacked in layers (prevent with prompt rules, check retrieval and routing, judge-free
  tests such as unanswerable and trap questions, a judge model as a screen, reading samples) and why the
  judge model itself needs validating.

## 10. Working style (unchanged from Project 1)

Claude drafts the code, Bahar studies it and asks questions — needs to be able to explain and defend it in
an interview, not just have it work. Any change to previously-reviewed code gets described precisely
before being made, not shown as a diff afterward. Cost-consciousness matters — keep an eye on spend the
same way as Project 1.
