# Build Journal

How this project actually got built, in order, including the parts that went wrong.

Every failure in here is one I hit for real. I've kept the broken code in each
chapter so the fix has something to be a fix *of*.

---

## Chapter 0 — What I started with

A CSV a senior handed me: 3,101 rows of "which company recruited at which college
in which year", covering 27 engineering colleges around Mumbai and Pune from 2014
to 2018. Five columns:

```
,name of company,college name,region,year
0,zeus,Pillai College of Engineering,Navi Mumbai,2018
1,Aditya Birla,Pillai College of Engineering,Navi Mumbai,2018
```

The idea was simple: juniors kept asking the same questions in the placement
WhatsApp group, so let them ask a chatbot instead of scrolling a spreadsheet.

---

## Chapter 1 — The first version, and the trap I walked into

I got a working RAG pipeline in an evening. Chroma locally, Groq for the LLM,
LangChain to glue it. Then I wanted it hosted, so I migrated to MongoDB Atlas and
Gemini.

The trap: **I migrated by writing new files instead of changing the old ones.**

A month later the repo looked like this:

```
app.py            Streamlit + MongoDB Atlas + Gemini
query.py          CLI + Chroma + Groq + Llama-3.3      <- the old stack, still there
ingest.py         writes to Atlas, connection string hardcoded
ingest_atlas.py   also writes to Atlas, reads from .env  <- near-duplicate
avl_models.py     a script I wrote once to list Gemini models
main.py           print("Hello from placement-rag!")
```

Two complete RAG stacks, two ingestion scripts that disagreed about where the
connection string lives, and 26 MB of stale `chroma_db/` still on disk feeding
nothing.

**What I learned.** A migration isn't finished when the new path works. It's
finished when the old path is *deleted*. Anyone reading the repo — including me,
three weeks later — cannot tell which file is real. I deleted five files and kept
one pipeline: MongoDB Atlas + Gemini + MiniLM + LangChain.

---

## Chapter 2 — The credential I published to the internet

`ingest.py` line 53:

```python
MONGO_URI = "mongodb+srv://suyash:<db_password>@<cluster>.mongodb.net/"
```

and line 73, sitting on its own at the bottom of the file, outside any function,
where I had pasted it to copy from later:

```python
    <the actual password, verbatim>
```

That's the database password. On a public repository. Python never executed the
line because it's unreachable after `return`, so nothing ever complained.

I also found this in `ingest_atlas.py`:

```python
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True,   # <- added to make an error go away
    serverSelectionTimeoutMS=30000
)
```

I'd added that flag months earlier because of a TLS error on college wifi. What it
actually does is disable certificate verification, which is the entire point of
TLS — it turns "encrypted and verified" into "encrypted to whoever is answering".

**Fixes.**
1. Rotated the Atlas password. Deleting the line does nothing — git history keeps it.
2. Restricted Atlas Network Access to specific IPs instead of `0.0.0.0/0`.
3. Every secret now comes from `.env` via pydantic-settings, with a validator that
   refuses to start if the URI still contains a placeholder.
4. Removed `tlsAllowInvalidCertificates`.

**What I learned.** "It's just a demo project" is exactly how credentials leak, and
a bot scraping GitHub for `mongodb+srv://` doesn't care that it's a college project.

---

## Chapter 3 — The demo that couldn't answer its own demo question

My chat box placeholder said:

```python
st.chat_input("Ex: Show me companies with salary > 10 LPA")
```

My README advertised "Eligibility Filtering: Instantly filters companies based on
CGPA, branch, and 10th/12th percentages."

My dataset had five columns: company, college, region, year.

There was **no package column. No CGPA. No branch. No eligibility criteria at
all.** I had written the README describing the project I imagined, not the one I
had. Ask it the question printed in its own input box and it confidently produced
something, because the LLM will always produce something.

**Fix.** I generated a synthetic eligibility layer — `package_lpa`, `cgpa_cutoff`,
`allowed_branches`, 10th/12th percentages, backlog policy, selection rounds, role,
location — on top of the real company/college/year rows, and said so plainly in the
README and in the UI footer.

It's derived from a hash of `(company, college, year)`, so it's deterministic and
correlated rather than random noise: package tracks company tier and drifts with
year, CGPA cutoff tracks package, branch sets follow what the company actually
does. Microsoft comes out around 18 LPA with a 7.0 cutoff for CS/IT/ENTC; TCS comes
out around 3.3 LPA. Plausible enough to build and demo retrieval against, and
labelled clearly enough that nobody mistakes it for real placement statistics.

**What I learned.** The README is a promise. If the data can't keep it, either get
the data or change the README — but don't ship the gap.

---

## Chapter 4 — Cleaning the data I assumed was clean

Before generating anything I actually looked at the rows properly:

- **Six rows had `Name of Company` as the company name.** Header rows from
  whatever spreadsheets got concatenated to build this file. They'd been embedded
  and indexed as if they were real companies.
- **`L&T Infotech` and `L & T Infotech` were separate companies.** So were several
  other spacing variants. Same recruiter, split across two identities, so every
  query about them saw half the records.
- **Duplicate `(company, college, year)` triples**, from the same drive appearing
  in more than one source sheet.

334 of 3,101 rows were junk or duplicates — about 11%. After cleaning and name
resolution: 2,764 rows, 1,966 distinct companies (down from 2,116 before casing
variants were merged).

```python
def canonical_company(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip().strip(".,-")
    return re.sub(r"\s*&\s*", "&", name)
```

**What I learned.** I'd spent weeks tuning retrieval on top of data I had never
actually inspected. Twenty minutes of `Counter(...)` and `sorted(set(...))` at the
start would have saved most of it.

---

## Chapter 5 — The chunking bug that was hiding in plain sight

This was my ingestion:

```python
loader = DirectoryLoader(folder, glob="**/*.csv", loader_cls=CSVLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
```

I'd copied it from a tutorial about PDFs and never questioned it.

`CSVLoader` produces one document per row. Each row is about 40 characters. Then
`RecursiveCharacterTextSplitter(chunk_size=1000)` — whose job is to make chunks
*close to* 1,000 characters — packed roughly **twenty unrelated companies into
every chunk**. And `chunk_overlap=100` duplicated rows across chunk boundaries on
top of that.

So retrieving "the best chunk" for a question about one company handed Gemini a
blob containing nineteen other companies. The answer was often right by luck,
because the LLM could find the relevant line in the mess. When it was wrong, it was
wrong in the worst way — confidently citing a package that belonged to a different
company two rows down.

**Fix.** For structured data, the record *is* the chunk. There is no splitter in
the pipeline at all now. One document per drive, written as a sentence:

```
Microsoft recruited at Veermata Jijabai Technological Institute (Mumbai) in 2014
for the role of Technology Analyst. Package offered: 18.32 LPA. Eligibility:
minimum CGPA 7.0, open to Computer Engineering, Electronics & Telecommunication,
Information Technology branches, minimum 70% in 10th and 75% in 12th. Active
backlogs: Not allowed. Selection process: Online Coding Test -> Technical
Interview 1 -> ... Job location: Pune. Hiring type: Internship + PPO.
```

The sentence form matters too. `Microsoft,VJTI,Mumbai,2014,18.32` embeds badly —
sentence-transformers were trained on prose, not on comma-delimited fields. Writing
it out gives the embedding model something it recognises, and gives BM25 real
tokens to match.

**What I learned.** Chunking advice is always given for prose. Nobody says "this
doesn't apply to CSVs", and the failure is silent — no error, just quietly worse
answers.

---

## Chapter 6 — "How many times did Infosys visit?"

Row-level documents can only answer row-level questions. Asking how many drives a
company ran, or which years it recruited, requires seeing all its rows at once —
and no single row contains that.

Retrieval would return four Infosys rows out of thirteen, and the model would
faithfully answer "Infosys visited 4 times", which is wrong, and cited, and looks
completely credible.

**Fix.** A second document type: one aggregated profile per company, indexed
alongside the row documents.

```
Company profile: TCS. TCS conducted 15 campus drive(s) across 11 college(s) in the
years 2015, 2016, 2017, 2018. Colleges visited: ... Packages offered ranged from
3.05 LPA to 3.8 LPA (average 3.31 LPA). Roles hired for: ... CGPA cutoff ranged
from 6.0 to 7.0.
```

The query router detects aggregate phrasing ("how many", "which years", "tell me
about", "overview") and sorts profile documents to the front.

4,730 documents total: 2,764 drive records + 1,966 company profiles.

**What I learned.** Index at the granularity of the *questions*, not just the
granularity of the source data.

---

## Chapter 7 — The question embeddings fundamentally cannot answer

This is the one that changed how I think about RAG.

```python
docs = vector_store.similarity_search(prompt, k=4)
```

Ask "show me companies with salary above 10 LPA" and you get four documents about
salaries. Not four documents *above 10 LPA* — four documents that are *about the
topic of salary*.

I checked the cosine similarity between the embeddings of:
- "companies with salary above 10 LPA"
- "companies with salary above 5 LPA"

They're near-identical. Of course they are. The embedding captures what the
sentence is *about*. The threshold is one token in a 384-dimensional average.
**There is no value of `k`, no better embedding model, and no chunking strategy
that fixes this**, because the information isn't in the vector.

**Fix.** Numeric and categorical constraints get lifted out of the text before
retrieval and pushed to the database as a pre-filter:

```python
"Which CS companies visited in 2018 with package more than 12 lakhs"
->  filters = {
      "package_lpa": {"$gte": 12.0},
      "year":        {"$eq": 2018},
      "branches":    {"$in": ["Computer Engineering"]},
    }
```

Regex handles the common phrasings — no LLM call, no latency. An LLM extractor
covers unusual phrasing as a fallback.

One subtlety that took me a second attempt to get right. "I have 7.2 CGPA, which
companies am I eligible for?" must become `cgpa_cutoff <= 7.2`. My first version
generated `cgpa_cutoff == 7.2`. The number in the question describes the *student*;
the field describes the *company's bar*. Wrong direction, plausible-looking
results, no error.

And these have to be **pre**-filters, applied inside the vector search, not applied
to the results afterwards. Post-filtering top-6 for `package > 10` leaves you with
one result, because the six were chosen without knowing about the constraint.

**What I learned.** RAG isn't only "embed and search". Knowing what embeddings
*can't* represent is as important as knowing what they can.

---

## Chapter 8 — Adding BM25, and being wrong about why

I added hybrid search expecting it to fix exact-name lookups. The textbook argument
is that `all-MiniLM-L6-v2` has weak signal for rare proper nouns — "Zeus Learning"
isn't a concept it learned, just two tokens averaged into something generic —
whereas BM25 nails it because `Zeus` has enormous IDF.

**When I measured it, that wasn't the win.** Exact-name questions already scored
1.00 on pure vector search (Chapter 15 has the numbers). The reason is my own
Chapter 5 fix: because each document is a sentence that *contains the company name
verbatim*, the name is right there in the text the embedding was built from. The
theory is sound; it just didn't apply to a corpus I'd already made easy.

Where BM25 actually earned its place:

| Question type | Vector only | + BM25 hybrid |
| --- | --- | --- |
| aggregate ("which years did Capgemini recruit?") | 0.50 | **1.00** |
| multi-constraint ("CS companies in 2018 above 12 LPA") | 0.33 | **1.00** |
| exact name | 1.00 | 1.00 |

Aggregate questions improve because company-profile documents repeat the company
name many times over a short document — exactly the shape BM25's term frequency and
length normalization reward, and exactly what a cosine average dilutes.
Multi-constraint questions improve because once the pre-filter has narrowed things
to a few dozen candidates, semantic similarity has little left to discriminate on,
and lexical overlap becomes the more useful signal.

**Fix.** Run both and fuse with **Reciprocal Rank Fusion**:

```
RRF(d) = Σ  1 / (k + rankₛ(d))        k = 60
        s∈S
```

RRF throws away the scores and keeps only the ranks, which neatly sidesteps the
fact that a cosine similarity of 0.82 and a BM25 score of 14.3 are not comparable
quantities. A document that both systems rank 3rd beats one that a single system
ranks 1st.

This needed a second Atlas index — a `search` type index for full text, alongside
the `vectorSearch` one. Easy to forget, because vector search keeps working
perfectly well without it and nothing tells you the lexical leg is dead.

**What I learned.** Two things, and the second one matters more. "Semantic search is
better than keyword search" is the sentence that keeps people from building hybrid
search — they're good at different things. But also: I had a confident story about
*why* my change worked, and the measurement said the story was wrong while the
change was right. If I'd shipped without the eval harness I'd have gone into an
interview and explained it incorrectly with total conviction.

---

## Chapter 9 — The reranker, and why retrieval order isn't retrieval quality

Hybrid search decides *which* documents come back. It's much weaker at deciding
what order they come back in — RRF is rank-based, so it rewards agreement, not
relevance.

The retriever's embedding model is a **bi-encoder**: it embeds the query and the
document separately and compares the two vectors afterwards. It never actually
looks at them together. That's what makes it fast enough to index 4,880 documents,
and it's also its ceiling.

A **cross-encoder** takes `(query, document)` as one input and lets every query
token attend to every document token. Far more accurate, and far too slow to run
over the whole collection — which is fine, because it doesn't have to.

```
hybrid search  ->  25 candidates  ->  cross-encoder  ->  best 6  ->  LLM
```

`cross-encoder/ms-marco-MiniLM-L-6-v2`, about 250 ms on CPU for 25 pairs.

Over-fetching 25 matters: the reranker can only reorder what it's given. If the
right document is at rank 20 and you retrieved 6, it never gets a chance.

**What the numbers actually say.** Hit-rate doesn't move at all — 0.967 with and
without the reranker. It finds nothing new, which makes sense: it can only reorder
what hybrid search already returned. What it does is push the right record to the
top. MRR goes 0.807 → 0.903, meaning the best answer moves from around rank 3 to
rank 1.

That costs ~700 ms of the ~840 ms median query time. Whether that's worth it
depends on what you're optimizing, so it stays a config flag (`RERANK_ENABLED`)
rather than a hard-coded stage — and the eval harness can A/B it whenever the data
changes.

**The cross-encoder has its own lexical trap.** Asking *"Show me Computer
Engineering companies from 2018"*, the reranker put `SATYAM COMPUTER SERVICES LTD`
and `Advance Computer Services` at ranks 1 and 2 — because the literal token
*Computer* appears in the company name and in the query. Both are 1990s IT services
firms that have nothing to do with the question.

They didn't reach the answer, because the document-type preference from Chapter 15
demotes company profiles on a specific question like this one. But it's a good
reminder that a reranker is a language model with a language model's biases, not an
oracle — and the retrieval trace panel is what made it visible at all.

**What I learned.** Two-stage retrieval — cheap and broad, then expensive and
narrow — is the standard shape of every serious search system. But "improves
retrieval" is too vague to be a claim. *Recall* and *ordering* are different axes,
and this stage only moves one of them.

---

## Chapter 10 — TCS ate the whole context window

"Tell me about TCS" returned six documents, all TCS, five of them nearly identical
sentences differing only in college name. The model had 6 slots of context and
learned nothing from 5 of them.

**Fix.** Cap results per company after reranking (currently 3). It's a crude stand-in
for **MMR** (Maximal Marginal Relevance), which formalizes the same idea — score
each candidate on relevance *minus* similarity to what you've already selected —
but on this data the simple cap does the job.

Combined with the company-profile documents from Chapter 6, "tell me about TCS" now
gets the profile *plus* a couple of representative drives.

---

## Chapter 11 — Follow-up questions retrieved garbage

```
Me:  What package did Morgan Stanley offer?
Bot: 25.57 LPA ...
Me:  and its CGPA cutoff?
Bot: [complete nonsense]
```

`app.py` kept `st.session_state.messages` and displayed the history. It never
*used* it. Every question was embedded and searched in isolation.

"and its CGPA cutoff?" has no subject. Embedded on its own it's meaningless — the
nearest neighbours are whatever documents happen to talk about cutoffs.

**Fix.** Before retrieval, the last few turns and the follow-up go to the LLM to be
rewritten as a standalone question:

```
"and its CGPA cutoff?"  ->  "What is Morgan Stanley's CGPA cutoff?"
```

Then *that* gets embedded. One extra LLM call, only when history exists. The
rewritten question is shown in the retrieval trace so you can see it happen.

**What I learned.** Displaying chat history and *using* chat history are completely
different features. I had built the one that looks like the other.

---

## Chapter 12 — Answers with no receipts

The original prompt was:

```python
system_instructions = f"""
You are the official T&P Assistant.
Use the following data extracted from the placement records to answer.
If the data doesn't contain the answer, politely say you don't have that info.

RELEVANT DATA:
{context}
"""
```

Two problems. The context blocks were unlabelled, so even when the model was right
there was no way to check *which* record a number came from. And "politely say you
don't have that info" is a soft suggestion — the model would still rather guess than
refuse.

**Fix.** Number every context block and require citations:

```
RECORDS:
[1] Microsoft recruited at ...
[2] Morgan Stanley recruited at ...
```

with a hard rule to append `[n]` after every fact and never invent a company,
package, or cutoff. The frontend maps `[2]` back to an expandable card showing the
exact source record.

Citations turn out to be the real defence against hallucination — not because the
model becomes more honest, but because a fabricated claim has no record to point at,
and I can see that immediately.

I added three deliberately unanswerable questions to the evaluation set ("what was
the hostel fee", "which company will visit in 2027", "the placement officer's phone
number") to check it declines instead of improvising.

---

## Chapter 13 — Streamlit ran out of room

Streamlit got the first version working in an evening and I don't regret it. But:

- Every interaction re-runs the entire script top to bottom.
- No token streaming — the page sits blank for 4 seconds, then everything appears.
- Building an expandable citation card or an inspectable retrieval panel means
  fighting the framework.
- There's no API. Nothing else can ever consume this.

**Fix.** Split it.

**Backend — FastAPI.** `POST /api/chat` streams the answer over Server-Sent
Events. Retrieval metadata is sent *first*, so citation cards render while the
model is still writing. Plus `/api/health`, `/api/stats`, `/api/companies`.

SSE rather than WebSockets because the stream only goes one direction, it rides on
plain HTTP, and it reconnects on its own. One wrinkle: the browser's `EventSource`
is GET-only, so the frontend reads the stream off a `fetch` POST response body and
parses the SSE frames by hand.

**Frontend — React + TypeScript + Vite + Tailwind.** Streaming chat, expandable
citation cards, dark mode, and the piece I'm most pleased with: a **retrieval trace
panel** that shows, per question, the extracted filters, what the vector leg found,
what the BM25 leg found, which documents only one leg found, the cross-encoder
scores, and the timing of each stage.

It started as a debugging tool. It's now the thing I'd actually demo, because it
makes the pipeline visible instead of asking you to believe it.

**The bug that ate an hour.** The API streamed perfectly under `curl`. In the
browser the UI sat on "searching placement records" forever — no console error, no
failed request, backend logging a clean `200`.

`sse-starlette` separates lines with **CRLF**, so a frame ends with `\r\n\r\n`. My
hand-rolled parser scanned the buffer for `\r\n\r\n`'s more famous cousin:

```ts
let boundary = buffer.indexOf("\n\n");   // never matches "\r\n\r\n"
```

There is no `\n\n` substring inside `\r\n\r\n` — the bytes are `\r \n \r \n`. So
the parser buffered the entire response and never dispatched a single event. `curl`
hid it completely, because `curl` doesn't care where frames begin.

```ts
buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
```

Found it by dumping the raw bytes with `od -c`. Lesson: when a stream "works in
curl but not in the browser", stop reading your code and look at the actual bytes.

**Also fixed here:** both models load from disk on first use, which showed up as a
~10 second stall for whoever asked the first question. They're now warmed during
FastAPI's lifespan startup.

---

## Chapter 14 — Re-running ingestion duplicated everything

I re-ran `ingest.py` after a data fix. Document count went from 3,098 to 6,196.

`MongoDBAtlasVectorSearch.from_documents()` generates a fresh `_id` per document
every call. Every run appended a complete second copy. Retrieval then returned the
same record twice in the top-6 and I burned an evening on "why is my reranker
broken".

**Fix.** A deterministic id per document — `sha1(doc_type|company|college|year)` —
passed to `add_documents(docs, ids=...)`, which upserts.

```python
ids = [d.metadata["doc_id"] for d in documents]
store.add_documents(chunk, ids=ids[start:start + batch_size])
```

Re-running now leaves the count unchanged, and the script says so explicitly.

**What I learned.** Any pipeline you'll run more than once needs to be idempotent,
and "run it twice and check the count" is a ten-second test I should do by default.

---

## Chapter 15 — I had no idea if any of this helped

Every change up to here was justified with "this feels better". I had no numbers.
For all I knew the reranker was making things worse and costing 250 ms to do it.

**Fix.** A golden set of 33 questions across seven categories — semantic, exact
name, numeric filter, eligibility, aggregate, multi-constraint, and refusal — run
against three configurations: vector only, + BM25 hybrid, + cross-encoder rerank.

Relevance is written as a *predicate over metadata* rather than a frozen list of
document ids:

```json
{
  "id": "multi-01",
  "question": "Computer Engineering companies in 2018 paying more than 12 LPA",
  "relevant": {
    "doc_type": "record",
    "branches_contains": "Computer Engineering",
    "year": 2018,
    "package_lpa": { "gte": 12 }
  }
}
```

so the golden set survives re-ingestion instead of breaking every time ids change.

Metrics: **hit-rate@k**, **precision@k**, **MRR**. Deliberately not recall —
"companies above 10 LPA" has 235 relevant records, so `recall@6` caps at 0.026 no
matter how good retrieval is. That number would measure the question, not the
system.

**The first run was humiliating.**

| Configuration | hit-rate@6 | precision@6 | MRR |
| --- | --- | --- | --- |
| Vector only | 0.733 | 0.378 | 0.523 |
| + BM25 hybrid | 0.733 | 0.294 | 0.392 |
| + cross-encoder rerank | 0.700 | **0.394** | 0.576 |

Hybrid was *worse* than plain vector search. Reranking made hit-rate go **down**.
Every improvement I'd been so pleased with was, by the numbers, not one.

Two real bugs were hiding underneath.

**Bug one — a filter field I never declared.** The eval crashed partway through the
first mode with:

```
PlanExecutor error :: Path 'active_backlogs' needs to be indexed as filter
```

My query router could emit an `active_backlogs` filter, but that field wasn't in
the vector index's `filters` list. Atlas rejects the whole query rather than
ignoring the clause. It had never surfaced because none of my manual test questions
happened to mention backlogs. The eval set had one that did.

**Bug two — reranking undid the granularity choice.** I sorted candidates to prefer
company-profile documents for aggregate questions... and *then* reranked, which
threw the sort away. Worse, on every non-aggregate question the profile documents
were free to outrank the individual drives, so "which companies visited in 2018"
came back full of company summaries instead of 2018 drives.

The fix was one line moved and one line added: apply the document-type preference
*after* reranking (stable sort, so the reranked order survives inside each group),
and make the preference explicit in both directions — profiles for aggregate
questions, individual records for everything else.

| Configuration | hit-rate@6 | precision@6 | MRR | median latency |
| --- | --- | --- | --- | --- |
| Vector only | 0.833 | 0.556 | 0.753 | 106 ms |
| + BM25 hybrid | 0.967 | 0.600 | 0.807 | 128 ms |
| + cross-encoder rerank | 0.967 | 0.617 | **0.903** | 837 ms |

**What I learned.** This is the whole argument for evaluation in one afternoon. The
harness didn't just confirm my improvements — it found two bugs I'd shipped, and it
proved that my explanation for *why* hybrid search helped (Chapter 8) was wrong even
though the change itself was right. Without it I was doing vibes-driven development
and would never have known.

---

## What I'd do next

- The query router is regex-first — fast and predictable, but brittle for phrasing
  I didn't anticipate. A structured-output LLM call would generalize better at the
  cost of latency.
- No caching yet. Query embeddings and repeated questions are the obvious wins.
- `all-MiniLM-L6-v2` is a 2021 model. `bge-base-en-v1.5` or `e5-base-v2` score
  meaningfully better on MTEB for roughly 3× the compute — worth measuring with the
  harness that now exists.
- Real placement data instead of a synthetic eligibility layer.
- No tests and no CI. The eval harness catches retrieval regressions; nothing
  catches ordinary bugs.
