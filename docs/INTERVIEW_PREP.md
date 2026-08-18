# Interview Revision Sheet — Placement RAG

Everything in this project that an interviewer can reasonably ask about, with the
maths, the trade-offs, and the reason each choice was made here specifically.

Paste this whole file into an LLM and ask it to quiz you.

---

## 0. Thirty-second project summary

> A retrieval-augmented question answering system over ~2,700 campus recruitment
> records. Records are rendered as sentences, embedded with `all-MiniLM-L6-v2`, and
> stored in MongoDB Atlas. A query is parsed for structured constraints (package,
> year, branch, CGPA), which become a metadata pre-filter. Retrieval runs vector
> search and BM25 full-text search in parallel and fuses them with Reciprocal Rank
> Fusion, then a cross-encoder reranks the top 25 down to 6. Gemini 2.5 Flash
> generates a grounded answer with inline citations, streamed to a React frontend
> over SSE. Retrieval quality is measured with a 33-question golden set.

Know this cold. Every follow-up question comes out of it.

---

## 1. RAG fundamentals

**What problem does RAG solve?**
An LLM only knows its training data. Fine-tuning on private data is expensive, has
to be redone whenever the data changes, and still doesn't give you provenance. RAG
keeps the knowledge outside the model: retrieve the relevant facts at query time,
put them in the prompt, and the model reasons over text it can actually see.

**The three stages**

| Stage | What happens | Where it lives here |
| --- | --- | --- |
| Indexing (offline) | load → chunk → embed → store | `rag/documents.py`, `rag/ingest.py` |
| Retrieval (online) | embed query → search → filter → rerank | `rag/retriever.py` |
| Generation (online) | stuff context into prompt → LLM → answer | `rag/chain.py` |

**Why not just put everything in the prompt?**
2,764 records × ~90 tokens ≈ 250k tokens per request. Even with a model that
accepts it, you pay for every token on every query, latency scales with context
length, and accuracy *drops* — models attend poorly to the middle of very long
contexts (see §11, lost-in-the-middle).

**RAG vs fine-tuning vs long context**
- RAG — knowledge that changes, needs citations, is too large for a prompt.
- Fine-tuning — teaching *behaviour*, style, or output format, not facts.
- Long context — small, static corpora where retrieval isn't worth the machinery.

---

## 2. Embeddings

An embedding maps text to a fixed-length vector such that semantically similar text
lands nearby. This project uses **`sentence-transformers/all-MiniLM-L6-v2`**:

| Property | Value | Why it matters |
| --- | --- | --- |
| Dimensions | 384 | 4× smaller than OpenAI's 1536 → cheaper index, faster search |
| Layers | 6 (distilled from BERT) | runs on CPU in milliseconds |
| Max sequence length | **256 word-pieces** | anything longer is silently **truncated** |
| Training | contrastive on 1B+ sentence pairs | tuned for sentence similarity, not token prediction |
| Size on disk | ~80 MB | ships fine in a container |

**The truncation trap.** MiniLM does not error on long input, it just drops the
tail. A 1,000-character chunk can exceed 256 word-pieces, so the end of the chunk
contributes nothing to the vector. This is a real bug source: retrieval quality
degrades and nothing in the logs tells you. Documents here are ~90 tokens, well
inside the limit.

**Bi-encoder vs cross-encoder** — the single most important distinction in this
project:

|  | Bi-encoder (retrieval) | Cross-encoder (reranking) |
| --- | --- | --- |
| Input | query and document **separately** | query and document **together** |
| Output | two vectors, compared afterwards | one relevance score |
| Cost | documents embedded once, offline | every (query, doc) pair at query time |
| Can index? | yes — ANN over millions | no |
| Accuracy | good | noticeably better |
| Used here | `all-MiniLM-L6-v2` | `ms-marco-MiniLM-L-6-v2` |

The bi-encoder never actually compares the query to the document — it compares two
independent summaries. The cross-encoder lets every query token attend to every
document token. That's why the standard pattern is bi-encoder for recall over
thousands, cross-encoder for precision over the surviving ~25.

**Why normalize embeddings?** With unit-length vectors, `cos(a,b) = a · b`. The
expensive division disappears and the vector database can use plain dot product.
Set once in `rag/embeddings.py` via `normalize_embeddings=True`.

**The mismatch failure.** If ingestion and query time use different models — or the
same model with different normalization — the vectors still have the right shape,
so nothing crashes. Similarity scores just become meaningless and retrieval returns
noise. This is why the embedding model is constructed in exactly one place.

---

## 3. Cosine similarity and friends

**Formula**

```
                 A · B          Σ aᵢbᵢ
cos(A, B) = ───────────── = ──────────────────
              ‖A‖ ‖B‖       √(Σaᵢ²) · √(Σbᵢ²)
```

Range −1 to 1; for typical sentence embeddings it lands in roughly 0 to 1.
MongoDB Atlas rescales to `(1 + cos)/2` so scores come back in [0, 1].

**Why cosine and not Euclidean?**
Cosine measures *direction* only, ignoring magnitude. Embedding magnitude tends to
track text length and frequency rather than meaning, so ignoring it is what you
want. On normalized vectors the two are monotonically related anyway:

```
‖A − B‖² = ‖A‖² + ‖B‖² − 2A·B = 2 − 2cos(A,B)     (when ‖A‖ = ‖B‖ = 1)
```

so ranking by cosine and ranking by Euclidean distance give the same order. The
choice matters only when vectors are *not* normalized.

**Dot product** — fastest, but unnormalized magnitude leaks into the score.
Use it only when vectors are already unit length (which is the case here).

**The killer limitation, and the reason this project has a query router.**
`"companies above 10 LPA"` and `"companies above 5 LPA"` produce nearly identical
vectors — cosine ≈ 0.95. The embedding encodes the *topic*, not the *threshold*.
No embedding model, no chunking strategy, and no amount of `k` tuning fixes this.
Numeric constraints must be extracted from the text and pushed to the database as
a filter. That is what `rag/query_router.py` does.

---

## 4. Vector search: exact vs approximate

**Exact (flat / brute force)** — compare the query to every vector. Perfect recall,
O(N·d). At 4,730 × 384 that is ~1.8M multiply-adds, genuinely fine. At 10M vectors
it is not.

**Approximate Nearest Neighbour (ANN)** — trade a little recall for a lot of speed.

**HNSW** (Hierarchical Navigable Small World), what Atlas uses:
- A multi-layer graph. Upper layers are sparse (long-range "express lanes"), lower
  layers dense.
- Search enters at the top, greedily walks toward the query, descends a layer,
  repeats. Roughly O(log N).
- Key parameters: `M` (edges per node — higher = better recall, more memory),
  `efConstruction` (build-time candidate list), `efSearch` (query-time candidate
  list — the main recall/latency dial).

**Other index families worth naming:** IVF (cluster into Voronoi cells, probe
`nprobe` of them), PQ / Product Quantization (compress vectors into codebooks —
big memory win, some accuracy loss), and combinations like IVFPQ.

**`oversampling_factor`** — Atlas fetches `k × factor` candidates internally before
applying filters and returning `k`. Default 10. Raising it recovers recall when a
selective pre-filter would otherwise leave you with too few survivors.

---

## 5. BM25 — the lexical half

**Best Matching 25**, the default ranking function in Lucene, Elasticsearch, and
MongoDB Atlas Search. Purely lexical: it matches *words*, not meaning.

```
                                     f(qᵢ, D) · (k₁ + 1)
BM25(D, Q) = Σ  IDF(qᵢ) · ────────────────────────────────────────
             i∈Q                                    |D|
                            f(qᵢ, D) + k₁ · (1 − b + b · ─────)
                                                         avgdl
```

| Symbol | Meaning |
| --- | --- |
| `f(qᵢ, D)` | how many times term *i* appears in document *D* |
| `\|D\|` | length of *D* in words |
| `avgdl` | average document length in the collection |
| `k₁` | term-frequency saturation, typically **1.2–2.0** |
| `b` | length-normalization strength, typically **0.75** |

```
              N − n(qᵢ) + 0.5
IDF(qᵢ) = ln ─────────────────── + 1
                n(qᵢ) + 0.5
```
`N` = total documents, `n(qᵢ)` = documents containing the term.

**The two ideas that make BM25 better than TF-IDF**

1. **Saturation (`k₁`).** In TF-IDF, a term appearing 100 times scores 100×. In
   BM25 the term-frequency factor asymptotes at `k₁ + 1`, so the 10th occurrence
   adds far less than the 2nd. `k₁ → 0` makes it binary presence/absence;
   `k₁ → ∞` makes it linear like TF-IDF.
2. **Length normalization (`b`).** Long documents contain more words by accident.
   `b = 1` fully normalizes by length, `b = 0` not at all, `b = 0.75` is the
   empirical sweet spot.

**Why this project needs it.** Vector search is bad at rare proper nouns.
`"Zeus Learning"` is a low-frequency token combination that MiniLM has essentially
no signal for, so the nearest neighbours come back as other generic ed-tech-ish
records. BM25 nails it, because `Zeus` has a very high IDF — it appears in almost
no other document. Conversely, BM25 completely misses `"which companies hire
software developers"` matching a record that says `Product Engineer`. The two
methods fail on opposite query types, which is exactly why you fuse them.

**Note the naming.** It is BM**25** (the 25th "Best Matching" formula in the
Okapi series) — not "BM 2.5".

---

## 6. Hybrid search and Reciprocal Rank Fusion

Vector scores (0–1 cosine) and BM25 scores (unbounded, corpus-dependent) are not
comparable. You cannot average them without normalizing, and normalization is
brittle because BM25's range shifts with the query.

**RRF sidesteps the problem entirely by throwing away the scores and keeping only the ranks.**

```
                       1
RRF(d) = Σ  wₛ · ─────────────
        s∈S       k + rankₛ(d)
```

`S` = the retrieval systems being fused, `rankₛ(d)` = position of *d* in system *s*
(1-based), `k` = a penalty constant (**60** by default, from the original Cormack
et al. paper), `wₛ` = optional per-system weight.

**Worked example** — `k = 60`:

| Document | Vector rank | BM25 rank | RRF score |
| --- | --- | --- | --- |
| A | 1 | — | 1/61 = 0.0164 |
| B | 3 | 2 | 1/63 + 1/62 = 0.0320 |
| C | — | 1 | 1/61 = 0.0164 |

B wins: neither system ranked it first, but both liked it. That "agreement beats
brilliance" behaviour is the whole point of RRF.

**Why k = 60?** It flattens the difference between the top ranks. `1/61` vs `1/62`
is a 1.6% gap, so a document has to be found by *multiple* systems to overtake one
that a single system loved. Lowering `k` sharpens the emphasis on rank 1; raising
it flattens further. In this project it is exposed as `vector_penalty` and
`fulltext_penalty` — raising one reduces that leg's influence.

**Properties:** no score normalization needed, no training, robust across
different score distributions, extends to any number of retrievers.

**Alternative:** weighted score fusion (`α · norm(vec) + (1−α) · norm(bm25)`).
Can beat RRF if you tune α per corpus. Needs tuning; RRF does not.

---

## 7. Chunking

**The trade-off.** Small chunks → precise retrieval, but facts get split across
boundaries. Large chunks → complete context, but the embedding is a blurry average
of several topics and dilutes the signal.

**Common strategies**

| Strategy | Description | Good for |
| --- | --- | --- |
| Fixed-size | N characters, fixed overlap | uniform prose |
| Recursive character | split on `\n\n`, then `\n`, then ` ` | general documents |
| Document-aware | split on markdown headings, code functions | structured text |
| Semantic | split where consecutive-sentence embedding similarity drops | expensive, high quality |
| **Row-aware** | one chunk per structured record | **CSV / tabular data — used here** |

**What went wrong originally.** The first version loaded the CSV with
`CSVLoader` (one document per row) and then ran
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` over the
result. Each row is ~40 characters, so the splitter happily packed ~20 unrelated
companies into a single 1,000-character chunk. Retrieving that chunk to answer a
question about one company handed the model 19 irrelevant companies as context —
and the overlap duplicated rows across chunk boundaries on top of it.

**The fix.** For structured data the record *is* the chunk. No splitter is involved
at all. `rag/documents.py` renders one document per drive.

**Why prose instead of CSV syntax?** `TCS,Pillai,Pune,2016,3.4` embeds poorly —
sentence-transformers were trained on natural language. The same facts written as
*"TCS recruited at Pillai College of Engineering (Pune) in 2016 … Package offered:
3.4 LPA"* produce a much more useful vector, and BM25 gets real tokens to match on.

**Multi-granularity indexing.** Row documents can only answer row questions.
*"How many times did Infosys visit?"* requires seeing all rows at once, so a second
document type — one aggregated profile per company — is indexed alongside. The
router detects aggregate phrasing and prefers those documents.

---

## 8. Metadata filtering

**Pre-filter vs post-filter**

| | Pre-filter | Post-filter |
| --- | --- | --- |
| When | during the ANN traversal | after top-k is returned |
| Result count | always returns k | may return far fewer than k |
| Index support | must declare filter fields in the index | none needed |
| Used here | ✅ | for cosmetic trimming only |

Post-filtering is the classic beginner bug: retrieve top-6, then filter for
`package > 10`, and get back 1 result — because the filter was applied to a list
that was already chosen without knowing about it. Atlas pre-filters *inside* the
HNSW traversal, so all 6 returned documents satisfy the constraint.

**Cost:** a very selective pre-filter makes the graph sparse and hurts recall,
which is what `oversampling_factor` compensates for.

**Filterable fields must be declared** in the vector index definition
(`filters=[...]` in `scripts/build_indexes.py`). Filtering on an undeclared path
does not degrade gracefully — Atlas rejects the entire query:

```
PlanExecutor error :: Path 'active_backlogs' needs to be indexed as filter
```

Anything the query router can emit therefore has to be in that list, and the index
has to be rebuilt when the list changes.

**Zero-match handling.** If a filter matches no documents, this project logs it,
drops the filter, and retries, rather than answering "I don't know" to a question
the corpus can partly answer. The trace surfaces `filter_relaxed: true` so the
behaviour is visible rather than magic.

---

## 9. Query understanding

**Filter extraction.** Regex first (fast, deterministic, no token cost), LLM
fallback for phrasings the regexes miss. Handles:
`above/over/more than X LPA` · `between X and Y` · `X+ LPA` · years and year
ranges · branch aliases (`CS`, `CSE`, `comps`, `ENTC`, `EXTC`) · CGPA · backlogs.

**The eligibility subtlety.** *"I have 7.2 CGPA, what am I eligible for?"* must
become `cgpa_cutoff <= 7.2`, **not** `cgpa_cutoff == 7.2`. The number in the
question describes the *student*; the field describes the *company's bar*. Getting
the direction of the inequality wrong is a silent correctness bug that returns
plausible-looking results. Good thing to volunteer in an interview.

**Conversational condensing.** *"and its CGPA cutoff?"* embeds to nothing useful —
there is no subject in it. Before retrieval, the last few turns plus the follow-up
are sent to the LLM and rewritten into a standalone question
(*"What is Morgan Stanley's CGPA cutoff?"*). Without this step, multi-turn chat
retrieves garbage on every follow-up.

**HyDE (Hypothetical Document Embeddings)** — an alternative worth naming: ask the
LLM to *write a fake answer*, embed that, and search with it. A hypothetical answer
sits closer in embedding space to real answers than a question does. Costs an extra
LLM call; not used here because hybrid search already covers the failure mode HyDE
targets.

**Other techniques:** multi-query (generate N paraphrases, retrieve for each, union
the results), step-back prompting (ask a more general question first), query
decomposition (split a compound question into sub-questions).

---

## 10. Reranking

Retrieve 25 candidates → cross-encoder scores all 25 → keep the best 6.

**Why over-fetch?** The reranker can only reorder what retrieval gave it. If the
right document is at rank 20, retrieving 6 means the reranker never sees it. Wider
candidate set = higher ceiling.

**Why not rerank everything?** A cross-encoder forward pass per (query, document)
pair. 25 pairs ≈ 200–400 ms on CPU. 4,730 pairs would be ~40 seconds.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on MS MARCO passage
ranking. Output is an unbounded logit, not a probability — only the *ordering* is
meaningful, so don't threshold on the raw value.

**Diversity capping.** After reranking, at most 3 records per company are kept.
Without it, a query like *"tell me about TCS"* fills all 6 slots with near-identical
TCS rows and the model has nothing to compare against. This is a cheap stand-in for
**MMR** (Maximal Marginal Relevance), which formalizes it:

```
MMR = argmax [ λ · sim(d, q) − (1 − λ) · max sim(d, dⱼ) ]
       d∈R\S                            dⱼ∈S
```
λ = 1 is pure relevance, λ = 0 is pure diversity.

---

## 11. Generation

**Prompt structure:** role → hard rules → numbered context blocks → question.
Numbering the blocks `[1] [2] [3]` is what makes citation possible: the model is
told to append the block number after every fact, and the frontend maps those
numbers back to source records.

**Temperature 0.1.** This is an extraction-and-summarization task with one correct
answer. Creativity is a defect. Not 0.0 exactly, so phrasing stays natural.

**Grounding rules.** The prompt explicitly instructs the model to say it does not
know rather than invent a company or package. Combined with citations, a reviewer
can check any claim against the record it came from.

**Lost in the middle.** Models attend most reliably to the *beginning* and *end* of
a long context; accuracy on facts placed in the middle drops measurably (Liu et
al., 2023). Two mitigations here: keep the context small (6 documents, not 50), and
put the best-reranked documents first.

**Streaming.** The answer is streamed token by token over **Server-Sent Events**.
Time-to-first-token is what users actually perceive as speed, and it is dominated
by retrieval, not generation.

**SSE vs WebSocket:** SSE is one-directional (server → client), rides on plain
HTTP, reconnects automatically, and needs no special infrastructure. WebSockets are
bidirectional and heavier. Chat completion only ever streams one way, so SSE is the
right tool. Note `EventSource` is GET-only, which is why the frontend reads the
stream off a `fetch` POST response body and parses SSE frames by hand.

---

## 12. Evaluation

**Retrieval metrics**

| Metric | Definition | When to use |
| --- | --- | --- |
| **hit-rate@k** | fraction of queries with ≥1 relevant doc in top-k | the "did it work at all" metric |
| **precision@k** | relevant retrieved / k | when many documents are relevant |
| **recall@k** | relevant retrieved / total relevant | when few documents are relevant |
| **MRR** | mean of 1/(rank of first relevant) | when the top result is what matters |
| **nDCG@k** | discounted gain, normalized by ideal ranking | graded (not binary) relevance |

`MRR = (1/|Q|) Σ 1/rankᵢ` — first relevant at rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25.

`DCG@k = Σ relᵢ / log₂(i + 1)`, then `nDCG = DCG / IDCG`.

**Why this project reports hit-rate, precision and MRR but not recall.** Some
golden questions ("companies above 10 LPA") have 235 relevant records. `recall@6`
caps at 6/235 = 0.026 no matter how perfect the retrieval is, so the number would
measure the question, not the system.

**Generation metrics**
- **Faithfulness / groundedness** — is every claim supported by the retrieved
  context? Catches hallucination.
- **Answer relevance** — does it actually address the question?
- **Context precision / recall** — RAGAS-style, evaluates the retrieval stage
  through the lens of the final answer.

**LLM-as-judge:** cheap and scalable, but biased — toward longer answers, toward
its own outputs, and sensitive to option order. Fine for relative comparison
between configurations, weak as an absolute score.

**The golden set here** is 33 questions across seven categories (semantic, exact
name, numeric filter, eligibility, aggregate, multi-constraint, refusal).
Relevance is a *predicate over metadata*, not a frozen list of document ids, so the
golden set survives re-ingestion. Refusal cases check that the system declines
questions the corpus cannot answer instead of inventing something.

---

## 13. MongoDB Atlas specifics

**`$vectorSearch`** (aggregation stage):
```javascript
{ $vectorSearch: {
    index: "vector_index",
    path: "embedding",
    queryVector: [...384 floats],
    numCandidates: 250,      // ANN candidate pool — must be >= limit
    limit: 25,
    filter: { package_lpa: { $gte: 10 } }   // pre-filter
}}
```

**`$search`** — full-text (BM25) via Atlas Search / Lucene.

**`$rankFusion`** — native server-side hybrid search in MongoDB 8.1+, which does
RRF inside the database. This project fuses in the application layer via LangChain's
`MongoDBAtlasHybridSearchRetriever`, which keeps the penalty constants tunable per
query and makes the fusion inspectable in the retrieval trace.

**Two indexes are required.** `vector_index` (type `vectorSearch`) and `text_index`
(type `search`). Forgetting the second one is easy to miss because vector search
keeps working — you just silently lose every exact-name query. Free-tier M0 allows
3 search indexes.

**Score access:** `{"$meta": "vectorSearchScore"}` and `{"$meta": "searchScore"}`.

**Storage layout:** LangChain flattens metadata to top-level fields, so a document
is `{_id, text, embedding, company, year, package_lpa, ...}` and filters use plain
field paths.

---

## 14. Edge cases this system handles

| # | Edge case | Behaviour |
| --- | --- | --- |
| 1 | `"above 10 LPA"` — threshold not embeddable | regex → `package_lpa: {$gte: 10}` pre-filter |
| 2 | Aggregate question ("which years did X recruit?") | company-profile documents + BM25; 0.50 -> 1.00 hit-rate |
| 3 | Filter matches zero documents | logged, dropped, retried; surfaced in trace |
| 4 | Follow-up with no subject (*"and its cutoff?"*) | LLM condenses using prior turns |
| 5 | One company floods top-k | max 3 records per company |
| 6 | Profile documents crowding out individual drives | document-type preference applied *after* reranking |
| 7 | Question the corpus cannot answer | grounded prompt refuses; 3 golden-set cases test it |
| 8 | Re-running ingestion | deterministic SHA-1 doc ids → upsert, not duplicate |
| 9 | `"L & T Infotech"` / `"TCS"` vs `"tcs"` | spacing, casing and trailing `(2014)` normalised at ingest — 2,116 -> 1,966 companies |
| 10 | Header rows leaked into the CSV | filtered as junk during enrichment (334 rows dropped) |
| 11 | Cold start — models load on first query | warmed during FastAPI lifespan startup |
| 12 | Student CGPA vs company cutoff direction | `cgpa_cutoff <= student_cgpa`, not `==` |
| 13 | Embedding config drift between ingest and query | single construction point in `rag/embeddings.py` |
| 14 | Chunk longer than 256 word-pieces | record documents kept ~90 tokens |
| 15 | Filter path missing from the index definition | Atlas rejects the whole query; every router-emitted field is declared in `FILTERABLE_FIELDS` |

---

## 15. Scaling and production questions

**"This works for 5k documents. What about 5 million?"**
- Embedding: batch on GPU, or move to a hosted embedding API.
- Index: HNSW already gives O(log N) search; tune `efSearch` for the recall you need.
- Reranking becomes the bottleneck at high QPS → cache, or use a distilled reranker.
- Shard by a natural key (college, year) and search shards in parallel.

**"How do you keep the index fresh?"**
Deterministic document ids make ingestion idempotent, so re-running is safe. For
incremental updates, track a content hash per record and only re-embed changed
rows. For deletes, reconcile ids present in the source against ids in the collection.

**"Where's the latency?"** Query embedding ~10 ms · Atlas hybrid search ~100–200 ms ·
cross-encoder rerank ~200–400 ms (CPU) · Gemini first token ~500–900 ms. The
reranker is the biggest thing you control, which is why it's config-toggleable and
measured rather than assumed.

**Caching layers:** query embeddings (LRU), full retrieval results for repeated
questions, and prompt caching on the LLM side for the static system prompt.

**Cost:** embeddings and reranking run locally, so the only per-query cost is
Gemini. Small context (6 documents) keeps input tokens low.

---

## 16. Questions you should expect

**"Why hybrid search instead of just vector search?"**
The textbook answer is that they fail on opposite query types — vector search is
weak on rare proper nouns, BM25 is weak on paraphrases. That is true in general, but
worth answering from the measurements rather than the theory: on this corpus
exact-name questions were *already* at 1.00 for pure vector search, because each
document is a sentence containing the company name verbatim. Hybrid earned its place
on **aggregate** questions (0.50 -> 1.00) and **multi-constraint** questions
(0.33 -> 1.00). Company-profile documents repeat the company name in a short
document, which is what BM25's term frequency rewards and what a cosine average
dilutes; and once a pre-filter has narrowed to a few dozen candidates, semantic
similarity has little left to discriminate on.

Being able to say "I expected X, measured Y, and here is why" is a stronger answer
than reciting the general case.

**"Why a cross-encoder if you already have hybrid search?"**
Hybrid search decides *which* documents come back; it doesn't order them well.
Fusion is rank-based, so a document ranked 3rd by both systems beats one ranked 1st
by only one — good for recall, indifferent to precision at the top. The measurements
show exactly that: adding the reranker leaves hit-rate unchanged at 0.967 (it finds
nothing new — it can only reorder what it was given) but lifts MRR from 0.807 to
0.903, moving the best record from roughly rank 3 to rank 1. It costs ~700 ms of an
~840 ms query, which is why it is a config flag rather than a fixed stage.

**"How do you know it's better?"** 33-question golden set, three configurations,
hit-rate / precision@k / MRR measured per configuration and per question category.
Results are in the README. The harness also caught two bugs I had already shipped —
a filter field missing from the index definition, and a document-type preference
being applied before reranking instead of after — neither of which showed up in
manual testing.

**"How do you prevent hallucination?"** Grounded prompt with explicit refusal
instruction, mandatory inline citations mapped back to source records, temperature
0.1, and refusal test cases in the golden set. Citations are the real defence — a
fabricated claim has no record to point at.

**"What would you do differently / what's next?"**
Honest answers: the query router is regex-first, which is fast but brittle for
unusual phrasing — a structured-output LLM call would generalize better. There is no
caching layer yet. `all-MiniLM-L6-v2` is a 2021 model; `bge-base-en-v1.5` or
`e5-base-v2` would score better on MTEB at ~3× the compute. And package/CGPA data
in the demo is synthetic, so the retrieval machinery is real but the numbers are not.

**"Why MongoDB Atlas instead of Pinecone / Weaviate / pgvector?"**
The records are documents with rich metadata, and Atlas stores the operational data
and the vectors in the same place — no sync job between a primary database and a
separate vector store. It also provides BM25 full-text search on the same
collection, which is what makes hybrid retrieval possible without a second system.

---

## 17. Glossary speed-run

**ANN** approximate nearest neighbour · **BM25** Okapi lexical ranking function ·
**Bi-encoder** encodes query and doc separately · **Cross-encoder** scores the pair
jointly · **Chunking** splitting source data into indexable units · **Cosine
similarity** angle between vectors · **DCG/nDCG** rank-discounted relevance gain ·
**Embedding** dense vector representation of text · **Faithfulness** whether an
answer is supported by its context · **Groundedness** same idea, used
interchangeably · **HNSW** hierarchical graph ANN index · **HyDE** search using a
hypothetical generated answer · **IDF** inverse document frequency · **LCEL**
LangChain Expression Language · **Lost in the middle** attention degradation in long
contexts · **MMR** maximal marginal relevance · **MRR** mean reciprocal rank ·
**Pre-filter** metadata filter applied during ANN traversal · **PQ** product
quantization · **RAG** retrieval-augmented generation · **Reranking** second-stage
precision scoring · **RRF** reciprocal rank fusion · **Semantic search** retrieval
by meaning · **SSE** server-sent events · **TF-IDF** term frequency × inverse
document frequency · **Vector store** database with ANN index support.
