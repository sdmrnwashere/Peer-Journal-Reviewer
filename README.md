# JPR: Journal Peer Reviewer

Upload a manuscript, get back a referee report. The paper is chunked, embedded
with a local sentence-transformer, indexed in FAISS, and reviewed by
Llama-3.1-8B through the Hugging Face inference endpoint. The output follows a
fixed house style: one flat numbered list, no headings, no em-dashes, and an
explicit reject / major revision / minor revision / accept at the end.

I wrote this because reading a 12-page submission and writing structured
comments takes me an afternoon, and a first pass that catches the obvious
numerical inconsistencies takes about ninety seconds here.

## Why the embedding model is doing real work

A serverless Llama endpoint will not swallow a 40,000-character journal paper
in one request. The usual workaround is to truncate, which means the model
reviews your introduction and hallucinates the rest.

Instead, the paper goes into a FAISS index and six probes are fired at it in
parallel, one per dimension of a review:

| Probe | What it pulls out |
|---|---|
| `problem_and_claims` | motivation, research gap, numbered contributions |
| `method` | architecture, equations, loss, hyperparameters, parameter counts |
| `data_and_protocol` | dataset sizes, splits, folds, preprocessing, augmentation |
| `results` | accuracy tables, confusion matrices, ablations, error bars |
| `baselines` | comparisons against prior published methods |
| `limits_and_admin` | limitations, complexity, ethics approval, code availability |

Retrieval is MMR rather than plain cosine similarity, with `lambda_mult=0.5`.
Plain similarity search returns six paragraphs that all say the same thing about
the method; MMR trades a little relevance for coverage, which is what a review
needs. The hits are then deduplicated by chunk id and sorted back into document
order, so the model reads the paper roughly front to back instead of in
relevance order.

Short papers skip all of this. A `RunnableBranch` checks the character count and
passes the whole text through if it fits under 20,000 characters, because
retrieval on a four-page paper only loses information.

## Setup

```bash
git clone <your-repo> pdf-peer-reviewer
cd pdf-peer-reviewer

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # then paste your HF token into it
```

Two things to expect on the first install. `sentence-transformers` pulls in
PyTorch, which is a 2 to 2.5 GB download on Windows. And the first time you
click Run, MiniLM downloads about 90 MB of weights into your Hugging Face
cache. After that the embedding step is offline and takes a couple of seconds
per paper on a CPU.

You also need to accept the Llama 3.1 license on the model page once, otherwise
the endpoint returns a 403 with a fairly unhelpful message.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

There is a CLI too, if you want to batch through a folder of submissions:

```bash
python chain.py path/to/paper.pdf
```

## Files

```
app.py          Streamlit UI, caching, streaming, download button
chain.py        loading, splitting, embedding, retrieval, the chains
prompts.py      the reviewer persona and the house-style rules
requirements.txt
.env.example
```

The split is deliberate. `prompts.py` is the file you will actually edit: if the
reviews come back too soft, or you want a different checklist, that is where the
checklist lives. `chain.py` should stay stable.

## The chain, concretely

```
PDF bytes
  -> PyPDFLoader
  -> RecursiveCharacterTextSplitter (1200 / 150 by default)
  -> HuggingFaceEmbeddings (local, CPU)
  -> FAISS
  -> RunnableBranch: whole text if short, else RunnableParallel of 6 retrievers
  -> merge, dedupe, restore page order
  -> PromptTemplate -> ChatHuggingFace -> StrOutputParser
  -> RunnableLambda (style enforcement)
  -> PydanticOutputParser -> Verdict
```

Two structured-output schemas, both parsed with `PydanticOutputParser`:

- `PaperMeta` reads the front matter for title, venue type, field, claimed
  contributions and datasets.
- `Verdict` reads the finished review back and pulls out the recommendation as
  a `Literal["reject", "major revision", "minor revision", "accept"]`, plus the
  blocking issues.

An 8B model returns malformed JSON often enough that both parsers are wrapped in
`safe_metadata` and `safe_verdict`, which fall back to a regex over the raw text
rather than crashing the app mid-review.

The style enforcement at the tail is a regex pass, not another model call. It
strips markdown fences, headings, bold, italics and chatbot preambles, and
converts em-dashes and en-dashes into commas while leaving numeric ranges like
2019-2024 alone. The prompt already asks for all of this; the regex is there
because an 8B model ignores it about one time in three. It only touches
punctuation and markup, never the content of a claim.

## Settings worth knowing about

`Chunk size` at 1200 characters with 150 overlap is a compromise. Smaller
chunks retrieve more precisely but split tables and equations down the middle,
which matters here because catching a confusion matrix that does not sum to the
test-set size is one of the main things this is for. If your papers are
table-heavy, push it to 1600.

`Passages retrieved per probe` at 5 gives 30 candidate passages before
deduplication, which usually lands around 18,000 to 22,000 characters of
evidence. Raising it past 7 starts hitting the context limit and the endpoint
truncates from the front, which silently removes your instructions.

`Temperature` at 0.2. Higher and the model starts inventing table numbers.

The three embedding models trade speed for retrieval quality. MiniLM is the
default at 384 dimensions and roughly 2 seconds for a 20-page paper on an
i5. bge-small is noticeably better on technical text for about double the time.
mpnet is 768 dimensions and around 3x MiniLM.

## Known limits

It cannot read scanned PDFs. The app warns you when text extraction comes back
below 300 characters per page; run those through `ocrmypdf in.pdf out.pdf`
first.

Figures are invisible to it. Every comment about a figure is inferred from the
caption and the surrounding text, so it will not catch an axis label that is
wrong or a plot that contradicts its table.

It does not search the literature, so its novelty judgements are only as good as
the related-work section the authors wrote. If you want the version that checks
novelty claims against what was actually published in 2024 and 2025, add a
`TavilySearchResults` tool and a second chain that queries on
`meta.claimed_contributions`.

And the obvious one: this is a first pass. It is good at arithmetic
inconsistencies, missing ablations, absent significance tests and unsupported
abstract claims. It is not good at judging whether an idea is interesting. Use
it to clear the mechanical objections before you spend your own attention on the
science.

## Streamlit or a plain HTML front end?

Use Streamlit. Concretely, for this app:

Everything in `app.py` that touches the browser is about 180 lines. The same
thing in HTML needs a FastAPI or Flask backend, a multipart upload endpoint, a
job queue or a server-sent-events channel so the review can stream instead of
hanging for ninety seconds, session handling so the FAISS index survives between
requests, and your own CSS. Call it 600 to 800 lines across four or five files
to reach the same place. The file uploader, the spinner, the token streaming,
the tabs and the download button are all one-liners in Streamlit.

Streamlit also gives you `@st.cache_resource` for free, which matters more than
it sounds. The embedding model is a few hundred MB in RAM and the FAISS index
takes a few seconds to build. Caching them keyed on the file hash means changing
a slider does not rebuild anything. In a hand-rolled app you write that layer
yourself.

The catch you should know about: Streamlit re-executes the entire script top to
bottom on every widget interaction. That is why the index build is behind
`@st.cache_resource` and every result lives in `st.session_state`. If you add
features and things start recomputing unexpectedly, that is why.

Switch to HTML plus FastAPI when one of these becomes true. You need more than
one person using it at once with separate sessions, since Streamlit's model gets
awkward there. You need authentication or you are putting it on a public URL.
You need it embedded inside an existing lab website. Or you need a layout
Streamlit cannot express, like a side-by-side PDF viewer with the review pinned
next to the page it refers to, which is honestly a good enough reason on its own.

For local use on one machine, none of those apply, so the extra work buys you
nothing. Start with Streamlit. If it outgrows that, `chain.py` is already
independent of the UI, so porting means writing a new front end against the same
functions rather than rewriting the pipeline.

## Troubleshooting

**403 or "model requires accepting the license"**: go to the Llama 3.1 model
page on Hugging Face, accept the terms, and make sure your token has read
access.

**`StopIteration` or an empty response from the endpoint**: serverless
inference for a given model is not always warm. Wait a minute and retry, or
switch `repo_id` to `mistralai/Mistral-7B-Instruct-v0.3` in the sidebar.

**The review stops mid-sentence**: you hit `max_new_tokens`. Raise it, or drop
the target number of review points.

**Input validation error about token count**: the evidence is too long for the
endpoint's context window. Lower `Passages retrieved per probe`, or lower
`EVIDENCE_BUDGET` in `chain.py`.

**FAISS fails to install on Windows**: `pip install faiss-cpu` needs Python
3.9 to 3.12. On 3.13, either downgrade or swap FAISS for Chroma, which is a
two-line change in `build_vectorstore`.
