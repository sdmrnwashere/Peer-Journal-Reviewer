"""
LangChain pipeline for the PDF peer reviewer.

Flow
----
    PDF  ->  PyPDFLoader  ->  RecursiveCharacterTextSplitter  ->  HF embeddings
         ->  FAISS  ->  RunnableBranch (whole paper OR multi-probe retrieval)
         ->  RunnableParallel (metadata + evidence)
         ->  review prompt  ->  ChatHuggingFace  ->  StrOutputParser
         ->  RunnableLambda (house-style enforcement)
         ->  PydanticOutputParser (structured verdict)

The embedding model is what lets this work on papers that are far longer than
the chat context window: instead of truncating the manuscript, we index it and
pull the passages that matter for each dimension of the review.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from prompts import (
    METADATA_PROMPT,
    FULL_REVIEW_PROMPT,
    STRENGTHS_PROMPT,
    REMEDIES_PROMPT,
    CONDENSE_PROMPT,
    VERDICT_PROMPT,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LLM_REPO = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# If the whole manuscript fits in this many characters we skip retrieval and
# feed the model everything. Roughly 4 chars per token, so 20000 chars is about
# 5k tokens of evidence, which leaves plenty of room for the instructions and a
# long review on an 8k-context endpoint.
FULL_TEXT_BUDGET = 20_000
EVIDENCE_BUDGET = 22_000


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class PaperMeta(BaseModel):
    """Bibliographic facts pulled off the front matter."""

    title: str = Field(description="Exact title of the manuscript")
    venue_type: Literal["journal", "conference", "preprint", "unknown"] = Field(
        description="What kind of submission this looks like"
    )
    domain: str = Field(description="Technical field, e.g. 'radar signal processing'")
    claimed_contributions: List[str] = Field(
        default_factory=list,
        description="The contributions the authors explicitly claim, verbatim or near-verbatim",
    )
    datasets: List[str] = Field(
        default_factory=list, description="Names of datasets used, empty list if none stated"
    )


class Verdict(BaseModel):
    """The editorial bottom line, pulled back out of the finished review."""

    recommendation: Literal["reject", "major revision", "minor revision", "accept"] = Field(
        description="The explicit editorial recommendation made in the review"
    )
    blocking_issues: List[str] = Field(
        default_factory=list,
        description="Short phrases naming the issues that must be fixed before acceptance",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How confident the review is in its own judgement"
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_llm(
    repo_id: str = DEFAULT_LLM_REPO,
    max_new_tokens: int = 2400,
    temperature: float = 0.2,
) -> ChatHuggingFace:
    """The endpoint from your snippet, wrapped so it uses the chat template."""
    llm1 = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.03,
    )
    return ChatHuggingFace(llm=llm1)


def build_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    """Runs locally on CPU. First call downloads the weights (about 90 MB for MiniLM)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_pdf(file_bytes: bytes, filename: str = "manuscript.pdf") -> List[Document]:
    """Write the upload to a temp file because PyPDFLoader wants a path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    for i, page in enumerate(pages):
        page.metadata["source"] = filename
        page.metadata["page"] = page.metadata.get("page", i)
    return pages


def split_pages(
    pages: List[Document], chunk_size: int = 1200, chunk_overlap: int = 150
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def build_vectorstore(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def full_text(pages: List[Document]) -> str:
    return "\n\n".join(p.page_content for p in pages)


def extraction_quality(pages: List[Document]) -> float:
    """Characters per page. Below about 300 the PDF is probably scanned images."""
    if not pages:
        return 0.0
    return len(full_text(pages)) / len(pages)


# ---------------------------------------------------------------------------
# Retrieval: one probe per review dimension
# ---------------------------------------------------------------------------

PROBES: Dict[str, str] = {
    "problem_and_claims": (
        "problem statement, motivation, research gap, stated novelty and "
        "numbered contributions of this work"
    ),
    "method": (
        "proposed method, model architecture, network layers, equations, "
        "algorithm steps, loss function, hyperparameters, parameter count"
    ),
    "data_and_protocol": (
        "dataset description, number of samples per class, subjects, train "
        "validation test split, cross validation folds, preprocessing, "
        "data augmentation"
    ),
    "results": (
        "accuracy precision recall F1 score results table, confusion matrix, "
        "ablation study, standard deviation, statistical significance test"
    ),
    "baselines": (
        "comparison with state of the art methods, related work, prior "
        "published approaches and their reported performance"
    ),
    "limits_and_admin": (
        "limitations, future work, computational complexity, inference time, "
        "ethics approval, informed consent, code availability, funding, "
        "conflict of interest"
    ),
}


def build_probe_retrieval(vectorstore: FAISS, k: int = 5, fetch_k: int = 20) -> RunnableParallel:
    """
    A RunnableParallel that fires every probe at the index at once.

    MMR rather than plain similarity, because for a review we want coverage of
    the paper, not six near-duplicate paragraphs about the same thing.
    """
    branches = {}
    for name, query in PROBES.items():
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        branches[name] = RunnableLambda(lambda _inp, q=query, r=retriever: r.invoke(q))
    return RunnableParallel(branches)


def merge_evidence(probe_hits: Dict[str, List[Document]], budget: int = EVIDENCE_BUDGET) -> str:
    """
    Deduplicate by chunk_id, restore document order, and label each passage with
    its page so the model can cite locations in the review.
    """
    seen: Dict[int, Document] = {}
    for docs in probe_hits.values():
        for d in docs:
            cid = d.metadata.get("chunk_id", hash(d.page_content))
            seen.setdefault(cid, d)

    ordered = sorted(seen.values(), key=lambda d: d.metadata.get("chunk_id", 0))

    parts, used = [], 0
    for d in ordered:
        page = d.metadata.get("page", "?")
        block = f"[page {page}]\n{d.page_content.strip()}"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# House-style enforcement (post-processing)
# ---------------------------------------------------------------------------

_DASH_RE = re.compile(r"\s*[\u2014\u2013]\s*")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", flags=re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^\s*(certainly|sure|of course|great question|here you go|absolutely)[!,.:].*?\n",
    flags=re.IGNORECASE,
)


def enforce_house_style(text: str) -> str:
    """
    The style rules are in the prompt, but an 8B model will still slip in an
    em-dash and a bold heading now and then. This is the belt to the prompt's
    braces. Purely cosmetic, it never changes a claim.
    """
    if not text:
        return text

    out = _FENCE_RE.sub("", text.strip())
    out = _PREAMBLE_RE.sub("", out)

    # Em-dash and en-dash to comma, except in numeric ranges like 2019-2024.
    def _dash(m: re.Match) -> str:
        s, e = m.start(), m.end()
        before = out[max(0, s - 1): s]
        after = out[e: e + 1]
        if before.isdigit() and after.isdigit():
            return "-"
        return ", "

    out = _DASH_RE.sub(_dash, out)

    out = _HEADING_RE.sub("", out)
    out = _BOLD_RE.sub(r"\1", out)
    out = _ITALIC_RE.sub(r"\1", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r",\s*,", ",", out)
    return out.strip()


STYLE_GUARD = RunnableLambda(enforce_house_style)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def build_metadata_chain(model: ChatHuggingFace):
    parser = PydanticOutputParser(pydantic_object=PaperMeta)
    prompt = METADATA_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | model | parser


def safe_metadata(model: ChatHuggingFace, excerpt: str, fallback_title: str) -> PaperMeta:
    """
    An 8B model does not always return clean JSON. Rather than crashing the app
    on a malformed brace, fall back to a stub so the review can still run.
    """
    try:
        return build_metadata_chain(model).invoke({"excerpt": excerpt[:6000]})
    except Exception:
        m = re.search(r"^\s*(.{15,180})\s*$", excerpt.strip(), flags=re.MULTILINE)
        return PaperMeta(
            title=(m.group(1).strip() if m else fallback_title),
            venue_type="unknown",
            domain="not stated",
            claimed_contributions=[],
            datasets=[],
        )


def build_evidence_chain(vectorstore: Optional[FAISS], pages: List[Document], k: int = 5):
    """
    RunnableBranch: short papers go through whole, long ones go through
    retrieval. The branch condition looks at the character count that the caller
    puts in the input dict.
    """
    whole_paper = RunnableLambda(lambda inp: inp["text"][:FULL_TEXT_BUDGET])

    if vectorstore is None:
        return whole_paper

    retrieval = build_probe_retrieval(vectorstore, k=k) | RunnableLambda(merge_evidence)

    return RunnableBranch(
        (lambda inp: len(inp["text"]) <= FULL_TEXT_BUDGET, whole_paper),
        retrieval,
    )


def _text_chain(prompt, model: ChatHuggingFace, guard: bool = True):
    """
    guard=False leaves the style filter off so the chain can stream token by
    token. A plain RunnableLambda at the tail of a chain buffers the whole
    stream before it fires, which would defeat the point. When streaming, the
    caller applies enforce_house_style to the accumulated buffer instead.
    """
    chain = prompt | model | StrOutputParser()
    return chain | STYLE_GUARD if guard else chain


def build_review_chain(model: ChatHuggingFace, guard: bool = True):
    return _text_chain(FULL_REVIEW_PROMPT, model, guard)


def build_strengths_chain(model: ChatHuggingFace, guard: bool = True):
    return _text_chain(STRENGTHS_PROMPT, model, guard)


def build_remedies_chain(model: ChatHuggingFace, guard: bool = True):
    return _text_chain(REMEDIES_PROMPT, model, guard)


def build_condense_chain(model: ChatHuggingFace, guard: bool = True):
    return _text_chain(CONDENSE_PROMPT, model, guard)


def build_verdict_chain(model: ChatHuggingFace):
    parser = PydanticOutputParser(pydantic_object=Verdict)
    prompt = VERDICT_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | model | parser


def safe_verdict(model: ChatHuggingFace, review: str) -> Optional[Verdict]:
    try:
        return build_verdict_chain(model).invoke({"review": review[:8000]})
    except Exception:
        # Last resort: regex the recommendation straight out of the review text.
        for label in ["major revision", "minor revision", "reject", "accept"]:
            if re.search(rf"\b{label}\b", review, flags=re.IGNORECASE):
                return Verdict(recommendation=label, blocking_issues=[], confidence="low")
        return None


# ---------------------------------------------------------------------------
# One-shot convenience path (used by the CLI, and handy for testing)
# ---------------------------------------------------------------------------

def review_pdf(
    file_bytes: bytes,
    filename: str = "manuscript.pdf",
    llm_repo: str = DEFAULT_LLM_REPO,
    embed_model: str = DEFAULT_EMBED_MODEL,
    k: int = 5,
    n_points: int = 14,
) -> Dict[str, Any]:
    pages = load_pdf(file_bytes, filename)
    text = full_text(pages)
    chunks = split_pages(pages)

    embeddings = build_embeddings(embed_model)
    store = build_vectorstore(chunks, embeddings)
    model = build_llm(llm_repo)

    meta = safe_metadata(model, text[:6000], fallback_title=filename)
    evidence = build_evidence_chain(store, pages, k=k).invoke({"text": text})

    context_note = (
        "The evidence below is the complete manuscript."
        if len(text) <= FULL_TEXT_BUDGET
        else (
            "The evidence below consists of passages retrieved from the "
            "manuscript by semantic search, separated by dividers and labelled "
            "with page numbers. Gaps between passages are omitted text, not "
            "missing content, so do not criticise the paper for something that "
            "is simply absent from this excerpt."
        )
    )

    review = build_review_chain(model).invoke(
        {
            "title": meta.title,
            "venue_type": meta.venue_type,
            "domain": meta.domain,
            "evidence": evidence,
            "n_points": n_points,
            "context_note": context_note,
        }
    )
    verdict = safe_verdict(model, review)

    return {
        "meta": meta,
        "evidence": evidence,
        "review": review,
        "verdict": verdict,
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "chars": len(text),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python chain.py path/to/paper.pdf")
        raise SystemExit(1)

    with open(sys.argv[1], "rb") as fh:
        result = review_pdf(fh.read(), os.path.basename(sys.argv[1]))

    print(result["review"])
    print("\n" + "=" * 60)
    print("VERDICT:", result["verdict"])
