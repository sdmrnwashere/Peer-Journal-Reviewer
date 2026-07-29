"""
Streamlit front end for the LangChain PDF peer reviewer.

Run with:  streamlit run app.py
"""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

import chain as C

load_dotenv()

st.set_page_config(page_title="PDF Peer Reviewer", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Cached resources. Streamlit reruns the whole script on every interaction, so
# without these the embedding model would reload and the index would rebuild
# every time you touch a widget.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    return C.build_embeddings(model_name)


@st.cache_resource(show_spinner=False)
def get_llm(repo_id: str, max_new_tokens: int, temperature: float):
    return C.build_llm(repo_id, max_new_tokens=max_new_tokens, temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_index(file_hash: str, file_bytes: bytes, filename: str, embed_model: str,
              chunk_size: int, chunk_overlap: int):
    """Keyed on file_hash, so re-running with the same PDF is instant."""
    pages = C.load_pdf(file_bytes, filename)
    chunks = C.split_pages(pages, chunk_size, chunk_overlap)
    store = C.build_vectorstore(chunks, get_embeddings(embed_model))
    return pages, chunks, store


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    token_present = bool(
        os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    )
    if token_present:
        st.success("Hugging Face token loaded")
    else:
        st.error("No HUGGINGFACEHUB_API_TOKEN found in .env")

    llm_repo = st.text_input("LLM repo id", value=C.DEFAULT_LLM_REPO)

    embed_model = st.selectbox(
        "Embedding model (runs locally)",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
        help="MiniLM is the fastest on CPU. mpnet is more accurate and about 3x slower.",
    )

    st.divider()
    chunk_size = st.slider("Chunk size (chars)", 600, 2000, 1200, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 25)
    k = st.slider("Passages retrieved per probe", 3, 10, 5)
    n_points = st.slider("Target number of review points", 8, 22, 14)

    st.divider()
    max_new_tokens = st.slider("Max new tokens", 800, 4000, 2400, 200)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    stream_output = st.checkbox("Stream the review", value=True)

    st.divider()
    st.caption(
        "Six semantic probes are fired at the index in parallel (claims, method, "
        "data protocol, results, baselines, limitations). Their hits are merged, "
        "deduplicated and put back in document order before the review is written."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Journal peer reviewer")
st.caption(
    "Upload a manuscript. It gets chunked, embedded and indexed, then reviewed "
    "in the numbered referee-report house style."
)

uploaded = st.file_uploader("Manuscript (PDF)", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

with st.spinner("Reading, chunking and embedding the manuscript..."):
    pages, chunks, store = get_index(
        file_hash, file_bytes, uploaded.name, embed_model, chunk_size, chunk_overlap
    )
    text = C.full_text(pages)

quality = C.extraction_quality(pages)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages", len(pages))
c2.metric("Chunks", len(chunks))
c3.metric("Characters", f"{len(text):,}")
c4.metric("Mode", "Whole paper" if len(text) <= C.FULL_TEXT_BUDGET else "Retrieval")

if quality < 300:
    st.warning(
        f"Only {quality:.0f} characters per page were extracted. This PDF is "
        "probably a scan. Run it through OCR first (ocrmypdf input.pdf out.pdf) "
        "or the review will be based on almost nothing."
    )

variant = st.radio(
    "Output",
    ["Full referee report", "Strongest points only"],
    horizontal=True,
)

go = st.button("Run review", type="primary", disabled=not token_present)

if go:
    model = get_llm(llm_repo, max_new_tokens, temperature)

    with st.spinner("Extracting metadata..."):
        meta = C.safe_metadata(model, text[:6000], fallback_title=uploaded.name)
    st.session_state["meta"] = meta

    with st.spinner("Selecting evidence..."):
        evidence = C.build_evidence_chain(store, pages, k=k).invoke({"text": text})
    st.session_state["evidence"] = evidence

    context_note = (
        "The evidence below is the complete manuscript."
        if len(text) <= C.FULL_TEXT_BUDGET
        else (
            "The evidence below consists of passages retrieved from the "
            "manuscript by semantic search, separated by dividers and labelled "
            "with page numbers. Gaps between passages are omitted text, not "
            "missing content, so do not criticise the paper for something that "
            "is simply absent from this excerpt."
        )
    )

    if variant == "Full referee report":
        chain_obj = C.build_review_chain(model, guard=not stream_output)
        payload = {
            "title": meta.title,
            "venue_type": meta.venue_type,
            "domain": meta.domain,
            "evidence": evidence,
            "n_points": n_points,
            "context_note": context_note,
        }
    else:
        chain_obj = C.build_strengths_chain(model, guard=not stream_output)
        payload = {"title": meta.title, "evidence": evidence}

    st.subheader("Review")
    try:
        if stream_output:
            placeholder = st.empty()
            buffer = ""
            for piece in chain_obj.stream(payload):
                buffer += piece
                placeholder.text(C.enforce_house_style(buffer))
            review = C.enforce_house_style(buffer)
            placeholder.text(review)
        else:
            with st.spinner("Writing the review..."):
                review = chain_obj.invoke(payload)
            st.text(review)
        st.session_state["review"] = review
    except Exception as e:
        st.error(f"The model call failed: {e}")
        st.stop()

    if variant == "Full referee report":
        with st.spinner("Extracting the verdict..."):
            st.session_state["verdict"] = C.safe_verdict(model, review)

# ---------------------------------------------------------------------------
# Persisted panels
# ---------------------------------------------------------------------------

if "review" in st.session_state:
    review = st.session_state["review"]
    meta = st.session_state.get("meta")

    tab_v, tab_m, tab_e, tab_x = st.tabs(
        ["Verdict", "Metadata", "Evidence used", "Follow-ups"]
    )

    with tab_v:
        verdict = st.session_state.get("verdict")
        if verdict is None:
            st.info("No structured verdict was extracted.")
        else:
            st.metric("Recommendation", verdict.recommendation.title())
            st.write(f"Confidence: {verdict.confidence}")
            if verdict.blocking_issues:
                st.write("Blocking issues:")
                for issue in verdict.blocking_issues:
                    st.write(f"- {issue}")

    with tab_m:
        if meta:
            st.write(f"Title: {meta.title}")
            st.write(f"Venue type: {meta.venue_type}")
            st.write(f"Domain: {meta.domain}")
            if meta.datasets:
                st.write("Datasets: " + ", ".join(meta.datasets))
            if meta.claimed_contributions:
                st.write("Claimed contributions:")
                for cc in meta.claimed_contributions:
                    st.write(f"- {cc}")

    with tab_e:
        st.caption(
            "Exactly the text the model saw. If a review point looks wrong, "
            "check here first: the passage it needed may never have been retrieved."
        )
        st.text_area("Evidence", st.session_state.get("evidence", ""), height=400)

    with tab_x:
        cols = st.columns(2)
        if cols[0].button("Propose remedies for every concern"):
            model = get_llm(llm_repo, max_new_tokens, temperature)
            with st.spinner("Working out remedies..."):
                out = C.build_remedies_chain(model).invoke(
                    {
                        "title": meta.title if meta else uploaded.name,
                        "review": review,
                        "evidence": st.session_state.get("evidence", "")[:8000],
                    }
                )
            st.text(out)
        if cols[1].button("Condense, keep every point"):
            model = get_llm(llm_repo, max_new_tokens, temperature)
            with st.spinner("Condensing..."):
                out = C.build_condense_chain(model).invoke({"review": review})
            st.text(out)

    st.download_button(
        "Download review as .md",
        data=review,
        file_name=f"review_{uploaded.name.rsplit('.', 1)[0]}.md",
        mime="text/markdown",
    )
