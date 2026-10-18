from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
"""Single-file Streamlit interface for a multi-PDF RAG application.This initial step creates the app shell only. PDF ingestion, indexing, and
question answering will be added in later steps.
"""
"""Single-file Streamlit interface for a multi-PDF RAG application."""

"""Single-file Streamlit interface for a multi-PDF RAG application."""

from io import BytesIO
import os
import re

import streamlit as st


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVAL_K = 5
RETRIEVAL_FETCH_K = 15


st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="📚",
    layout="wide",
)


# Keep the vector store and indexing summary available across Streamlit reruns.
if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_chunk_count" not in st.session_state:
    st.session_state.indexed_chunk_count = 0
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "indexing_complete" not in st.session_state:
    st.session_state.indexing_complete = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []


st.title("📚 Multi-PDF Research Assistant")
st.write(
    "Upload several PDF documents together, then ask questions across the whole collection. "
    "Answers will include the source file and page for checking the evidence."
)

with st.expander("Setup and run instructions"):
    st.markdown("Install the required packages:")
    st.code(
        "python -m pip install streamlit langchain-community langchain-core "
        "langchain-openai langchain-text-splitters langsmith faiss-cpu pypdf pymupdf",
        language="powershell",
    )
    st.markdown("Set your API keys and tracing project in the same PowerShell window, then start the app:")
    st.code(
        '$env:OPENAI_API_KEY="your-api-key"\n'
        '$env:LANGSMITH_TRACING="true"\n'
        '$env:LANGSMITH_API_KEY="your-langsmith-api-key"\n'
        '$env:LANGSMITH_PROJECT="multi-pdf-rag"\n'
        "streamlit run rag_app.py",
        language="powershell",
    )
    st.caption(
        "On macOS or Linux, export the same variables before running `streamlit run rag_app.py`. "
        "Set `OPENAI_CHAT_MODEL` to override the default chat model. Traces can include questions, "
        "retrieved PDF text, prompts, and answers; enable tracing only for documents you're allowed "
        "to send to your LangSmith workspace."
    )

api_key = os.getenv("OPENAI_API_KEY")
langsmith_tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
    "true",
    "1",
    "yes",
}
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT", "default")
if api_key:
    st.success("OpenAI API key found in the OPENAI_API_KEY environment variable.")
else:
    st.warning(
        "Set the OPENAI_API_KEY environment variable before indexing documents or asking questions. "
        "The key is read from your environment and is not stored in this app."
    )
if langsmith_tracing_enabled and langsmith_api_key:
    st.success(f"LangSmith tracing is enabled for project: {langsmith_project}.")
    st.caption(
        "Questions, retrieved PDF passages, model prompts, and answers may appear in LangSmith traces."
    )
elif langsmith_tracing_enabled:
    st.warning("LangSmith tracing is on, but LANGSMITH_API_KEY is not set; traces may not be uploaded.")
else:
    st.info("LangSmith tracing is off. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable it.")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple PDFs in one upload. They will be combined into one searchable collection.",
)

if uploaded_files:
    st.caption(f"Selected {len(uploaded_files)} PDF file(s).")
    with st.expander("Selected files", expanded=True):
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")
else:
    st.info("Choose one or more PDF files to get started.")

build_disabled = not uploaded_files or not api_key
if st.button("Build searchable collection", type="primary", disabled=build_disabled):
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.last_evidence = []

    # Import the heavier document and vector-store dependencies only when needed.
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    pages = []
    notices = []
    seen_names = set()

    with st.spinner("Reading PDFs and preparing text chunks..."):
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            normalized_name = filename.casefold()
            if normalized_name in seen_names:
                notices.append(f"Skipped duplicate filename: {filename}")
                continue
            seen_names.add(normalized_name)

            file_pages = []
            pypdf_problem = None
            try:
                reader = PdfReader(BytesIO(uploaded_file.getvalue()))
                if reader.is_encrypted:
                    raise ValueError("the PDF is encrypted and could not be opened")

                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if not page_text:
                        continue
                    file_pages.append(
                        Document(
                            page_content=page_text,
                            metadata={"source": filename, "page": page_number},
                        )
                    )
            except Exception as exc:
                pypdf_problem = exc

            if pypdf_problem is None and file_pages:
                pages.extend(file_pages)
            else:
                # Discard any partial PyPDF extraction before trying the fallback.
                file_pages = []
                try:
                    import pymupdf

                    with pymupdf.open(
                        stream=uploaded_file.getvalue(), filetype="pdf"
                    ) as pdf_document:
                        for page_number, page in enumerate(pdf_document, start=1):
                            page_text = page.get_text("text").strip()
                            if not page_text:
                                continue
                            file_pages.append(
                                Document(
                                    page_content=page_text,
                                    metadata={"source": filename, "page": page_number},
                                )
                            )
                except Exception as fallback_error:
                    original_problem = (
                        str(pypdf_problem)
                        if pypdf_problem is not None
                        else "PyPDF found no extractable text"
                    )
                    notices.append(
                        f"Could not read {filename}. PyPDF: {original_problem}. "
                        f"PyMuPDF: {fallback_error}. The file was skipped."
                    )
                else:
                    if file_pages:
                        pages.extend(file_pages)
                        notices.append(
                            f"Used PyMuPDF to read {filename} after the primary PDF reader "
                            "could not extract it completely."
                        )
                    else:
                        notices.append(
                            f"No extractable text found in {filename}; it was skipped."
                        )

        if pages:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            chunks = splitter.split_documents(pages)
        else:
            chunks = []

        if chunks:
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key,
                )
                new_store = FAISS.from_documents(chunks, embeddings)
            except Exception as exc:
                st.session_state.store = None
                st.session_state.indexed_chunk_count = 0
                st.session_state.indexed_files = []
                st.session_state.indexing_complete = False
                st.error(f"Indexing failed while creating embeddings or the FAISS index: {exc}")
            else:
                st.session_state.store = new_store
                st.session_state.indexed_chunk_count = len(chunks)
                st.session_state.indexed_files = sorted(
                    {chunk.metadata["source"] for chunk in chunks}
                )
                st.session_state.indexing_complete = True
                st.success(
                    f"Indexed {len(chunks)} chunks from "
                    f"{len(st.session_state.indexed_files)} PDF(s)."
                )
        else:
            st.session_state.store = None
            st.session_state.indexed_chunk_count = 0
            st.session_state.indexed_files = []
            st.session_state.indexing_complete = False
            st.error("No usable text was found, so no searchable collection was created.")

    for notice in notices:
        st.warning(notice)

st.divider()

if st.session_state.store is None:
    st.caption("Your searchable collection will appear here after indexing completes.")
else:
    st.metric("Indexed chunks", st.session_state.indexed_chunk_count)
    st.caption("Indexed files: " + ", ".join(st.session_state.indexed_files))

st.subheader("Ask a question")
if st.session_state.store is None:
    st.text_input(
        "Question",
        placeholder="Index PDFs before asking a question.",
        disabled=True,
    )
else:
    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="For example: How many leave days do new joiners get?",
        )
        submitted = st.form_submit_button("Ask", type="primary", disabled=not api_key)

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            from langsmith import traceable
            from langchain_openai import ChatOpenAI

            with st.spinner("Searching the PDFs and preparing an answer..."):
                try:
                    @traceable(name="multi_pdf_rag_question", run_type="chain")
                    def run_rag_question(query):
                        matches = st.session_state.store.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": RETRIEVAL_K,
                                "fetch_k": RETRIEVAL_FETCH_K,
                                "lambda_mult": 0.7,
                            },
                        ).invoke(query)

                        evidence = []
                        context_blocks = []
                        for index, document in enumerate(matches, start=1):
                            source_id = f"S{index}"
                            filename = document.metadata.get("source", "Unknown file")
                            page_number = document.metadata.get("page", "unknown")
                            evidence.append(
                                {
                                    "id": source_id,
                                    "source": filename,
                                    "page": page_number,
                                    "content": document.page_content,
                                }
                            )
                            context_blocks.append(
                                f"[{source_id}] Source: {filename}, page {page_number}\n"
                                f"Passage:\n{document.page_content}"
                            )

                        if not evidence:
                            answer = (
                                "I couldn’t find relevant passages in the indexed PDFs, "
                                "so I can’t answer that from the available evidence."
                            )
                        else:
                            context = "\n\n---\n\n".join(context_blocks)
                            model = ChatOpenAI(
                                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
                                api_key=api_key,
                                temperature=0,
                            )
                            response = model.invoke(
                                [
                                    (
                                        "system",
                                        "Answer the user's question using only the supplied PDF passages. "
                                        "Do not add facts from memory or outside knowledge. If the passages "
                                        "do not support an answer, say that the available documents do not "
                                        "provide enough evidence. Compare relevant passages across documents "
                                        "before answering. Treat claims as conflicting only when they address "
                                        "the same subject and scope but cannot both be true; explain differences "
                                        "in date, jurisdiction, population, or conditions as qualifications "
                                        "rather than contradictions. When sources directly disagree, describe "
                                        "each position and cite evidence for both sides. Do not choose a winner "
                                        "unless the passages themselves resolve the disagreement. Cite every "
                                        "factual claim with the supporting source ID exactly as shown, such as "
                                        "[S1]. Do not cite passages that do not support the claim.",
                                    ),
                                    (
                                        "human",
                                        f"Question: {query}\n\nRetrieved passages:\n{context}",
                                    ),
                                ]
                            )
                            answer = response.content
                            if not isinstance(answer, str):
                                answer = str(answer)

                        return {"answer": answer, "evidence": evidence}

                    result = run_rag_question(question.strip())
                    st.session_state.last_question = question.strip()
                    st.session_state.last_answer = result["answer"]
                    st.session_state.last_evidence = result["evidence"]
                except Exception as exc:
                    st.error(f"Could not retrieve or generate an answer: {exc}")

    if st.session_state.last_answer:
        st.markdown(f"**Q:** {st.session_state.last_question}")
        evidence_by_id = {
            item["id"]: item for item in st.session_state.last_evidence
        }
        referenced_ids = re.findall(r"\[(S\d+)\]", st.session_state.last_answer)

        citation_number_by_location = {}
        citation_rows = []
        citation_number_by_id = {}
        for source_id in referenced_ids:
            item = evidence_by_id.get(source_id)
            if item is None:
                continue
            location = (item.get("source"), item.get("page"))
            if location not in citation_number_by_location:
                citation_number = len(citation_rows) + 1
                citation_number_by_location[location] = citation_number
                citation_rows.append((citation_number, *location))
            citation_number_by_id[source_id] = citation_number_by_location[location]

        def render_citation(match):
            source_id = match.group(1)
            citation_number = citation_number_by_id.get(source_id)
            return f"[{citation_number}]" if citation_number else ""

        display_answer = re.sub(
            r"\[(S\d+)\]", render_citation, st.session_state.last_answer
        )

        if citation_rows:
            st.markdown(display_answer)
            st.markdown("**Sources**")
            for citation_number, filename, page_number in citation_rows:
                if not filename or filename == "Unknown file":
                    filename = "Unknown file"
                page_label = (
                    f"page {page_number}"
                    if page_number not in (None, "", "unknown")
                    else "page unavailable"
                )
                st.text(f"[{citation_number}] {filename} — {page_label}")
        elif st.session_state.last_evidence:
            st.warning(
                "The response did not include a verifiable source citation, so it was "
                "withheld. Please try asking the question another way."
            )
        else:
            st.markdown(display_answer)
            st.caption("No relevant passages were retrieved from the indexed PDFs.")

        st.caption(
            "Page numbers refer to the PDF page positions used during extraction. "
            "Scanned pages, unusual layouts, or extraction errors can affect the text "
            "available for answering, even though the displayed page position is retained."
        )
