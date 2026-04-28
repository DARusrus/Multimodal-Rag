
import os, time, re, tempfile, shutil, warnings
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🦁",
    layout="wide",
)

# ── Imports (heavy — cached after first load) ───────────────
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from faster_whisper import WhisperModel
from gtts import gTTS
from PIL import Image
import numpy as np
from collections import deque
from typing import List, Dict
import pathlib, requests

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 120
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
CHROMA_DIR    = "/tmp/chroma_db_app"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════
# CACHED MODEL LOADERS — load once, reuse across interactions
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Loading LLM (this takes ~2 min)...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    llm = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, do_sample=False,
        temperature=1.0, repetition_penalty=1.05,
        return_full_text=False,
    )
    return llm

@st.cache_resource(show_spinner="⏳ Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="⏳ Loading BLIP captioning model...")
def load_blip_caption():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner="⏳ Loading BLIP VQA model...")
def load_blip_vqa():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model     = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    return processor, model

@st.cache_resource(show_spinner="⏳ Loading Whisper STT...")
def load_whisper():
    compute = "float16" if DEVICE == "cuda" else "int8"
    return WhisperModel("base", device=DEVICE, compute_type=compute)

# ══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def load_and_index(uploaded_files, arxiv_id: str, k: int, chunk_size: int, chunk_overlap: int):
    """Load documents, chunk, embed and store in Chroma. Returns vectorstore."""
    raw_docs = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(tmp_path).load()
            elif suffix == ".csv":
                docs = CSVLoader(tmp_path).load()
            else:
                docs = []
            raw_docs.extend([
                Document(
                    page_content=clean_text(d.page_content),
                    metadata={"source": uf.name}
                ) for d in docs if len(d.page_content.strip()) > 30
            ])
        finally:
            os.unlink(tmp_path)

    if arxiv_id.strip():
        try:
            docs = ArxivLoader(query=arxiv_id.strip(), load_max_docs=1).load()
            raw_docs.extend([
                Document(
                    page_content=clean_text(d.page_content),
                    metadata={"source": f"arxiv:{arxiv_id.strip()}"}
                ) for d in docs
            ])
        except Exception as e:
            st.warning(f"arXiv load failed: {e}")

    if not raw_docs:
        return None, 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for i, doc in enumerate(raw_docs):
        for j, text in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_index": j, "doc_index": i}
            ))

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings  = load_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    return vectorstore, len(raw_docs), len(chunks)

def build_prompt(query: str, docs: List) -> str:
    context = "\n\n".join(
        f"[Source {i+1} | {d.metadata.get('source', '?')}]\n{d.page_content.strip()}"
        for i, d in enumerate(docs)
    )
    return (
        "[INST] You are a helpful assistant. Answer using ONLY the context. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query} [/INST]"
    )

def build_prompt_with_memory(query: str, docs: List, history: list) -> str:
    context = "\n\n".join(
        f"[Source {i+1} | {d.metadata.get('source', '?')}]\n{d.page_content.strip()}"
        for i, d in enumerate(docs)
    )
    history_str = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in history[-6:]
    )
    history_section = f"\nCONVERSATION HISTORY:\n{history_str}\n" if history_str else ""
    return (
        "[INST] You are a helpful assistant. Answer using ONLY the context. "
        "Use the conversation history for follow-up references.\n\n"
        f"CONTEXT:\n{context}{history_section}\n"
        f"QUESTION: {query} [/INST]"
    )

def rag_answer(query: str, vectorstore, k: int, history: list) -> Dict:
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs    = [r[0] for r in results]
    scores  = [r[1] for r in results]
    prompt  = build_prompt_with_memory(query, docs, history)
    llm     = load_llm()
    t0      = time.time()
    answer  = llm(prompt)[0]["generated_text"].strip()
    elapsed = time.time() - t0
    return {"answer": answer, "sources": docs, "scores": scores, "time": elapsed}

def generate_caption(image: Image.Image) -> str:
    processor, model = load_blip_caption()
    dtype   = torch.float16 if DEVICE == "cuda" else torch.float32
    inputs  = processor(image.convert("RGB"), return_tensors="pt").to(DEVICE, dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

def answer_vqa(image: Image.Image, question: str) -> str:
    processor, model = load_blip_vqa()
    dtype   = torch.float16 if DEVICE == "cuda" else torch.float32
    inputs  = processor(image.convert("RGB"), question, return_tensors="pt").to(DEVICE, dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)

def transcribe_audio(audio_bytes: bytes, suffix: str) -> str:
    whisper = load_whisper()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = whisper.transcribe(tmp_path, beam_size=5)
        return " ".join(s.text.strip() for s in segments) or "[no speech detected]"
    finally:
        os.unlink(tmp_path)

def synthesize_speech(text: str) -> bytes:
    tts = gTTS(text=text, lang="en", slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data

# ══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════
if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_count"    not in st.session_state: st.session_state.doc_count    = 0
if "chunk_count"  not in st.session_state: st.session_state.chunk_count  = 0

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🦁 RAG Assistant")
    st.markdown("---")

    st.subheader("📂 Document Sources")
    uploaded_files = st.file_uploader(
        "Upload PDF or CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )
    arxiv_id = st.text_input("arXiv Paper ID (e.g. 2201.11095)", value="")

    st.subheader("⚙️ Settings")
    k             = st.slider("Retrieval top-k",    1, 10, 3)
    chunk_size    = st.slider("Chunk size",        300, 1500, 900, step=50)
    chunk_overlap = st.slider("Chunk overlap",      0,  300, 120, step=10)
    speak_answers = st.toggle("🔊 Speak answers (TTS)", value=False)

    if st.button("🔍 Build Index", type="primary", use_container_width=True):
        if not uploaded_files and not arxiv_id.strip():
            st.error("Please upload a file or enter an arXiv ID.")
        else:
            with st.spinner("Indexing documents ..."):
                vs, n_docs, n_chunks = load_and_index(
                    uploaded_files, arxiv_id, k, chunk_size, chunk_overlap
                )
            if vs:
                st.session_state.vectorstore  = vs
                st.session_state.doc_count    = n_docs
                st.session_state.chunk_count  = n_chunks
                st.session_state.chat_history = []
                st.success(f"✅ Indexed {n_docs} docs → {n_chunks} chunks")
            else:
                st.error("No documents loaded. Check your files or arXiv ID.")

    st.markdown("---")
    st.subheader("📊 Index Status")
    if st.session_state.vectorstore:
        st.metric("Documents", st.session_state.doc_count)
        st.metric("Chunks",    st.session_state.chunk_count)
        st.metric("Device",    DEVICE.upper())
    else:
        st.info("No index built yet.")

# ══════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Image", "🎙️ Audio"])

# ─── TAB 1: CHAT ──────────────────────────────────────────
with tab1:
    st.header("💬 Document Chat")

    if not st.session_state.vectorstore:
        st.info("👈 Build an index first using the sidebar.")
    else:
        # Display chat history
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["user"])
            with st.chat_message("assistant"):
                st.write(turn["assistant"])
                with st.expander("📚 Sources used"):
                    for i, (src, score) in enumerate(zip(turn["sources"], turn["scores"])):
                        st.markdown(
                            f"**[{i+1}]** `{src.metadata.get('source', '?')}` "
                            f"— score: `{score:.4f}`\n\n"
                            f"> {src.page_content[:200]}..."
                        )

        # Chat input
        user_input = st.chat_input("Ask a question about your documents ...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking ..."):
                    result = rag_answer(
                        user_input,
                        st.session_state.vectorstore,
                        k,
                        st.session_state.chat_history
                    )
                st.write(result["answer"])
                st.caption(f"⏱️ Generated in {result['time']:.2f}s")
                with st.expander("📚 Sources used"):
                    for i, (src, score) in enumerate(zip(result["sources"], result["scores"])):
                        st.markdown(
                            f"**[{i+1}]** `{src.metadata.get('source', '?')}` "
                            f"— score: `{score:.4f}`\n\n"
                            f"> {src.page_content[:200]}..."
                        )

                if speak_answers:
                    with st.spinner("🔊 Synthesizing speech ..."):
                        audio_bytes = synthesize_speech(result["answer"])
                    st.audio(audio_bytes, format="audio/mp3")

            st.session_state.chat_history.append({
                "user"     : user_input,
                "assistant": result["answer"],
                "sources"  : result["sources"],
                "scores"   : result["scores"],
            })

        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ─── TAB 2: IMAGE ─────────────────────────────────────────
with tab2:
    st.header("🖼️ Image Understanding")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_column_width=True)

        with col2:
            if st.button("📝 Generate Caption", use_container_width=True):
                with st.spinner("Generating caption ..."):
                    caption = generate_caption(image)
                st.success(f"**Caption:** {caption}")

            st.markdown("---")
            vqa_question = st.text_input("Ask a question about the image")
            if st.button("🔍 Answer Visual Question", use_container_width=True):
                if vqa_question.strip():
                    with st.spinner("Analyzing image ..."):
                        vqa_answer = answer_vqa(image, vqa_question)
                    st.info(f"**Answer:** {vqa_answer}")
                else:
                    st.warning("Please enter a question.")

# ─── TAB 3: AUDIO ─────────────────────────────────────────
with tab3:
    st.header("🎙️ Voice Interface")
    uploaded_audio = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_audio:
        st.audio(uploaded_audio)
        suffix = Path(uploaded_audio.name).suffix.lower()

        if st.button("🎙️ Transcribe Audio", use_container_width=True):
            with st.spinner("Transcribing ..."):
                transcript = transcribe_audio(uploaded_audio.read(), suffix)
            st.session_state["last_transcript"] = transcript
            st.success(f"**Transcript:** {transcript}")

        if "last_transcript" in st.session_state:
            st.markdown("---")
            st.markdown(f"**Last transcript:** {st.session_state['last_transcript']}")

            if st.session_state.vectorstore:
                if st.button("🔁 Run RAG on transcript", use_container_width=True):
                    with st.spinner("Generating answer ..."):
                        result = rag_answer(
                            st.session_state["last_transcript"][:300],
                            st.session_state.vectorstore,
                            k,
                            st.session_state.chat_history
                        )
                    st.write(f"**Answer:** {result['answer']}")
                    with st.spinner("🔊 Synthesizing spoken response ..."):
                        audio_bytes = synthesize_speech(result["answer"])
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        "⬇️ Download spoken response",
                        data=audio_bytes,
                        file_name="rag_response.mp3",
                        mime="audio/mp3"
                    )
            else:
                st.info("Build an index first to run RAG on the transcript.")
