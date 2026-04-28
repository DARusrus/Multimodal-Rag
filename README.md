---
title: Multimodal RAG Assistant
emoji: 🦁
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🦁 Multimodal RAG Assistant

A portfolio-grade Retrieval-Augmented Generation assistant that combines:
- 📄 **Document QA** — PDF, CSV, arXiv ingestion with Chroma vector search
- 🧠 **Mistral-7B LLM** — grounded answers with source citations
- 💬 **Conversation memory** — multi-turn contextual follow-ups
- 🖼️ **Image understanding** — BLIP captioning + visual QA
- 🎙️ **Voice interface** — Faster-Whisper STT + gTTS TTS

## Tech Stack
`Mistral-7B` · `BLIP` · `Faster-Whisper` · `LangChain` · `ChromaDB` · `Streamlit`

## How to Use
1. Upload a PDF/CSV or enter an arXiv ID in the sidebar
2. Click **Build Index**
3. Chat with your documents, analyze images, or use the voice interface
