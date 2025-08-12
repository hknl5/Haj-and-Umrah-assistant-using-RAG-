
# Voice RAG System for Raspberry Pi

This project implements a **real-time Retrieval-Augmented Generation (RAG)** pipeline with **Speech-to-Text (STT)**, **LLM-based retrieval and answering**, and **Text-to-Speech (TTS)** — optimized to run on a **Raspberry Pi**.  
It allows you to speak a query to the Raspberry Pi, which will transcribe your voice, retrieve relevant knowledge chunks, generate an intelligent answer, and speak it back to you.

---

## 📌 Project Flow

```
 Voice Input
    ↓
 Speech-to-Text (STT)
    ↓
 Retrieval from Knowledge Base (RAG)
    ↓
 Local or API-based LLM Generation
    ↓
 Text-to-Speech (TTS)
```

---

##  Features

- **Real-time voice interaction** on Raspberry Pi.
- **Speech-to-Text (STT)** transcription of spoken queries.
- **Retrieval-Augmented Generation (RAG)** for factually accurate answers.
- **Local LLM or API-based** text generation.
- **Text-to-Speech (TTS)** to respond in natural voice.
- Designed for **low-resource devices** like Raspberry Pi 5.

---

## ⚙️ Installation & Running

### 1️⃣ (Optional) Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> For Raspberry Pi audio dependencies:
```bash
sudo apt-get install -y libportaudio2 libsndfile1
sudo apt-get install -y build-essential cmake
```

### 3️⃣ Run the Real-Time RAG Script
```bash
python realtime_for_raspberry.py
```

---

## 📂 Files Overview

- **realtime_for_raspberry.py** — Main script for real-time voice RAG on Raspberry Pi.
- **STT_RAG_TTS.ipynb** — Notebook for testing STT → RAG → TTS pipeline.
- **first_chunkin-embeddin-e5.ipynb** — Notebook for chunking & embedding documents.
- **second_vector-chromdb.ipynb** — Vector database setup and indexing.
- **third_retrieval-reranking.ipynb** — Retrieval and re-ranking pipeline.
- **fourth-1-generation-local-llm.ipynb** — Local LLM generation notebook.
- **fourth-2-generation-w-voice-API.ipynb** — LLM + TTS API generation notebook.

---

##  License
This project is for educational and research purposes.
