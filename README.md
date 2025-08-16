
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
### scores

for sores:

```bash
GT_TRANSCRIPT="<your Query>" python realtime_w_scores.py
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
## Experimental Results

Below are the collected runtime metrics from our real-time RAG + STT + TTS pipeline.

### Query 1: *"How to perform Umrah"*
- **Retrieval Cosine Distances (top 3):** 0.1081, 0.1328, 0.1417
- **LLM:** completion_tokens=84, prompt_tokens=832, latency=3.278s, tokens/sec=25.63
- **STT Latency:** 2.455s
- **RAG→LLM Latency:** 3.278s
- **TTS Latency:** 4.062s
- **End-to-End Latency (short):** 9.796s
- **STT Accuracy:** WER=0.500, CER=0.150
- **Wake Word:** last_score=0.791, count=1, false=0, false_rate=0.000
- **System Usage:** CPU=9.4%, RAM=808.3 

---

### Query 2: *"What shall I say after Tawaf?"*
- **Retrieval Cosine Distances (top 3):** 0.1757, 0.1794, 0.1812
- **LLM:** completion_tokens=71, prompt_tokens=832, latency=1.899s, tokens/sec=37.39
- **STT Latency:** 2.277s
- **RAG→LLM Latency:** 1.899s
- **TTS Latency:** 2.980s
- **End-to-End Latency (short):** 7.157s
- **STT Accuracy:** (GT transcript not provided → N/A)
- **Wake Word:** last_score=0.827, count=1, false=0, false_rate=0.000
- **System Usage:** CPU=9.2%, RAM=824.3 

---

### Observations
- **Wake word model** is stable with low false activation rate (0.000).
- **End-to-end latency** across both examples is between ~7–10s, which is acceptable for real-time dialogue at the booth.
- **WER/CER** are within a reasonable range for noisy conditions (0.50 / 0.15).
- **System usage** (CPU < 16%, RAM ~800 MB) confirms the solution runs within device resource constraints.

---


##  License
This project is for educational and research purposes.
