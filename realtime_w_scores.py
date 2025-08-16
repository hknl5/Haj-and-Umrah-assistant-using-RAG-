import os
import sys
import io
import time
import json
import shutil
import keyboard
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
import chromadb
import openai
import select
import termios
import tty
from base64 import b64encode, b64decode
from tempfile import NamedTemporaryFile
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from IPython.display import HTML, Javascript, display

# --- metrics helpers (drop-in) ---
from tempfile import NamedTemporaryFile
import tty
import time

MET = {
    "wake": {"count": 0, "false": 0, "last_score": None, "last_ts": None},
}

def _now():
    return time.perf_counter()

def _print_sys_usage(tag=""):
    try:
        import psutil, os
        p = psutil.Process(os.getpid())
        rss_mb = p.memory_info().rss / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=None)
        print(f"[SYS] {tag} CPU%={cpu_pct:.1f}  RSS={rss_mb:.1f} MB")
    except Exception:

        pass
def _reset_stdin():
    try:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _levenshtein(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[lb]

def wer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = _levenshtein(ref_words, hyp_words)
    return d / len(ref_words)

def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    d = _levenshtein(ref, hyp)
    return d / len(ref)



api_key = ""
client = OpenAI(api_key=api_key)

# Config for embedding model and Chroma
MODEL_NAME = 'intfloat/e5-base-v2'
#CHROMA_PATH = "/Users/hanouf/Downloads/akid_final/hajj_e5_chroma_backup"
COLLECTION_NAME = 'hajj_e5'
PASSAGE_PREFIX = 'passage: '
QUERY_PREFIX = 'query: '


# Maximum tokens for generation and context
MAX_TOKENS = 256

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load E5 model and tokenizer for query encoding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def embed_query(text: str):
    """Encode a query string into an embedding vector using E5 and normalise it."""
    input_text = QUERY_PREFIX + text
    encoded = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        token_embeds = out.last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1)
        sum_embeds = (token_embeds * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1)
        embed = (sum_embeds / sum_mask).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(embed)
    if norm > 0:
        embed = embed / norm
    return embed

# Copy the folder into a writable location (if it came from a read-only dataset)
#shutil.copytree(CHROMA_PATH, '/Users/hanouf/Downloads/akid_final/hajj_e5_chroma')

# Then point Chroma at the copy
CHROMA_PATH = '/Users/hanouf/Downloads/akid_final/hajj_e5_chroma'
clientc = chromadb.PersistentClient(path=CHROMA_PATH)
collection = clientc.get_or_create_collection(name='hajj_e5', metadata={'hnsw:space': 'cosine'})

# Helper search function with lexical re-ranking as fallback
def search(query_str: str, top_k: int = 10, re_rank: bool = True):
    query_embed = embed_query(query_str)
    result = collection.query(query_embeddings=[query_embed.tolist()], n_results=top_k)
    ids = result['ids'][0]
    dists = result['distances'][0]
    docs = result['documents'][0]
    metas = result['metadatas'][0]
    hits = []
    for id_, dist, doc, meta in zip(ids, dists, docs, metas):
        hits.append({'id': id_, 'distance': float(dist), 'text': doc, 'metadata': meta})
    if re_rank:
        query_tokens = set(query_str.lower().split())
        for h in hits:
            text_tokens = set(h['text'].lower().split())
            h['lexical_score'] = len(query_tokens & text_tokens)
        hits.sort(key=lambda x: x['lexical_score'], reverse=True)
    return hits



def build_prompt(question: str, sources: list):
    """Construct a prompt for the LLM using the question and retrieved sources."""
    prompt_lines = []
    prompt_lines.append("You are an assistant answering questions about Hajj and Umrah.")
    prompt_lines.append("Answer concisely in plain English so that the response can be read aloud.")
    prompt_lines.append("Keep the answer to no more than 3–4 sentences.")
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("Sources:")
    for i, src in enumerate(sources, 1):
        text = src['text'].replace("", " ").strip()
        if len(text) > 300:
            text = text[:297] + '...'
        prompt_lines.append(f"[{i}] {text}")
    prompt_lines.append("Answer:")
    return "".join(prompt_lines)


def generate_answer(question: str, top_k: int = 5):
    t_llm_start = _now()
    hits = search(question, top_k=top_k, re_rank=True)


    if hits:
        top3 = hits[:3]
        print("[RET] top cosine distances:",
              ", ".join(f"{h['distance']:.4f}" for h in top3))

    prompt = build_prompt(question, hits)
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        stop=["Sources:", "Question:"]
    )
    t_llm_end = _now()

   
    try:
        usage = getattr(result, "usage", None)
        ctok = getattr(usage, "completion_tokens", None) if usage else None
        ptok = getattr(usage, "prompt_tokens", None) if usage else None
        if ctok is not None:
            dur = t_llm_end - t_llm_start
            tps = (ctok / dur) if dur > 0 else float("inf")
            print(f"[LLM] completion_tokens={ctok} prompt_tokens={ptok} "
                  f"latency={dur:.3f}s tokens_per_sec={tps:.2f}")
        else:
            print(f"[LLM] latency={t_llm_end - t_llm_start:.3f}s (usage unavailable)")
    except Exception:
        print(f"[LLM] latency={t_llm_end - t_llm_start:.3f}s")


    answer = result.choices[0].message.content.strip()
    return answer, hits


USE_NEW_SDK = True
SAMPLE_RATE = 16000
DURATION = 5

def record_wav_to_buffer(seconds=DURATION, sr=SAMPLE_RATE):
    print("🎙️ Speak now...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    buf.name = "mic.wav"
    return buf

def transcribe(wav_buf) -> str:
    if USE_NEW_SDK:
        # New SDK (>=1.x): gpt-4o-transcribe
        return client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=wav_buf,
            response_format="text"
        )
    else:
        # Legacy SDK: whisper-1
        wav_buf.seek(0)
        result = openai_legacy.Audio.transcribe("whisper-1", wav_buf)
        return result["text"]

def tts_to_wav_file(text: str, voice="ash") -> str:

    if not text:
        raise ValueError("TTS: empty text")

    tmp = NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()

    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="wav", 
        )
        audio_bytes = resp.read() if hasattr(resp, "read") else resp.content
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise


def play_wav(path):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# -------- ESC detection on macOS/Linux without admin ----------
class RawKey:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

def esc_pressed(timeout_ms=0):
    """Non-blocking check for ESC; works in VSCode terminal on macOS."""
    r, _, _ = select.select([sys.stdin], [], [], timeout_ms/1000.0)
    if r:
        ch = sys.stdin.read(1)
        return ch == "\x1b"  # ESC
    return False
# --------------------------------------------------------------


import sounddevice as sd #safa
from openwakeword.model import Model #safa

# === SAFA wake word thing ===

FRAME_DURATION = 0.08  # 80 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

WW_MODEL_PATH = "/Users/hanouf/Downloads/akid_final/data/Safa.onnx"  # PATH 
wake_model = Model(inference_framework="onnx", wakeword_models=[WW_MODEL_PATH])

def wait_for_wakeword(threshold: float = 0.5) -> bool:

    detected = {"ok": False}

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        scores = wake_model.predict(indata[:, 0])
        for name, score in scores.items():
            if score > threshold:
                # === اضيفي هذه الثلاثة اسطر ===
                MET["wake"]["count"] += 1
                MET["wake"]["last_score"] = float(score)
                MET["wake"]["last_ts"] = _now()
                # ===============================
                print(f"[WakeWord] Detected: {name} ({score:.2f})")
                detected["ok"] = True
                raise sd.CallbackStop()


    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype='int16',
            callback=audio_callback
        ):
            while not detected["ok"]:
                sd.sleep(50)
    except sd.CallbackStop:
        pass
    except KeyboardInterrupt:
        print("\n[WakeWord] Stopped by user.")
        return False

    return detected["ok"]

def main_once():
    print(" Listening for wake word...")
    ok = wait_for_wakeword(threshold=0.5)
    if not ok:
        print("No wakeword detected. Exiting...")
        return

    # Record -> STT
    wav_buf = record_wav_to_buffer()
    t_stt_start = _now()
    txt = transcribe(wav_buf)
    t_stt_end = _now()
    print("User:", txt, "\n")

    # Retrieval + LLM
    t_mid_start = _now()
    answer, sources = generate_answer(txt, top_k=5)
    t_mid_end = _now()

    print("Answer:", answer, "\n")
    print("Sources used:")
    for i, src in enumerate(sources, 1):
        dist = src.get("distance", None)
        dtxt = f"  (dist={dist:.4f})" if dist is not None else ""
        preview = src["text"][:150].replace("\n", " ")
        print(f"[{i}]{dtxt} {preview}")

    # TTS -> playback
    t_tts_start = _now()
    audio_path = tts_to_wav_file(answer, voice="alloy")
    t_tts_end = _now()
    play_wav(audio_path)

    # Latency prints
    stt_lat = t_stt_end - t_stt_start
    post_stt_to_pre_tts = t_mid_end - t_mid_start
    tts_proc = t_tts_end - t_tts_start
    e2e_short = t_tts_end - t_stt_start
    print(f"[LAT] STT={stt_lat:.3f}s  postSTT->preTTS={post_stt_to_pre_tts:.3f}s  "
          f"TTS_proc={tts_proc:.3f}s  E2E(short)={e2e_short:.3f}s")

    # Wake stats
    wf = MET["wake"]["false"]
    wc = MET["wake"]["count"]
    rate = (wf / wc) if wc else 0.0
    print(f"[WakeWord] last_score={MET['wake']['last_score']}  "
          f"count={wc}  false={wf}  false_rate={rate:.3f}")

    # System usage
    _print_sys_usage(tag="end-of-run")
    print(" Done. Exiting program.")


if __name__ == "__main__":
    main_once()


