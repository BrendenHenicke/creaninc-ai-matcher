import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx
import io
import sqlite3, hashlib, json, time
import sys
import traceback

# ========= Config =========
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ======== Persistent Storage Paths ========
# Use /data if available (Render persistent disk mount)
PERSIST_PATH = "/data" if os.path.exists("/data") else "."
INDEX_PATH = os.path.join(PERSIST_PATH, "resume_index.faiss")
STORE_PATH = os.path.join(PERSIST_PATH, "resume_store.pkl")
CACHE_DB = os.path.join(PERSIST_PATH, "reasoning_cache.db")

print("------------------------------------------------------")
print(f"📦 Persistent path configured: {PERSIST_PATH}")
print(f"📄 Index path: {INDEX_PATH}")
print(f"📄 Store path: {STORE_PATH}")
print("------------------------------------------------------")

# ========= Logging =========
if not os.path.exists("logs"):
    os.makedirs("logs")
handler = RotatingFileHandler("logs/backend.log", maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ========= Flask =========
app = Flask(__name__)
CORS(app)

# ========= OpenAI =========
load_dotenv()

# Nuke proxy env vars to avoid "proxies" kwarg bug
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
    os.environ.pop(_k, None)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OPENAI_API_KEY in environment variables.")
    raise ValueError("OPENAI_API_KEY is missing. Set it in environment variables.")

client = OpenAI(api_key=api_key)

# ========= FAISS + Store =========
index = None
resume_store = []  # list[dict]: {"name": str, "text": str}

def _new_index():
    return faiss.IndexFlatL2(EMBED_DIM)

def _save_index_and_store():
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(resume_store, f)
    logger.info(f"Saved FAISS index and resume store (count={len(resume_store)})")

def _load_index_and_store():
    global index, resume_store
    if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(STORE_PATH, "rb") as f:
                resume_store = pickle.load(f)
            logger.info(f"✅ Loaded FAISS index with {len(resume_store)} resumes from disk.")
            print(f"✅ Loaded FAISS index with {len(resume_store)} resumes from {PERSIST_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ Corrupted FAISS store detected, rebuilding from scratch: {e}")
            index = _new_index()
            resume_store = []
            _save_index_and_store()
    else:
        logger.warning("⚙️ No FAISS index found — starting empty.")
        print("⚙️ No FAISS index found — creating a new one.")
        index = _new_index()
        resume_store = []

def _embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.array(r.embedding, dtype="float32") for r in resp.data]
    return np.vstack(vecs)

def _rebuild_full_index():
    global index
    if not resume_store:
        index = _new_index()
        _save_index_and_store()
        return
    texts = [r.get("text", "") for r in resume_store]
    vecs = _embed_texts(texts)
    index = _new_index()
    index.add(vecs)
    _save_index_and_store()
    logger.info("Rebuilt FAISS index from resume_store (size=%d).", len(resume_store))

_load_index_and_store()

# ========= SQLite Cache =========
def get_db_connection():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            created_at REAL
        )
    """)
    return conn

def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_value(key):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row:
            logger.info(f"Cache hit for key: {key[:12]}...")
        return json.loads(row[0]) if row else None
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None

def set_cached_value(key, value):
    try:
        conn = get_db_connection()
        conn.execute("INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                     (key, json.dumps(value), time.time()))
        conn.commit()
        conn.close()
        logger.info(f"Cache write success for key: {key[:12]}...")
    except Exception as e:
        logger.error(f"Cache write error: {e}")

# ========= File Extraction =========
def extract_text_from_file_storage(file_storage):
    filename = (file_storage.filename or "").lower()
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_storage.read()))
            file_storage.seek(0)
            return "".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".docx"):
            tmp = io.BytesIO(file_storage.read())
            file_storage.seek(0)
            doc = docx.Document(tmp)
            return "\n".join(p.text for p in doc.paragraphs)
        elif filename.endswith(".txt"):
            content = file_storage.read().decode("utf-8", errors="ignore")
            file_storage.seek(0)
            return content
        else:
            file_storage.seek(0)
            return ""
    except Exception as e:
        logger.warning(f"File read error: {e}")
        try:
            file_storage.seek(0)
        except:
            pass
        return ""

# ========= Prompt Builder =========
def build_explain_prompt(job_description, resume_name, resume_text, rank, total):
    short_resume = (resume_text[:3000] + "...") if resume_text and len(resume_text) > 3000 else (resume_text or "")
    return (
        f"You are an experienced technical recruiter and hiring manager. "
        f"In 3–5 conversational sentences, explain why this candidate is a good fit "
        f"for the job and why they ranked #{rank} out of {total}. "
        f"Be concise, specific to the candidate's highlights, and compare briefly to others.\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME ({resume_name}) SUMMARY / EXCERPT:\n{short_resume}\n\n"
        f"Return plain text (3–5 sentences)."
    )

def get_reasoning_for_resume(prompt, cache_key):
    cached = get_cached_value(cache_key)
    if cached:
        return cached
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        reasoning = resp.choices[0].message.content.strip()
        set_cached_value(cache_key, reasoning)
        return reasoning
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return "Explanation unavailable (error generating reasoning)."

# ========= Routes =========
@app.route("/")
def home():
    return "Flask backend running."

@app.route("/health")
def health():
    return jsonify({"ok": True, "resume_count": len(resume_store)})

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(force=True, silent=True) or {}
        job_description = data.get("job_description", "")
        if not job_description.strip():
            return jsonify({"error": "Job description missing"}), 400

        emb = client.embeddings.create(model=EMBED_MODEL, input=job_description)
        job_vec = np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

        if index.ntotal == 0:
            return jsonify({"matches": [], "ranking_summary": "Resume database is empty."}), 200

        k = min(5, index.ntotal)
        distances, indices = index.search(job_vec, k=k)
        scores = 1 / (1 + distances)

        results = []
        for i, idx in enumerate(indices[0][:k]):
            if idx < 0 or idx >= len(resume_store):
                continue
            entry = resume_store[idx]
            name = entry.get("name") or f"Resume {idx}"
            resume_text = entry.get("text") or ""
            score = float(scores[0][i])
            reasoning_key = compute_hash(job_description + name + str(i))
            prompt = build_explain_prompt(job_description, name, resume_text, i + 1, k)
            reasoning = get_reasoning_for_resume(prompt, reasoning_key)
            results.append({
                "rank": i + 1,
                "name": name,
                "score": score,
                "reasoning": reasoning
            })

        return jsonify({"matches": results})
    except Exception as e:
        logger.exception(f"/search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files part"}), 400

        files = request.files.getlist("files")
        added = 0
        for f in files:
            text = extract_text_from_file_storage(f)
            name = f.filename or f"Resume_{int(time.time())}"
            if not text.strip():
                continue
            vec = _embed_texts([text])
            index.add(vec)
            resume_store.append({"name": name, "text": text})
            added += 1

        _save_index_and_store()
        logger.info(f"Uploaded {added} resumes (total={len(resume_store)})")
        return jsonify({"ok": True, "added": added, "total": len(resume_store)})
    except Exception as e:
        logger.exception(f"/upload_resume error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/list_resumes", methods=["GET"])
def list_resumes():
    out = [{"idx": i, "name": r.get("name", f"Resume {i}"), "chars": len(r.get("text", ""))} for i, r in enumerate(resume_store)]
    return jsonify({"resumes": out, "count": len(out)})

@app.route("/delete_resume", methods=["POST"])
def delete_resume():
    try:
        data = request.get_json(force=True, silent=True) or {}
        idx = data.get("idx")
        if idx is None or not (0 <= idx < len(resume_store)):
            return jsonify({"error": "Invalid idx"}), 400
        removed = resume_store.pop(idx)
        _rebuild_full_index()
        return jsonify({"ok": True, "removed": removed.get("name"), "remaining": len(resume_store)})
    except Exception as e:
        logger.exception(f"/delete_resume error: {e}")
        return jsonify({"error": str(e)}), 500

# ========= Main =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Backend running on port {port}")
    app.run(host="0.0.0.0", port=port)
