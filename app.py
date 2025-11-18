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

# ========= Config =========
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ======== Persistent Storage Paths ========
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

# Remove proxy vars to prevent OpenAI client bugs
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
    os.environ.pop(_k, None)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OPENAI_API_KEY in env variables.")
    raise ValueError("OPENAI_API_KEY missing.")

client = OpenAI(api_key=api_key)

# ========= FAISS + Store =========
index = None
resume_store = []  # list of dicts

def _new_index():
    return faiss.IndexFlatL2(EMBED_DIM)

def _save_index_and_store():
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(resume_store, f)
    logger.info(f"Saved FAISS index + store (count={len(resume_store)})")

def _load_index_and_store():
    global index, resume_store
    if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(STORE_PATH, "rb") as f:
                resume_store = pickle.load(f)
            logger.info(f"✅ Loaded FAISS index with {len(resume_store)} resumes.")
        except Exception as e:
            logger.warning(f"⚠️ Corrupted FAISS store. Rebuilding: {e}")
            index = _new_index()
            resume_store = []
            _save_index_and_store()
    else:
        index = _new_index()
        resume_store = []
        logger.warning("⚙️ No FAISS index found — starting empty.")

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
    texts = [r["text"] for r in resume_store]
    vecs = _embed_texts(texts)
    index = _new_index()
    index.add(vecs)
    _save_index_and_store()

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
        return json.loads(row[0]) if row else None
    except:
        return None

def set_cached_value(key, value):
    try:
        conn = get_db_connection()
        conn.execute("INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                     (key, json.dumps(value), time.time()))
        conn.commit()
        conn.close()
    except:
        pass

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

        file_storage.seek(0)
        return ""
    except:
        try:
            file_storage.seek(0)
        except:
            pass
        return ""

# ========= FIT + GAP PROMPT BUILDER =========
def build_explain_prompt(job_description, resume_name, resume_text, rank, total):
    safe_resume = resume_text or ""
    short_resume = safe_resume[:3000] + ("..." if len(safe_resume) > 3000 else "")

    return (
        "You are an expert technical recruiter performing a FIT + GAP analysis.\n\n"
        "Return your answer in this EXACT format (plain text only):\n\n"
        f"Fit Summary: <Write 3–4 full sentences explaining why the candidate is a strong match. "
        f"Explain their relevant skills, experience, domain knowledge, and why they ranked #{rank} "
        f"out of {total}.>\n\n"
        "Gap Analysis: <Write 3–4 full sentences stating EXACTLY what the candidate is missing. "
        "List missing skills, missing technologies, certifications, experience, domain gaps, or keywords.>\n\n"
        "Gap Severity: <Label the overall gap as Critical, Moderate, or Minor and explain why.>\n\n"
        "Recommended Next Steps: <1–2 sentences telling the candidate what they must learn "
        "or improve to fully match the job requirements.>\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME ({resume_name}) EXCERPT:\n{short_resume}"
    )

# ========= Reasoning Generator =========
def get_reasoning_for_resume(prompt, cache_key):
    cached = get_cached_value(cache_key)
    if cached:
        return cached
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=320,
            temperature=0.65
        )
        reasoning = resp.choices[0].message.content.strip()
        set_cached_value(cache_key, reasoning)
        return reasoning
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return "Explanation unavailable (generation error)."

# ========= Routes =========
@app.route("/")
def home():
    return "Backend OK."

@app.route("/health")
def health():
    return jsonify({"ok": True, "resume_count": len(resume_store)})

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(force=True) or {}
        job_description = data.get("job_description", "")

        if not job_description.strip():
            return jsonify({"error": "Job description missing"}), 400

        emb = client.embeddings.create(model=EMBED_MODEL, input=job_description)
        job_vec = np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

        if index.ntotal == 0:
            return jsonify({"matches": []}), 200

        k = min(5, index.ntotal)
        distances, idxs = index.search(job_vec, k=k)
        scores = 1 / (1 + distances)

        results = []
        for pos, idx in enumerate(idxs[0]):
            entry = resume_store[idx]
            name = entry["name"]
            text = entry["text"]
            score = float(scores[0][pos])

            cache_key = compute_hash(job_description + name + str(pos))
            prompt = build_explain_prompt(job_description, name, text, pos + 1, k)
            reasoning = get_reasoning_for_resume(prompt, cache_key)

            results.append({
                "rank": pos + 1,
                "name": name,
                "score": score,
                "reasoning": reasoning
            })

        return jsonify({"matches": results})

    except Exception as e:
        logger.exception("/search error")
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
            if not text.strip():
                continue

            name = f.filename
            vec = _embed_texts([text])
            index.add(vec)
            resume_store.append({"name": name, "text": text})
            added += 1

        _save_index_and_store()
        return jsonify({"ok": True, "added": added, "total": len(resume_store)})

    except Exception as e:
        logger.exception("/upload_resume error")
        return jsonify({"error": str(e)}), 500

@app.route("/list_resumes")
def list_resumes():
    out = [{
        "idx": i,
        "name": r["name"],
        "chars": len(r["text"])
    } for i, r in enumerate(resume_store)]
    return jsonify({"resumes": out})

@app.route("/delete_resume", methods=["POST"])
def delete_resume():
    try:
        data = request.get_json(force=True) or {}
        idx = data.get("idx")

        if idx is None or idx < 0 or idx >= len(resume_store):
            return jsonify({"error": "Invalid idx"}), 400

        removed = resume_store.pop(idx)
        _rebuild_full_index()

        return jsonify({"ok": True, "removed": removed["name"], "remaining": len(resume_store)})

    except:
        return jsonify({"error": "Delete failed"}), 500

# ========= Startup =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Backend running on port {port}")
    app.run(host="0.0.0.0", port=port)
