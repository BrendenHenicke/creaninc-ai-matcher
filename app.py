import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
import pickle
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx
import io
import sqlite3, hashlib, json, time
import sys
import traceback

# === Logging configuration ===
if not os.path.exists("logs"):
    os.makedirs("logs")

handler = RotatingFileHandler("logs/backend.log", maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# === Initialize Flask app ===
app = Flask(__name__)

# === Load environment variables ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Verify key presence ===
if not openai.api_key:
    logger.error("❌ Missing OPENAI_API_KEY in environment variables.")
    raise ValueError("OPENAI_API_KEY is missing. Please set it in Render Environment Variables.")

# === Load FAISS index and resume data ===
index = None
resume_store = []
try:
    if os.path.exists("resume_index.faiss") and os.path.exists("resume_store.pkl"):
        index = faiss.read_index("resume_index.faiss")
        with open("resume_store.pkl", "rb") as f:
            resume_store = pickle.load(f)
        logger.info("✅ FAISS index and resume data loaded successfully.")
    else:
        logger.warning("⚠️ No FAISS index or resume_store.pkl found — using empty store.")
        d = 1536  # embedding dimension
        index = faiss.IndexFlatL2(d)
        resume_store = []
except Exception as e:
    logger.error(f"Error loading FAISS or resume data: {e}")
    raise e

# === SQLite cache for reasoning & summaries ===
CACHE_DB = "reasoning_cache.db"

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

# === File extraction helper ===
def extract_text_from_file_storage(file_storage):
    filename = file_storage.filename.lower()
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_storage.read()))
            file_storage.seek(0)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return text
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

# === Prompt Builders ===
def build_explain_prompt(job_description, resume_name, resume_text, rank, total):
    short_resume = (resume_text[:3000] + "...") if resume_text and len(resume_text) > 3000 else (resume_text or "")
    prompt = (
        f"You are an experienced technical recruiter and hiring manager. "
        f"In 3–5 conversational sentences, explain why this candidate is a good fit "
        f"for the job and why they ranked #{rank} out of {total}. "
        f"Be concise, specific to the candidate's highlights, and compare briefly to others.\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME ({resume_name}) SUMMARY / EXCERPT:\n{short_resume}\n\n"
        f"Return plain text (3–5 sentences)."
    )
    return prompt

# === OpenAI call helper ===
def get_reasoning_for_resume(prompt, cache_key):
    cached = get_cached_value(cache_key)
    if cached:
        return cached
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        reasoning = resp.choices[0].message["content"].strip()
        set_cached_value(cache_key, reasoning)
        return reasoning
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return "Explanation unavailable (error generating reasoning)."

# === Flask routes ===
@app.route("/")
def home():
    logger.info("Health check at root route.")
    return "✅ Flask backend running successfully!"

@app.route("/search", methods=["POST"])
def search_resumes():
    try:
        job_description = None

        if "file" in request.files:
            uploaded_file = request.files["file"]
            job_description = extract_text_from_file_storage(uploaded_file)
            logger.info(f"Received file upload: {uploaded_file.filename}")
        else:
            data = request.get_json(force=True, silent=True) or {}
            job_description = data.get("job_description", "")
            logger.info("Received JSON job description request.")

        if not job_description or not job_description.strip():
            logger.warning("Empty job description submitted.")
            return jsonify({"error": "Job description missing"}), 400

        job_hash = compute_hash(job_description)

        # === Step 1: Create job embedding ===
        embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=job_description
        )
        job_vector = np.array(embedding["data"][0]["embedding"]).astype("float32").reshape(1, -1)

        # === Step 2: Search FAISS ===
        if index.ntotal == 0:
            logger.warning("⚠️ FAISS index is empty — no resumes to match.")
            return jsonify({"matches": [], "ranking_summary": "Resume database is empty."}), 200

        k = min(5, index.ntotal)
        distances, indices = index.search(job_vector, k=k)
        scores = 1 / (1 + distances)

        results = []
        for i, idx in enumerate(indices[0][:k]):
            if idx < 0 or idx >= len(resume_store):
                continue
            entry = resume_store[idx]
            name = entry.get("name") or entry.get("filename") or f"Resume {idx}"
            resume_text = entry.get("text") or entry.get("content") or ""
            score = float(scores[0][i]) if scores is not None else 0.0
            reasoning_key = compute_hash(job_description + name + str(i))
            prompt = build_explain_prompt(job_description, name, resume_text, i + 1, k)
            reasoning = get_reasoning_for_resume(prompt, reasoning_key)
            results.append({
                "rank": i + 1,
                "name": name,
                "score": score,
                "reasoning": reasoning
            })

        # === Step 3: Ranking summary ===
        summary_key = compute_hash(job_description + "_summary")
        ranking_summary = get_cached_value(summary_key)
        if not ranking_summary:
            try:
                comp_prompt = (
                    "You are an experienced hiring manager. Given the job description below and the ranked candidates, "
                    "write a concise 1-paragraph summary (2–3 sentences) explaining why the top candidate is the best fit "
                    "and how the next two compare.\n\n"
                    f"JOB DESCRIPTION:\n{job_description}\n\n"
                )
                for r in results:
                    comp_prompt += f"RANK {r['rank']}: {r['name']} — reason: {r['reasoning']}\n"
                comp_resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": comp_prompt}],
                    max_tokens=120,
                    temperature=0.6
                )
                ranking_summary = comp_resp.choices[0].message["content"].strip()
                set_cached_value(summary_key, ranking_summary)
            except Exception as e:
                logger.error(f"Ranking summary generation error: {e}")
                ranking_summary = ""

        logger.info(f"✅ Search completed successfully for request {job_hash[:12]}...")
        return jsonify({
            "matches": results,
            "ranking_summary": ranking_summary
        })

    except Exception as e:
        logger.exception(f"Backend error in /search: {e}")
        print("\n\n===== ERROR TRACEBACK =====", file=sys.stderr)
        traceback.print_exc()
        print("===== END TRACEBACK =====\n\n", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

