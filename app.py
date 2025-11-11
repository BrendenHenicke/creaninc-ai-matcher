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
import httpx

# ===== Logging (no emojis to avoid Windows console errors) =====
if not os.path.exists("logs"):
    os.makedirs("logs")

handler = RotatingFileHandler("logs/backend.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ===== Flask =====
app = Flask(__name__)
CORS(app)

# ===== Environment =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

# ===== Remove proxy variables =====
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:
    os.environ.pop(k, None)

# ===== OpenAI Client =====
client = OpenAI(
    api_key=api_key,
    http_client=httpx.Client(transport=httpx.HTTPTransport(proxy=None))
)

# ===== Load FAISS =====
if os.path.exists("resume_index.faiss") and os.path.exists("resume_store.pkl"):
    index = faiss.read_index("resume_index.faiss")
    resume_store = pickle.load(open("resume_store.pkl","rb"))
else:
    index = faiss.IndexFlatL2(1536)
    resume_store = []

# ===== SQLite cache =====
CACHE_DB = "reasoning_cache.db"

def get_db_connection():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            created real
        )
    """)
    return conn

def compute_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def cache_get(key):
    conn = get_db_connection()
    row = conn.execute("SELECT value FROM cache WHERE key=?",(key,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def cache_set(key, value):
    conn = get_db_connection()
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                 (key,json.dumps(value),time.time()))
    conn.commit()
    conn.close()

# ===== File extraction =====
def extract_text(file):
    name = file.filename.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file.read()))
            file.seek(0)
            return "".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            data = io.BytesIO(file.read())
            file.seek(0)
            doc = docx.Document(data)
            return "\n".join(p.text for p in doc.paragraphs)
        if name.endswith(".txt"):
            text = file.read().decode("utf-8","ignore")
            file.seek(0)
            return text
        return ""
    except:
        return ""

# ===== Prompt builder =====
def build_prompt(jd, name, resume, rank, total):
    short = resume[:3000]
    return (
        f"Explain in 3-5 sentences why this resume ranked {rank} of {total}.\n"
        f"JOB DESCRIPTION:\n{jd}\n\n"
        f"RESUME:\n{short}\n"
    )

# ===== Explanation generator =====
def explain(prompt, key):
    cached = cache_get(key)
    if cached:
        return cached

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=150
    )
    msg = resp.choices[0].message.content.strip()
    cache_set(key,msg)
    return msg

# ===== Routes =====
@app.route("/")
def home():
    return "Flask backend running."

@app.route("/search", methods=["POST"])
def search():
    try:
        if "file" in request.files:
            jd = extract_text(request.files["file"])
        else:
            jd = request.json.get("job_description","")

        if not jd.strip():
            return jsonify({"error":"Job description missing"}),400

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=jd
        )
        vec = np.array(emb.data[0].embedding).astype("float32").reshape(1,-1)

        if index.ntotal == 0:
            return jsonify({"matches":[]})

        k = min(5,index.ntotal)
        dist, idxs = index.search(vec,k)
        scores = 1/(1+dist)

        out=[]
        for i,idx in enumerate(idxs[0]):
            res = resume_store[idx]
            name = res.get("name",f"Resume {idx}")
            text = res.get("text","")
            score = float(scores[0][i])

            key = compute_hash(jd+name)
            prompt = build_prompt(jd,name,text,i+1,k)
            reason = explain(prompt,key)

            out.append({
                "rank":i+1,
                "name":name,
                "score":score,
                "reasoning":reason
            })

        return jsonify({"matches":out})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

# ===== Run Flask (CRITICAL FIX: no reloader) =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


