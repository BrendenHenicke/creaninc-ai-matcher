<<<<<<< HEAD
import streamlit as st
import os
import json
import sqlite3
import uuid
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from feedback_db import init_db, insert_feedback

import docx2txt
import PyPDF2
import pandas as pd

# --- Initialize database and environment ---
init_db()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

st.title("Crean AI Engineer Matcher")

# --- Upload Engineer Resumes ---
st.subheader("Step 1: Upload Engineer Resumes Database")
resumes_file = st.file_uploader(
    "Upload resumes file (JSON, TXT, DOCX, PDF, XLSX)",
    type=["json", "txt", "docx", "pdf", "xlsx"],
    key="resumes"
)

# --- Upload Job Descriptions ---
st.subheader("Step 2: Upload Job Description")
job_file = st.file_uploader(
    "Upload job description file (JSON, TXT, DOCX, PDF, XLSX)",
    type=["json", "txt", "docx", "pdf", "xlsx"],
    key="job"
)

job_description = None
resumes = []

# --- Extraction Function ---
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_type == "docx":
        return docx2txt.process(uploaded_file)
    elif file_type == "txt":
        return uploaded_file.read().decode("utf-8")
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file)
        return df.to_string(index=False)
    elif file_type == "json":
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

# --- Handle job description file ---
if job_file:
    try:
        jd_text = extract_text_from_file(job_file)
        if job_file.name.endswith("json"):
            job_data = json.loads(jd_text)
            job_description = job_data.get("job_description", jd_text)
        else:
            job_description = jd_text
    except Exception as e:
        st.error(f"Failed to process job description file: {e}")

# --- Handle resumes file ---
if resumes_file:
    try:
        res_text = extract_text_from_file(resumes_file)
        if resumes_file.name.endswith("json"):
            res_data = json.loads(res_text)
            resumes = res_data.get("resumes", [])
        else:
            resumes = [
                {"name": f"Candidate {i+1}", "skills": "", "experience": chunk.strip()}
                for i, chunk in enumerate(res_text.split("\n\n")) if chunk.strip()
            ]
    except Exception as e:
        st.error(f"Failed to process resumes file: {e}")

# --- Manual Entry Fallback ---
manual_input = st.text_area("Or paste job description and engineer details manually")
if manual_input:
    job_description = manual_input

# --- Simulated Engineer Database if none uploaded ---
if not resumes:
    resumes = [
        {"name": "Alice", "skills": "Python, SQL", "experience": "3 years in data analysis"},
        {"name": "Bob", "skills": "Java, AWS", "experience": "5 years in cloud infrastructure"},
        {"name": "Charlie", "skills": "C++, robotics", "experience": "2 years in embedded systems"},
        {"name": "Dana", "skills": "JavaScript, React", "experience": "4 years in frontend dev"},
        {"name": "Eli", "skills": "Python, Machine Learning", "experience": "6 years in AI"}
    ]

# --- Proceed if job description is available ---
if job_description:
    job_embedding = embedding_model.embed_query(job_description)
    resume_embeddings = []
    for resume in resumes:
        combined_text = f"{resume['name']}\nSkills: {resume['skills']}\nExperience: {resume['experience']}"
        vector = embedding_model.embed_query(combined_text)
        resume_embeddings.append((resume, vector))

    scored_resumes = [
        (resume, cosine_similarity([job_embedding], [vector])[0][0])
        for resume, vector in resume_embeddings
    ]

    top_5 = sorted(scored_resumes, key=lambda x: x[1], reverse=True)[:5]
    ranked_text = "\n\n".join(
        [f"Name: {r['name']}\nSkills: {r['skills']}\nExperience: {r['experience']}\nSimilarity Score: {round(s, 4)}" for r, s in top_5]
    )

    system_msg = SystemMessage(content="You are an expert recruiter. Explain why these 5 engineers were selected based on the job description.")
    user_msg = HumanMessage(content=f"Job Description:\n{job_description}\n\nTop 5 Engineers:\n{ranked_text}")
    response = llm.invoke([system_msg, user_msg])
    result_text = response.content

    st.subheader("Top Matches:")
    st.write(result_text)

    st.subheader("Provide Feedback for Each Candidate")
    candidate_names = [line.split('.', 1)[1].split(':', 1)[0].strip() for line in result_text.splitlines() if line.strip().startswith(tuple(str(i) for i in range(1, 6)))]

    feedback_list = []
    for name in candidate_names:
        st.markdown(f"### Feedback for {name}")
        rating = st.slider(f"Rate {name} (1=Poor, 5=Excellent)", 1, 5, key=f"rating_{name}")
        notes = st.text_area(f"Notes for {name}", key=f"notes_{name}")
        feedback_list.append({'name': name, 'rating': rating, 'notes': notes})

    st.subheader("Overall Feedback")
    user_feedback = st.text_area("Any other comments or suggestions?")

    if st.button("Submit Feedback"):
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "job_description": job_description,
            "resumes": resumes,
            "ai_output": result_text,
            "per_candidate_feedback": feedback_list,
            "overall_feedback": user_feedback
        }
        insert_feedback(feedback_entry)
        st.success("Feedback submitted! Thank you.")

    # --- Sidebar Summary ---
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    st.sidebar.subheader("\U0001F4CA Feedback Summary")

    c.execute("SELECT COUNT(*) FROM feedback")
    total = c.fetchone()[0]
    c.execute("SELECT AVG(rating) FROM candidate_feedback")
    avg_rating = round(c.fetchone()[0] or 0, 2)
    c.execute("SELECT overall_feedback FROM feedback")
    all_comments = " ".join([row[0] for row in c.fetchall()])
    keyword_counts = Counter(all_comments.lower().split())
    common_keywords = ", ".join([word for word, _ in keyword_counts.most_common(5)])

    st.sidebar.markdown(f"**Total Feedback Entries:** {total}")
    st.sidebar.markdown(f"**Avg Rating:** {avg_rating}")
    st.sidebar.markdown(f"**Top Keywords:** {common_keywords}")

    c.execute("SELECT overall_feedback FROM feedback ORDER BY timestamp DESC LIMIT 3")
    st.sidebar.markdown("**Latest Comments:**")
    for row in c.fetchall():
        st.sidebar.info(row[0])

    conn.close()

else:
    st.info("Please upload at least one file or enter job description text to proceed.")
=======
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
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

# === Logging configuration ===
if not os.path.exists("logs"):
    os.makedirs("logs")

handler = RotatingFileHandler("logs/backend.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# === Initialize Flask app ===
app = Flask(__name__)

# Load environment variables and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Load FAISS index and resume data ===
try:
    index = faiss.read_index("resume_index.faiss")
    with open("resume_store.pkl", "rb") as f:
        resume_store = pickle.load(f)
    logger.info("FAISS index and resume data loaded successfully.")
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

# === Flask routes ===
@app.route("/")
def home():
    logger.info("Health check at root route.")
    return "✅ Flask backend running successfully!"

@app.route("/search", methods=["POST"])
def search_resumes():
    try:
        job_description = None

        # Handle upload or JSON
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
        job_vector = client.embeddings.create(
            model="text-embedding-3-small",
            input=job_description
        ).data[0].embedding
        job_vector = np.array(job_vector).astype("float32").reshape(1, -1)

        # === Step 2: Search FAISS ===
        k = 5
        distances, indices = index.search(job_vector, k=k)
        scores = 1 / (1 + distances)

        results = []
        total_found = min(k, indices.shape[1] if indices is not None else 0)

        for i, idx in enumerate(indices[0][:k]):
            if idx < 0 or idx >= len(resume_store):
                name = f"Resume {idx}"
                resume_text = ""
            else:
                entry = resume_store[idx]
                name = entry.get("name") or entry.get("filename") or entry.get("title") or f"Resume {idx}"
                resume_text = entry.get("text") or entry.get("content") or ""

            score = float(scores[0][i]) if scores is not None else 0.0

            reasoning_key = compute_hash(job_description + name + str(i))
            prompt = build_explain_prompt(job_description, name, resume_text, i + 1, total_found)
            reasoning = get_reasoning_for_resume(prompt, reasoning_key)

            results.append({
                "rank": i + 1,
                "name": name,
                "score": score,
                "reasoning": reasoning
            })

        # === Step 4: Ranking summary (cached) ===
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
                comp_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": comp_prompt}],
                    max_tokens=120,
                    temperature=0.6
                )
                ranking_summary = comp_resp.choices[0].message.content.strip()
                set_cached_value(summary_key, ranking_summary)
            except Exception as e:
                logger.error(f"Ranking summary generation error: {e}")
                ranking_summary = ""

        logger.info(f"Search completed successfully for request {job_hash[:12]}...")
        return jsonify({
            "matches": results,
            "ranking_summary": ranking_summary
        })

    except Exception as e:
        logger.exception(f"Backend error in /search: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
>>>>>>> 10771d2d (Initial commit for Crean AI Matcher full app)


