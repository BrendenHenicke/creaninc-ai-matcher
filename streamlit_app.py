import streamlit as st
import requests
from PyPDF2 import PdfReader
import docx
import chardet
import io
import json
import time
import os

# ========================
# ✅ Backend URL (Render)
# ========================
BACKEND_URL = "https://creaninc-ai-backend.onrender.com"

LOG_FILE = "frontend_logs.json"

# --- Streamlit config ---
st.set_page_config(page_title="Crean AI Resume Matcher", page_icon="🤖", layout="wide")
st.title("🤖 Crean Inc. AI Resume Matcher")
st.markdown("Upload or paste a job description to let AI match the best engineers from your resume database.")

# --- Logging helper ---
def log_search_event(job_description, num_results, duration):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_snippet": (job_description[:80] + "...") if len(job_description) > 80 else job_description,
        "num_results": num_results,
        "duration_sec": round(duration, 2)
    }

    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(entry)
        logs = logs[-200:]
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except:
        pass

# --- Extract text helper ---
def extract_text(uploaded_file):
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif uploaded_file.name.endswith(".txt"):
            raw = uploaded_file.read()
            encoding = chardet.detect(raw)["encoding"]
            return raw.decode(encoding or "utf-8", errors="ignore")
        elif uploaded_file.name.endswith(".json"):
            return uploaded_file.getvalue().decode("utf-8")
        else:
            return ""
    except:
        return ""

# --- Input area ---
job_description = st.text_area("✍️ Paste the job description here:", height=200)
uploaded_file = st.file_uploader("📄 Or upload a file", type=["pdf", "docx", "txt", "json"])

if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        st.success(f"Extracted text from {uploaded_file.name}")
        job_description += "\n" + extracted_text

# --- Submit button ---
if st.button("Find Matching Engineers"):
    if not job_description.strip():
        st.warning("Please paste or upload a job description first.")
    else:
        with st.spinner("Matching resumes..."):
            start = time.time()
            try:
                response = requests.post(f"{BACKEND_URL}/search", json={"job_description": job_description})
                duration = time.time() - start

                if response.status_code != 200:
                    st.error("Backend returned an error. Check backend logs on Render.")
                else:
                    results = response.json()

                    if "matches" in results and results["matches"]:
                        matches = results["matches"]
                        log_search_event(job_description, len(matches), duration)

                        st.success(f"Found {len(matches)} matching engineers:")
                        for match in matches:
                            st.markdown(
                                f"### {match['name']}\n"
                                f"**Score:** {match['score']:.2f}\n\n"
                                f"**Rank:** {match['rank']}\n\n"
                                f"💬 **AI Explanation:** {match['reasoning']}\n"
                            )

                        if "ranking_summary" in results:
                            st.markdown("---")
                            st.subheader("📋 Ranking Summary")
                            st.write(results["ranking_summary"])
                    else:
                        st.info("No resumes matched.")

            except Exception as e:
                st.error("Could not connect to backend. Is it deployed on Render?")
                st.error(str(e))

st.markdown("---")
st.caption("Crean Inc. AI Resume Matcher © 2025")
