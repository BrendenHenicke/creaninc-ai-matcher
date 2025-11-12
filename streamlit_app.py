import streamlit as st
import requests
from PyPDF2 import PdfReader
import docx
import chardet
import json
import time
import os
import base64
from collections import Counter

# ---------------- CONFIG ----------------
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "https://creaninc-ai-backend.onrender.com")
LOG_FILE = "frontend_logs.json"

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Crean AI Resume Matcher", page_icon="🤖", layout="wide")

# ---------------- CSS HELPERS ----------------
def set_background_from_url(url: str):
    if not url:
        return
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('{url}') no-repeat center center fixed;
            background-size: cover;
        }}
        .block-container {{
            background: rgba(255,255,255,0.85);
            border-radius: 12px;
            padding: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_background_from_bytes(data: bytes, mime="image/jpeg"):
    b64 = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .block-container {{
            background: rgba(255,255,255,0.85);
            border-radius: 12px;
            padding: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- STATE ----------------
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND
if "bg_url" not in st.session_state:
    st.session_state.bg_url = ""
if "bg_file_bytes" not in st.session_state:
    st.session_state.bg_file_bytes = None
if "bg_file_mime" not in st.session_state:
    st.session_state.bg_file_mime = "image/jpeg"

# Background
if st.session_state.bg_file_bytes:
    set_background_from_bytes(st.session_state.bg_file_bytes, st.session_state.bg_file_mime)
elif st.session_state.bg_url:
    set_background_from_url(st.session_state.bg_url)

# ---------------- HEADER ----------------
st.title("🤖 Crean Inc. AI Resume Matcher")
st.caption("Upload or paste a job description and let AI find the best-matched engineers.")

tabs = st.tabs(["🏠 Home", "📂 Resume Manager", "📊 Analytics", "⚙️ Settings"])

# ======================================================================================
# TAB 1: HOME (SEARCH)
# ======================================================================================
with tabs[0]:

    def extract_text(uploaded_file):
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            elif fname.endswith(".docx"):
                doc = docx.Document(uploaded_file)
                return "\n".join(p.text for p in doc.paragraphs)
            elif fname.endswith(".txt"):
                raw = uploaded_file.read()
                enc = chardet.detect(raw)["encoding"]
                return raw.decode(enc or "utf-8", errors="ignore")
            else:
                return ""
        except:
            return ""

    job_description = st.text_area("✍️ Paste the job description here:", height=200)
    uploaded_file = st.file_uploader("📄 Or upload a job description file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        extracted_text = extract_text(uploaded_file)
        if extracted_text:
            job_description += ("\n" if job_description else "") + extracted_text
            st.success(f"Extracted text from {uploaded_file.name}")

    colA, colB = st.columns([1,1])

    with colA:
        if st.button("🔎 Find Matching Engineers", use_container_width=True):
            if not job_description.strip():
                st.warning("Please provide or upload a job description first.")
            else:
                with st.spinner("Analyzing and matching resumes..."):
                    try:
                        t0 = time.time()
                        response = requests.post(
                            f"{st.session_state.backend_url}/search",
                            json={"job_description": job_description},
                            timeout=60
                        )
                        dt = time.time() - t0

                        if response.status_code == 200:
                            results = response.json()
                            matches = results.get("matches", [])

                            if matches:
                                st.success(f"Found {len(matches)} matching engineers (in {dt:.2f}s):")
                                for m in matches:
                                    st.markdown(f"### {m['name']}")
                                    st.write(f"Rank: {m['rank']} • Score: {m['score']:.2f}")
                                    st.write(f"**AI reasoning:** {m['reasoning']}")
                            else:
                                st.info("No matching resumes found.")
                        else:
                            st.error("Backend error.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with colB:
        st.info(f"Backend: **{st.session_state.backend_url}**")
        try:
            r = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("Backend is reachable")
            else:
                st.warning("Backend returned an error")
        except:
            st.warning("Backend not reachable")

# ======================================================================================
# TAB 2: RESUME MANAGER
# ======================================================================================
with tabs[1]:
    st.subheader("Upload Resumes to Backend (Persistent)")

    resume_files = st.file_uploader(
        "Upload one or multiple resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("⬆️ Upload Resumes", use_container_width=True):
        if not resume_files:
            st.warning("No files selected.")
        else:
            files = []
            for f in resume_files:
                mime = (
                    "application/pdf" if f.name.endswith(".pdf") else
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if f.name.endswith(".docx") else
                    "text/plain"
                )
                files.append(("files", (f.name, f.getvalue(), mime)))

            try:
                resp = requests.post(f"{st.session_state.backend_url}/upload_resume", files=files)
                if resp.status_code == 200:
                    st.success(resp.json())
                else:
                    st.error("Upload failed.")
            except Exception as e:
                st.error(f"Error uploading: {e}")

    st.markdown("---")
    st.subheader("Resume Library")

    try:
        r = requests.get(f"{st.session_state.backend_url}/list_resumes")
        if r.status_code == 200:
            items = r.json().get("resumes", [])
            for item in items:
                c1, c2, c3 = st.columns([4,1,1])
                with c1:
                    st.write(f"**[{item['idx']}] {item['name']}** • {item['chars']} chars")
                with c2:
                    if st.button("👁 View", key=f"view{item['idx']}"):
                        prev = requests.get(f"{st.session_state.backend_url}/preview_resume", params={"idx": item["idx"]})
                        if prev.status_code == 200:
                            st.info(prev.json()["snippet"])
                with c3:
                    if st.button("🗑 Delete", key=f"del{item['idx']}"):
                        requests.post(f"{st.session_state.backend_url}/delete_resume", json={"idx": item["idx"]})
                        st.warning("Deleted. Refresh page.")
        else:
            st.error("Could not load resumes.")
    except:
        st.error("Backend not reachable.")

# ======================================================================================
# TAB 3: ANALYTICS
# ======================================================================================
with tabs[2]:
    st.subheader("System Usage Analytics")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        st.metric("Total Searches", len(logs))

        if logs:
            avg = sum(x["duration_sec"] for x in logs) / len(logs)
            st.metric("Avg Response Time", f"{avg:.2f}s")

            by_day = Counter([x["timestamp"][:10] for x in logs])
            st.bar_chart({"Searches": by_day})

            st.write("Recent Activity:")
            for row in logs[-10:][::-1]:
                st.write(f"{row['timestamp']} — {row['num_results']} results")
    else:
        st.info("No analytics available yet.")

# ======================================================================================
# TAB 4: SETTINGS
# ======================================================================================
with tabs[3]:
    st.subheader("Frontend Settings")

    new_url = st.text_input("Backend URL", value=st.session_state.backend_url)

    col1, col2 = st.columns(2)
    with col1:
        bg_url = st.text_input("Background Image URL", value=st.session_state.bg_url)
    with col2:
        bg_file = st.file_uploader("Upload Background Photo", type=["png","jpg","jpeg","webp"])

    if st.button("Apply Settings"):
        st.session_state.backend_url = new_url
        st.session_state.bg_url = bg_url

        if bg_file:
            st.session_state.bg_file_bytes = bg_file.getvalue()
            ext = bg_file.name.lower()
            if ext.endswith(".png"):
                st.session_state.bg_file_mime = "image/png"
            elif ext.endswith(".webp"):
                st.session_state.bg_file_mime = "image/webp"
            else:
                st.session_state.bg_file_mime = "image/jpeg"

        st.success("Settings applied. Refresh for full effect.")
