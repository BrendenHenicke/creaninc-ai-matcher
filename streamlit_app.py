import streamlit as st
import requests
from PyPDF2 import PdfReader
import docx
import chardet
import io
import json
import time
import os
import base64
from collections import Counter
from urllib.parse import urlparse

# ---------- Config ----------
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
LOG_FILE = "frontend_logs.json"

# ---------- Page ----------
st.set_page_config(page_title="Crean AI Resume Matcher", page_icon="🤖", layout="wide")

# ---------- CSS Helpers ----------
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

def set_background_from_bytes(data: bytes, mime: str = "image/jpeg"):
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

# ---------- State ----------
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND
if "bg_url" not in st.session_state:
    st.session_state.bg_url = ""
if "bg_file_bytes" not in st.session_state:
    st.session_state.bg_file_bytes = None
if "bg_file_mime" not in st.session_state:
    st.session_state.bg_file_mime = "image/jpeg"

# ---------- Background (apply early) ----------
if st.session_state.bg_file_bytes:
    set_background_from_bytes(st.session_state.bg_file_bytes, st.session_state.bg_file_mime)
elif st.session_state.bg_url:
    set_background_from_url(st.session_state.bg_url)

st.title("🤖 Crean Inc. AI Resume Matcher")
st.caption("Upload or paste a job description and let AI find the best-matched engineers.")

tabs = st.tabs(["🏠 Home", "📂 Resume Manager", "📊 Analytics", "⚙️ Settings"])

# =======================
# Tab 1: Home (Matching)
# =======================
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
            elif fname.endswith(".json"):
                return uploaded_file.getvalue().decode("utf-8")
            else:
                return ""
        except Exception as e:
            st.error(f"❌ Could not read file: {uploaded_file.name} ({e})")
            return ""

    job_description = st.text_area("✍️ Paste the job description here:", height=200)
    uploaded_file = st.file_uploader("📄 Or upload a job description file", type=["pdf", "docx", "txt", "json"])

    if uploaded_file:
        extracted_text = extract_text(uploaded_file)
        if extracted_text:
            st.success(f"✅ Extracted text from {uploaded_file.name}")
            job_description += ("\n" if job_description else "") + extracted_text

    def log_search_event(job_desc, num_results, duration):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_snippet": (job_desc[:80] + "...") if len(job_desc) > 80 else job_desc,
            "num_results": num_results,
            "duration_sec": round(duration, 2)
        }
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except:
                logs = []
        logs.append(entry)
        logs = logs[-200:]
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🔎 Find Matching Engineers", use_container_width=True):
            if not job_description.strip():
                st.warning("Please provide or upload a job description first.")
            else:
                with st.spinner("Analyzing and matching resumes..."):
                    t0 = time.time()
                    try:
                        response = requests.post(f"{st.session_state.backend_url}/search",
                                                 json={"job_description": job_description},
                                                 timeout=60)
                        dt = time.time() - t0
                        if response.status_code == 200:
                            results = response.json()
                            matches = results.get("matches", [])
                            num = len(matches)
                            log_search_event(job_description, num, dt)

                            if num > 0:
                                st.success(f"✅ Found {num} matching engineers (in {dt:.2f}s):")
                                for m in matches:
                                    with st.container(border=True):
                                        st.markdown(f"**{m['name']}**")
                                        st.write(f"Rank: {m['rank']}  •  Relevance: {m['score']:.2f}")
                                        st.write(f"**AI reasoning:** {m['reasoning']}")
                                if results.get("ranking_summary"):
                                    st.markdown("---")
                                    st.subheader("📋 Overall Ranking Summary")
                                    st.write(results["ranking_summary"])
                            else:
                                st.info("No matching resumes found.")
                        else:
                            st.error("❌ Error from backend. Check backend logs.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"⚠️ Could not connect to backend at {st.session_state.backend_url} ({e})")

    with colB:
        st.info(f"Backend: **{st.session_state.backend_url}**")
        try:
            r = requests.get(f"{st.session_state.backend_url}/health", timeout=10)
            if r.status_code == 200:
                st.success("Backend is reachable ✅")
            else:
                st.warning(f"Backend health check returned {r.status_code}")
        except Exception as e:
            st.warning(f"Health check failed: {e}")

# ==========================
# Tab 2: Resume Manager
# ==========================
with tabs[1]:
    st.subheader("Upload resumes (PDF, DOCX, TXT)")
    up_files = st.file_uploader("Select one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("⬆️ Upload to Backend", use_container_width=True):
        if not up_files:
            st.warning("Select at least one file.")
        else:
            files = []
            for f in up_files:
                # requests requires (name, bytes, mime)
                mime = "application/octet-stream"
                if f.name.lower().endswith(".pdf"):
                    mime = "application/pdf"
                elif f.name.lower().endswith(".docx"):
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif f.name.lower().endswith(".txt"):
                    mime = "text/plain"
                files.append(("files", (f.name, f.getvalue(), mime)))
            try:
                resp = requests.post(f"{st.session_state.backend_url}/upload_resume", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Uploaded: {data.get('added', 0)} file(s). Total resumes: {data.get('total', 0)}")
                else:
                    st.error(f"Upload failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.markdown("---")
    st.subheader("Resume Library")
    try:
        resp = requests.get(f"{st.session_state.backend_url}/list_resumes", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("resumes", []):
                c1, c2, c3, c4 = st.columns([3,1,1,1])
                with c1:
                    st.write(f"**[{item['idx']}]** {item['name']}  •  {item['chars']} chars")
                with c2:
                    if st.button("👁 Preview", key=f"pv{item['idx']}"):
                        prev = requests.get(f"{st.session_state.backend_url}/preview_resume", params={"idx": item["idx"]})
                        if prev.status_code == 200:
                            st.info(prev.json().get("snippet", ""))
                        else:
                            st.error("Preview failed.")
                with c3:
                    if st.button("🗑 Delete", key=f"del{item['idx']}"):
                        r = requests.post(f"{st.session_state.backend_url}/delete_resume", json={"idx": item["idx"]})
                        if r.status_code == 200:
                            st.success("Deleted. Refresh the page to see updates.")
                        else:
                            st.error("Delete failed.")
                with c4:
                    pass
        else:
            st.error("Could not fetch resume list.")
    except Exception as e:
        st.error(f"List error: {e}")

# ==========================
# Tab 3: Analytics
# ==========================
with tabs[2]:
    st.subheader("Usage Analytics")
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except:
            logs = []
    total = len(logs)
    st.metric("Total Searches", total)
    if total > 0:
        avg_time = sum(x.get("duration_sec", 0.0) for x in logs) / total
        st.metric("Avg Response Time (s)", f"{avg_time:.2f}")

        # searches per day
        by_day = Counter([x["timestamp"][:10] for x in logs])
        st.bar_chart({"searches": by_day})
        st.write("Recent activity:")
        for row in logs[-10:][::-1]:
            st.write(f"{row['timestamp']} — {row['num_results']} results — “{row['job_snippet']}”")
    else:
        st.info("No logs yet.")

# ==========================
# Tab 4: Settings
# ==========================
with tabs[3]:
    st.subheader("Backend & Theme")
    new_url = st.text_input("Backend URL", value=st.session_state.backend_url, help="Example: https://creaninc-ai-backend.onrender.com")
    col1, col2 = st.columns(2)
    with col1:
        bg_url = st.text_input("Background Image URL", value=st.session_state.bg_url)
    with col2:
        bg_file = st.file_uploader("…or Upload Background Image", type=["png","jpg","jpeg","webp"])

    if st.button("Apply Settings", use_container_width=True):
        st.session_state.backend_url = new_url.strip() or st.session_state.backend_url
        st.session_state.bg_url = bg_url.strip()
        if bg_file is not None:
            st.session_state.bg_file_bytes = bg_file.getvalue()
            # infer mime
            ext = (bg_file.name or "").lower()
            mime = "image/jpeg"
            if ext.endswith(".png"): mime = "image/png"
            if ext.endswith(".webp"): mime = "image/webp"
            st.session_state.bg_file_mime = mime
        st.success("Settings applied. Reload the page to ensure styles stick.")
