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


# ============================================================
# CONFIG
# ============================================================
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "https://creaninc-ai-backend.onrender.com")
LOG_FILE = "frontend_logs.json"

BACKGROUND_DIR = "backgrounds"
BACKGROUND_FILE = "space_window_bg.bin"
os.makedirs(BACKGROUND_DIR, exist_ok=True)
BACKGROUND_PATH = os.path.join(BACKGROUND_DIR, BACKGROUND_FILE)

st.set_page_config(page_title="Crean AI Resume Matcher", page_icon="🤖", layout="wide")


# ============================================================
# CINEMATIC SPACE WINDOW THEME (UPDATED + CLEAN)
# ============================================================
def inject_space_theme_css():
    st.markdown(
        """
        <style>

        /* GLOBAL FONT + RESET */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: #eaf3ff !important;
        }

        /* FULL BOLD BACKGROUND (EARTH AT NIGHT) */
        .stApp {
            background: var(--space-bg) no-repeat center center fixed !important;
            background-size: cover !important;
            animation: bgDrift 55s ease-in-out infinite alternate;
        }

        @keyframes bgDrift {
            0%   { background-position: center top; }
            50%  { background-position: center 40px; }
            100% { background-position: center -30px; }
        }

        /* RECTANGULAR SPACECRAFT WINDOW FRAME */
        .stApp::before {
            content: "";
            position: fixed;
            top: 25px;
            left: 25px;
            right: 25px;
            bottom: 25px;

            border-radius: 28px;
            border: 22px solid rgba(15, 15, 18, 0.92);

            box-shadow:
                inset 0 0 75px rgba(0,0,0,0.85),
                inset 0 0 22px rgba(0,0,0,0.6),
                0 0 55px rgba(0,0,0,0.9);

            pointer-events: none;
            z-index: -1;
        }

        /* GLASS PANEL FOR CONTENT */
        .block-container {
            background: rgba(5, 10, 22, 0.55) !important;
            backdrop-filter: blur(16px);
            border-radius: 22px !important;
            border: 1px solid rgba(120, 180, 255, 0.25);
            padding: 32px;
            box-shadow:
                0 0 40px rgba(0, 0, 0, 0.85),
                0 0 55px rgba(50, 120, 255, 0.20);
        }

        /* READABLE TEXT */
        h1, h2, h3, h4, h5, h6 {
            color: #eaf3ff !important;
            font-weight: 700 !important;
            text-shadow: 0 0 18px rgba(80, 180, 255, 0.40);
        }

        p, label, span, div, .stMarkdown {
            color: #ddecff !important;
            font-weight: 500;
        }

        /* TEXT INPUTS / TEXTAREAS */
        textarea, input, .stTextInput>div>div>input {
            background: rgba(255, 255, 255, 0.06) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid rgba(180, 200, 255, 0.3) !important;
        }

        /* NEON BUTTONS */
        .stButton>button {
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: white !important;
            border-radius: 999px;
            padding: 0.65rem 1.6rem;
            font-weight: 700;
            border: none;
            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.75),
                0 0 32px rgba(34, 197, 94, 0.55);
            transition: 0.18s ease-in-out;
        }

        .stButton>button:hover {
            transform: translateY(-2px) scale(1.03);
            box-shadow:
                0 0 25px rgba(56, 189, 248, 1),
                0 0 45px rgba(34, 197, 94, 0.95);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# Inject background from bytes
def set_space_bg_from_bytes(data: bytes, mime: str = "image/jpeg"):
    b64 = base64.b64encode(data).decode("utf-8")
    uri = f"data:{mime};base64,{b64}"

    st.markdown(
        f"""
        <style>
            :root {{
                --space-bg: url("{uri}");
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Load stored background
def load_persistent_background():
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as f:
            data = f.read()
        set_space_bg_from_bytes(data)


# Save new background
def save_new_background(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    with open(BACKGROUND_PATH, "wb") as f:
        f.write(bytes_data)

    ext = uploaded_file.name.lower()
    mime = (
        "image/png" if ext.endswith(".png")
        else "image/webp" if ext.endswith(".webp")
        else "image/jpeg"
    )
    set_space_bg_from_bytes(bytes_data, mime)


# ============================================================
# APPLY THE THEME
# ============================================================
inject_space_theme_css()
load_persistent_background()


# ============================================================
# BACKEND URL STATE
# ============================================================
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND


# ============================================================
# HEADER
# ============================================================
st.title("🚀 Crean Inc. AI Resume Matcher")
st.caption("Explore your talent universe through a cinematic space-station window.")


tabs = st.tabs(["🏠 Home", "📂 Resume Manager", "📊 Analytics", "⚙️ Settings"])


# ============================================================
# TAB 1: HOME
# ============================================================
with tabs[0]:

    def extract_text(uploaded_file):
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            elif name.endswith(".docx"):
                doc = docx.Document(uploaded_file)
                return "\n".join(p.text for p in doc.paragraphs)
            elif name.endswith(".txt"):
                raw = uploaded_file.read()
                enc = chardet.detect(raw)["encoding"]
                return raw.decode(enc or "utf-8", errors="ignore")
            return ""
        except:
            return ""

    job_description = st.text_area("✍️ Paste the job description here:", height=200)
    uploaded_file = st.file_uploader("📄 Or upload a job description file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        extracted = extract_text(uploaded_file)
        if extracted:
            job_description += ("\n" if job_description else "") + extracted
            st.success(f"Extracted text from {uploaded_file.name}")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("🔎 Find Matching Engineers", use_container_width=True):
            if not job_description.strip():
                st.warning("Please provide or upload a job description first.")
            else:
                with st.spinner("Analyzing resumes across your talent universe..."):
                    try:
                        t0 = time.time()
                        res = requests.post(
                            f"{st.session_state.backend_url}/search",
                            json={"job_description": job_description},
                            timeout=60,
                        )
                        dt = time.time() - t0

                        if res.status_code == 200:
                            data = res.json()
                            matches = data.get("matches", [])

                            if matches:
                                st.success(f"Found {len(matches)} matching engineers (in {dt:.2f}s):")
                                for m in matches:
                                    st.markdown(f"### ⭐ {m['name']}")
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
            ping = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
            if ping.status_code == 200:
                st.success("Backend online ✔")
            else:
                st.warning("Backend returned error")
        except:
            st.warning("Backend offline")


# ============================================================
# TAB 2: RESUME MANAGER
# ============================================================
with tabs[1]:
    st.subheader("Upload Resumes to Backend (Persistent Storage)")

    resume_files = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.button("⬆️ Upload Resumes", use_container_width=True):
        if not resume_files:
            st.warning("No files selected.")
        else:
            files = []
            for f in resume_files:
                ext = f.name.lower()
                mime = (
                    "application/pdf" if ext.endswith(".pdf") else
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext.endswith(".docx") else
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
                st.error(f"Upload error: {e}")

    st.markdown("---")
    st.subheader("Resume Library")

    try:
        resp = requests.get(f"{st.session_state.backend_url}/list_resumes")
        if resp.status_code == 200:
            items = resp.json().get("resumes", [])
            for item in items:
                c1, c2, c3 = st.columns([4, 1, 1])
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
                        st.warning("Deleted. Refresh the page.")
        else:
            st.error("Could not load resumes.")
    except:
        st.error("Backend not reachable.")


# ============================================================
# TAB 3: ANALYTICS
# ============================================================
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
        st.info("No analytics yet.")


# ============================================================
# TAB 4: SETTINGS
# ============================================================
with tabs[3]:
    st.subheader("Frontend Settings")

    st.write("Backend URL:")
    new_url = st.text_input("", value=st.session_state.backend_url)

    st.markdown("### Space Station Background Image")
    st.caption("Upload a new Earth-at-night photo to permanently change the window view.")

    bg_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])

    if st.button("Apply Settings"):
        st.session_state.backend_url = new_url

        if bg_file:
            save_new_background(bg_file)
            st.success("New background saved. Refresh page.")
        else:
            st.success("Settings updated.")

