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
# CREAN CORPORATE SPACE THEME
# ============================================================
def inject_space_theme_css():
    st.markdown(
        """
        <style>

        /* ----------------------------------------------------
           GLOBAL FONT + CREAN BRAND COLORS
        ---------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: #F2F7FF !important;
        }

        :root {
            --crean-blue: #00A0DF;
            --crean-blue-light: #0BB8FF;
            --crean-white: #F2F7FF;
            --crean-grey: rgba(255,255,255,0.85);
        }

        /* ----------------------------------------------------
           REMOVE STREAMLIT HAZE (THE DARK LAYER)
        ---------------------------------------------------- */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stApp > header,
        .stApp > div:first-child {
            background: transparent !important;
        }

        /* ----------------------------------------------------
           BACKGROUND — Earth + Satellite + Crean gradient fade
        ---------------------------------------------------- */
        .stApp {
            background:
                linear-gradient(
                    rgba(0, 20, 40, 0.88) 0%,
                    rgba(0, 20, 40, 0.75) 18%,
                    rgba(0, 20, 40, 0.55) 45%,
                    rgba(0, 20, 40, 0.35) 70%,
                    rgba(0, 20, 40, 0.20) 100%
                ),
                var(--space-bg, url("https://wallup.net/wp-content/uploads/2016/01/178756-Earth-space-satellite.jpg"))
                no-repeat center center fixed !important;

            background-size: cover !important;
        }

        /* ----------------------------------------------------
           CREAN GLASS PANEL (Light transparent UI)
        ---------------------------------------------------- */
        .block-container {
            background: rgba(255, 255, 255, 0.06) !important;
            backdrop-filter: blur(16px);
            border-radius: 18px !important;
            padding: 32px;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 0 28px rgba(0,0,0,0.40);
        }

        /* ----------------------------------------------------
           HEADINGS — bold + glowing blue edge
        ---------------------------------------------------- */
        h1, h2, h3, h4, h5, h6 {
            color: var(--crean-white) !important;
            font-weight: 800 !important;
            text-shadow: 0 0 14px rgba(0,160,223,0.55);
        }

        /* ----------------------------------------------------
           TEXT — clean & readable
        ---------------------------------------------------- */
        p, label, span, div, .stMarkdown, .stText {
            color: var(--crean-grey) !important;
        }

        /* ----------------------------------------------------
           INPUT FIELDS
        ---------------------------------------------------- */
        textarea, input, .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.10) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.28) !important;
            backdrop-filter: blur(5px);
        }

        /* ----------------------------------------------------
           CREAN BUTTON — bright blue solid
        ---------------------------------------------------- */
        .stButton > button {
            background: var(--crean-blue) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.75rem 1.8rem;
            font-weight: 700;
            border: none;
            box-shadow: 0 0 20px rgba(0,160,223,0.45);
            transition: 0.15s ease-in-out;
        }

        .stButton > button:hover {
            background: var(--crean-blue-light) !important;
            transform: translateY(-2px);
            box-shadow: 0 0 26px rgba(0,160,223,0.75);
        }

        /* ----------------------------------------------------
           TABS
        ---------------------------------------------------- */
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: white !important;
        }

        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid var(--crean-blue) !important;
            color: var(--crean-blue) !important;
        }

        /* ----------------------------------------------------
           FILE UPLOAD
        ---------------------------------------------------- */
        .stFileUploader {
            background: rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 16px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKGROUND PERSISTENCE
# ============================================================
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


def load_persistent_background():
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as f:
            data = f.read()
        set_space_bg_from_bytes(data)


def save_new_background(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    with open(BACKGROUND_PATH, "wb") as f:
        f.write(bytes_data)

    ext = uploaded_file.name.lower()
    mime = "image/png" if ext.endswith(".png") else "image/webp" if ext.endswith(".webp") else "image/jpeg"
    set_space_bg_from_bytes(bytes_data, mime)


# ============================================================
# APPLY THEME
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
st.caption("Industry-grade AI talent identification — now in your browser.")


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

    job_description = st.text_area("Paste the job description here:", height=200)
    uploaded_file = st.file_uploader("or upload a job description file", type=["pdf", "docx", "txt"])

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
                with st.spinner("Analyzing resumes..."):
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
                st.warning("Backend error")
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
            st.error("Couldn't load resumes.")
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

    new_url = st.text_input("Backend URL:", value=st.session_state.backend_url)

    st.markdown("### Background Image Override")
    st.caption("Upload an Earth-at-night photo to permanently replace the default view.")

    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg", "webp"])

    if st.button("Apply Settings"):
        st.session_state.backend_url = new_url

        if bg_file:
            save_new_background(bg_file)
            st.success("New background saved. Refresh page.")
        else:
            st.success("Settings updated.")
