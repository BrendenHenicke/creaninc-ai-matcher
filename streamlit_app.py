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
import numpy as np   # <-- needed for the graph

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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        :root {
            --crean-blue: #00A0DF;
            --crean-white: #F2F7FF;
            --crean-grey: rgba(255,255,255,0.86);
        }

        .stApp {
            background:
                linear-gradient(
                    rgba(0, 25, 55, 0.40) 0%,
                    rgba(0, 25, 55, 0.25) 35%,
                    rgba(0, 25, 55, 0.10) 65%,
                    rgba(0, 25, 55, 0.05) 100%
                ),
                var(--space-bg, url("https://wallup.net/wp-content/uploads/2016/01/178756-Earth-space-satellite.jpg"))
                no-repeat center center fixed !important;

            background-size: cover !important;
        }

        .block-container {
            background: rgba(255, 255, 255, 0.04) !important;
            backdrop-filter: blur(16px);
            border-radius: 18px !important;
            padding: 32px;
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 0 32px rgba(0,0,0,0.35);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--crean-white) !important;
            font-weight: 800 !important;
            text-shadow: 0px 0px 14px rgba(0, 160, 223, 0.55);
        }

        p, label, span, div, .stMarkdown, .stText {
            color: var(--crean-grey) !important;
            font-weight: 500;
        }

        textarea, input, .stTextInput>div>div>input {
            background: rgba(255, 255, 255, 0.10) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
        }

        .stButton>button {
            background: var(--crean-blue) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.7rem 1.7rem;
            font-weight: 700;
            border: none;
            box-shadow: 0px 0px 20px rgba(0,160,223,0.45);
            transition: 0.15s ease-in-out;
        }

        .stButton>button:hover {
            background: #0BB8FF !important;
            transform: translateY(-2px);
            box-shadow: 0px 0px 26px rgba(0,160,223,0.75);
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 1.05rem;
            font-weight: 700;
            color: white !important;
        }

        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid var(--crean-blue) !important;
            color: var(--crean-blue) !important;
        }

        .stFileUploader {
            background: rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 12px;
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

tabs = st.tabs(["🏠 Home", "📂 Resume Manager", "📄 Proposals", "⚙️ Settings"])


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
        except Exception:
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

                            # New: store results for graph + proposals tabs
                            st.session_state.last_results = matches
                            st.session_state.last_job_description = job_description

                            # Fetch graph embeddings
                            try:
                                graph_res = requests.get(f"{st.session_state.backend_url}/graph_data", timeout=10)
                                if graph_res.status_code == 200:
                                    st.session_state.last_graph_data = graph_res.json()
                            except:
                                st.warning("Graph data unavailable from backend.")

                            if matches:
                                st.success(f"Found {len(matches)} matching engineers (in {dt:.2f}s):")
                                for m in matches:
                                    st.markdown(f"### ⭐ {m['name']}")
                                    st.write(f"Rank: {m['rank']} • Score: {m['score']:.2f}")
                                    st.write("**Fit + Gap Analysis:**")
                                    st.write(m["reasoning"])
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
        except Exception:
            st.warning("Backend offline")


# ============================================================
# TAB 2: RESUME MANAGER
# (unchanged)
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
                        prev = requests.get(
                            f"{st.session_state.backend_url}/preview_resume",
                            params={"idx": item["idx"]},
                        )
                        if prev.status_code == 200:
                            st.info(prev.json().get("snippet", ""))
                with c3:
                    if st.button("🗑 Delete", key=f"del{item['idx']}"):
                        requests.post(
                            f"{st.session_state.backend_url}/delete_resume",
                            json={"idx": item["idx"]},
                        )
                        st.warning("Deleted. Refresh the page.")
        else:
            st.error("Couldn't load resumes.")
    except Exception:
        st.error("Backend not reachable.")


# ============================================================
# TAB 3: PROPOSALS *** UPDATED WITH MULTI-LINE GRAPH ***
# ============================================================
with tabs[2]:
    st.subheader("Client-Ready Engineer Proposals")

    results = st.session_state.get("last_results", None)

    if not results:
        st.info("Run a search on the Home tab to generate proposals.")
    else:
        for m in results:
            st.markdown(f"### Proposal for {m['name']} (Rank #{m['rank']})")
            st.write(m.get("proposal", "No proposal available."))
            st.markdown("---")

    st.subheader("📈 Resume–Job Description Similarity Graph")

    if "last_graph_data" in st.session_state:
        graph_data = st.session_state["last_graph_data"]

        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame(graph_data["engineers"])
        job_vec = np.array(graph_data["job_description"])

        # cosine similarity function
        def cosine_similarity(a, b):
            return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

        df["similarity"] = df["embedding"].apply(
            lambda emb: cosine_similarity(np.array(emb), job_vec)
        )

        fig = go.Figure()

        # plot each engineer
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[row["similarity"], row["similarity"]],
                mode="lines+markers",
                name=row["name"],
                line=dict(width=3)
            ))

        # baseline
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[1, 1],
            mode="lines",
            name="Job Description Baseline",
            line=dict(color="white", width=4, dash="dot")
        ))

        fig.update_layout(
            title="Resume Similarity vs Job Description",
            xaxis_title="Line (not time-based)",
            yaxis_title="Cosine Similarity",
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Graph will appear after first search.")


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
