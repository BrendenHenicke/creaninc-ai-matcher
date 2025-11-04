import os
import time
import json
import concurrent.futures
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from openai_utils import embed_with_retry  # 👈 custom retry wrapper (already built)
from PyPDF2 import PdfReader
import docx

# -------------------------------
# CONFIG
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_DIR = os.path.join(SCRIPT_DIR, "resumes")  # ✅ absolute path
OUTPUT_FILE = "resume_embeddings.json"
MODEL = "text-embedding-3-large"
BATCH_SIZE = 32
client = OpenAI()

# -------------------------------
# HELPERS
# -------------------------------
def read_resume(file_path):
    """Read a resume file (PDF, DOCX, or TXT) and return plain text."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"⚠️ Unsupported file format: {file_path}")
            return ""
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""
    return text.strip()

def load_existing_embeddings(output_file):
    """Load cache if embeddings already exist."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings, output_file):
    """Save updated embeddings to JSON."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

# -------------------------------
# MAIN
# -------------------------------
def main():
    print(f"\n📂 Looking for resumes in: {RESUME_DIR}")
    resume_files = [
        f for f in os.listdir(RESUME_DIR)
        if f.lower().endswith((".pdf", ".docx", ".txt"))
    ]

    if not resume_files:
        print("❌ No resumes found in directory.")
        return

    print(f"✅ Found {len(resume_files)} resumes:")
    for f in resume_files:
        print(f"   • {f}")

    # Load cache
    embeddings = load_existing_embeddings(OUTPUT_FILE)

    # Identify files that need embedding
    unprocessed_files = [f for f in resume_files if f not in embeddings]
    if not unprocessed_files:
        print("✨ All resumes already embedded.")
        return

    print(f"\n🧠 Embedding {len(unprocessed_files)} new resumes...")

    # Read all files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        contents = list(
            tqdm(
                executor.map(lambda f: read_resume(os.path.join(RESUME_DIR, f)), unprocessed_files),
                total=len(unprocessed_files),
                desc="📖 Reading resumes"
            )
        )

    # Batch embedding
    for i in tqdm(range(0, len(contents), BATCH_SIZE), desc="🚀 Generating embeddings"):
        batch_files = unprocessed_files[i:i+BATCH_SIZE]
        batch_texts = contents[i:i+BATCH_SIZE]

        # Filter out any empty texts
        valid_pairs = [(f, t) for f, t in zip(batch_files, batch_texts) if t.strip()]
        if not valid_pairs:
            continue

        valid_files, valid_texts = zip(*valid_pairs)
        response = embed_with_retry(client, MODEL, valid_texts)

        if response is None:
            print("❌ Failed after retries, skipping batch.")
            continue

        for j, file_name in enumerate(valid_files):
            embeddings[file_name] = {
                "embedding": response.data[j].embedding,
                "timestamp": time.time()
            }

        # Save after each batch
        save_embeddings(embeddings, OUTPUT_FILE)

    print("\n✅ All embeddings complete and saved!")

# -------------------------------
if __name__ == "__main__":
    main()


