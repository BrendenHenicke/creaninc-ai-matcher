import time
import random
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError

# === Load environment variables ===
load_dotenv()

# === Initialize OpenAI client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_with_retry(client, model, texts, max_retries=5, base_delay=2):
    """
    Generate embeddings with retry logic for rate limits or temporary API errors.
    Uses exponential backoff with jitter.
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=texts
            )
            return response

        except (RateLimitError, APIError, APIConnectionError) as e:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"⚠️ API error ({type(e).__name__}): {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"❌ Unrecoverable error: {e}")
            return None

    print("❌ Max retries reached. Skipping batch.")
    return None
