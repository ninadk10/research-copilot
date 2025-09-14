import os, requests
from dotenv import load_dotenv

# Load your token from .env
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("No Hugging Face token found. Please set HUGGINGFACEHUB_API_TOKEN in your .env")

headers = {"Authorization": f"Bearer {token}"}

def test_model(model_id, prompt):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "temperature": 0.0},
        "options": {"use_cache": False}  # force fresh response
    }
    resp = requests.post(url, headers=headers, json=payload)
    print(f"\n🔹 Model: {model_id}")
    print("Status:", resp.status_code)
    try:
        print("Response:", resp.json())
    except Exception:
        print("Raw:", resp.text)

if __name__ == "__main__":
    prompt = "What is an LLM?"

    # Try a few known models
    test_model("google/flan-t5-small", prompt)
    test_model("facebook/bart-large-cnn", "Summarize: Large language models are AI systems ...")
    test_model("bigscience/bloomz-560m", prompt)
