import requests
from pathlib import Path

def fetch_pdf(url, out_dir="data/papers"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / (url.split("/")[-1] + ".pdf")
    
    r = requests.get(url)
    r.raise_for_status()
    filename.write_bytes(r.content)
    return str(filename)
