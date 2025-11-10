# Nexus Reflective API

A minimal self-evolving language memory and reflection engine (FastAPI wrapper).

## Run locally

1. Create a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

2.
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn nexus_api:app --host 0.0.0.0 --port 8000 --reload

