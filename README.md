# Nexus Reflective API

A minimal self-evolving language memory and reflection engine (FastAPI wrapper).

## Run locally

1. Create a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

2. Install dependencies:
pip install -r requirements.txt

3. Install the spaCy medium model (recommended for semantic linking):
python -m spacy download en_core_web_md

Run the app:
uvicorn nexus_api:app --host 0.0.0.0 --port 8000 --reload

