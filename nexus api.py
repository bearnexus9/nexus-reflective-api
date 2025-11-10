# nexus_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from nexus_module import ConversationalNexus

app = FastAPI(title="Nexus Reflective API", version="1.0.0")
nexus = ConversationalNexus()

class Message(BaseModel):
    user_input: str

@app.post("/interact")
def interact(msg: Message):
    response, reflection = nexus.interact_and_learn(msg.user_input)
    return {"response": response, "reflection": reflection}

@app.get("/memory")
def memory_digest():
    data = nexus.top_memory()
    # present as list of [pattern, weight]
    return {"top_patterns": [[k, float(v)] for k, v in data]}
