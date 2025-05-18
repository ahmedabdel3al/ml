from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/vectorize")
def transformer(input: QueryInput):
    query_vector = model.encode(input.query).tolist()
    return {"query": input.query, "vectors": query_vector}
