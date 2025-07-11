import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db     = client[os.getenv("DB_NAME")]
items  = db["items"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/items")
async def list_items():
    return list(items.find({}, {"_id": 0}))

@app.post("/items")
async def create_item(item: dict):
    res = items.insert_one(item)
    if not res.inserted_id:
        raise HTTPException(500, "Insert failed")
    return {"status": "ok"}
