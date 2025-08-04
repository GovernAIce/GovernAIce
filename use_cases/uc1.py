import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer


import google.generativeai as genai


# MongoDB
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# LangChain and Gemini

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MongoDB Configuration
MONGO_URI = "mongodb+srv://maxdefanti:vytsf38rBFxM1GQ6@governaice.q68lcpg.mongodb.net/?retryWrites=true&w=majority&appName=GovernAIce"  # Update with your MongoDB URI
 # Update with your collection name
DATABASE_NAME = "chunked_data"
COLLECTION_NAME = "chunked_data"
VECTOR_FIELD = "plot_embedding"
INDEX_NAME = "vector_index"  # Must match your MongoDB Atlas vector index name


client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
collection = client[DATABASE_NAME][COLLECTION_NAME]

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==== 2. Load Embedding Model ====
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# ==== 3. Define function ====
def search_and_answer(user_input, country=None):
    # Generate embedding
    query_vector = embedder.encode(user_input).tolist()

    # Build vector search stage with country filter if provided
    vector_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "filter": {"country" : {"$eq" :country}},
            "path": "plot_embedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 5
        }
    }

    # Run aggregation
    results = collection.aggregate([vector_stage])
    retrieved_chunks = [doc["text"] for doc in results]
    
    # Prepare RAG prompt
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are an intelligent assistant. Use the context below to answer the user's question as concisely and informatively as possible. Your job is to provide a clean and
    concise evaluation of the users proposed company / initiative based on the context you will be provided and, as a fallback, your background knowledge. The user will describe
    their company/initiative and you will provide a risk asessment based on the guidelines set out in your context. Provide brief summaries of risk areas and compliance gaps.
    Cite the names of the documents you are referencing with short direct quotes and provide analysis as to how they relate. Summarize relevant information only. Respond exclusively in english

    Context:
    {context}

    Question: {user_input}

    Answer:
    """

    # Use Gemini Pro to generate answer
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    print("help")
    return response.text

# ==== 4. Run example ====

question = "I want to do an ai startup that performs machine learning on peoples genetic data without them knowing"
country_filter = "Canada"
answer = search_and_answer(question, country_filter)
print("\nAnswer:\n", answer)
