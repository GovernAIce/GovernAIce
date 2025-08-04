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

    prompt = f"""
    You are an intelligent assistant. Use the context below to answer the user's question as concisely and informatively as possible. Your job is to provide a clean and
    concise evaluation of the users proposed company / initiative with respect to the following frameworks. The user will describe
    their company/initiative and you will provide a framework adhesion score according to the following frameworks. Score on each subpoint and then return an overall score out of 100 after each section give a brief asessment as to why
    
    return your answers in this format: OECD Values-based Principles: overall: 80, Inclusivity and sustainability: 70, etc. NIST AI Lifecycle Framework: Overall: 72, Operate and Monitor: 90, etc. For EU risk asessment a full score means that the risk mitigation is >= to the risk or there is no risk score of 0 means that there is no risk mitigation and exclusively risk

    OECD Values-based Principles. components: inclusive growth, sustainable development, and well-being; human-centered values and fairness; transparency and explainability; robustness, security, and safety; and accountability

    NIST AI Risk Management Framework. components: Govern, Map, Measure, and Manage

    EU AI Risk Act. components: Unacceptable Risk Prevention vs Unacceptable Risk, High Risk Prevention vs High Risk, Limited Risk Prevention vs Limited Risk, Minimal Risk Prevention vs Minimal Risk

    Question: {user_input}

    Answer:
    """

    # Use Gemini Pro to generate answer
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    print("help")
    return response.text

# ==== 4. Run example ====

question = "I want to do an ai startup that performs machine learning on peoples genetic data with their consent"
country_filter = "Canada"
answer = search_and_answer(question, country_filter)
print("\nAnswer:\n", answer)
