
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch

# Constants
MODEL_NAME = "joelniklaus/legal-xlm-roberta-large"
DEVICE = 'cpu'
VECTOR_DIM = 1024  # dimension of embedding for legal-xlm-roberta-large

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# MongoDB Atlas connection
uri = "mongodb+srv://smaranrbbtech22:aimd0MyDfx25MmSdL@govai-xlm-r-v1.xx5wl1d.mongodb.net/?retryWrites=true&w=majority&appName=govai-xlm-r-v1"
client = MongoClient(uri)
db = client["govai-xlm-r-v2"]
collection = db["global_chunks"]

# Embedding utility
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(text: str):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = model(**encoded)
    embedding = mean_pooling(output, encoded["attention_mask"])
    return embedding[0].cpu().numpy()

def search_global_chunks(query: str, k: int = 10000, country: str = None, index_name: str = "global_vector_index"):
    print(f"🔎 Searching for: {query}")
    if country:
        print(f"🌍 Country filter: {country}")
    
    query_vector = embed_query(query)

    pipeline = []

    # Add vector search stage
    pipeline.append({
        "$vectorSearch": {
            "index": index_name,
            "queryVector": query_vector.tolist(),
            "path": "embedding",
            "numCandidates": 10000,
            "limit": k
        }
    })

    # Optional filter stage after vector search
    if country:
        pipeline.append({
            "$match": {
                "country": country
            }
        })

    # Project relevant fields
    pipeline.append({
        "$project": {
            "title": 1,
            "text": 1,
            "metadata": 1,
            "country": 1,
            "chunk_index": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    })

    results = list(collection.aggregate(pipeline))
    if not results:
        print("❌ No results found.")
    else:
        for doc in results:
            print(f"\n📄 Title: {doc['title']} | Country: {doc.get('country', 'Unknown')} [Chunk {doc['chunk_index']}]")
            print(f"🔢 Score: {doc['score']:.4f}")
            print(f"📜 Snippet: {doc['text'][:300]}...")
            print("—" * 60)
    
        print(results) 


# Search globally
# search_global_chunks("penalties for data misuse in general?")

# Search within UK only
search_global_chunks("AI policy regualtions", country="USA")
