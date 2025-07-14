# MongoDB Text Data Graph Visualization with Gemini API and LangChain
# This notebook demonstrates how to extract text data from MongoDB,
# process it using Gemini API through LangChain, and create visual graph representations

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# MongoDB
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# LangChain and Gemini
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Graph and Network Analysis
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# NLP and Text Processing
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")

## Configuration Section
# Set your API keys and database connection details here

# MongoDB Configuration
MONGO_URI = "mongodb+srv://s25intern:E623PjD5pGBQC8uA@governaice.q68lcpg.mongodb.net/?retryWrites=true&w=majority&appName=GovernAIce"  # Update with your MongoDB URI
DATABASE_NAME = "Training"           # Update with your database name
COLLECTION_NAME = "USA"       # Update with your collection name

# Google Gemini API Configuration
GOOGLE_API_KEY = "AIzaSyA4cWfJTyHNvTUxYTlbdulx18u2ymFXi80"    # Update with your Google API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Text field configuration
TEXT_FIELD = "text"  # Update with the field name containing your text data
ID_FIELD = "_id"        # Usually "_id" for MongoDB documents

print("Configuration loaded. Please update the values above with your actual credentials.")

## Database Connection and Data Extraction

def extract_text_data(collection, text_field: str, id_field: str = "_id", limit: int = 1000):
    """Extract text data from MongoDB collection"""
    try:
        # Query to get documents with non-empty text fields
        query = {text_field: {"$exists": True, "$ne": "", "$ne": None}}
        
        documents = list(collection.find(query).limit(limit))
        
        if not documents:
            print(f"No documents found with field '{text_field}'")
            return []
        
        # Extract text and IDs
        extracted_data = []
        for doc in documents:
            if text_field in doc and doc[text_field]:
                extracted_data.append({
                    'id': str(doc[id_field]),
                    'text': str(doc[text_field]),
                    'original_doc': doc
                })
        
        print(f"Extracted {len(extracted_data)} documents with text data.")
        return extracted_data
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        return []

# Connect to MongoDB and extract data
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
database = client["Training"]
collection = database["USA"]

if collection is not None:
    text_data = extract_text_data(collection, TEXT_FIELD, ID_FIELD)
    if text_data:
        print(f"\nSample document:")
        print(f"ID: {text_data[0]['id']}")
        print(f"Text preview: {text_data[0]['text'][:200]}...")
else:
    # Sample data for demonstration if MongoDB connection fails
    text_data = [
        {'id': '1', 'text': 'This is a sample document about machine learning and artificial intelligence.'},
        {'id': '2', 'text': 'Natural language processing is a key component of AI systems.'},
        {'id': '3', 'text': 'Graph neural networks are revolutionizing how we process structured data.'},
        {'id': '4', 'text': 'Deep learning models require large amounts of training data.'},
        {'id': '5', 'text': 'Computer vision and NLP are two major areas of AI research.'}
    ]
    print("Using sample data for demonstration.")

## LangChain and Gemini API Setup

def initialize_gemini_llm():
    """Initialize the Gemini LLM through LangChain"""
    try:
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        print("Gemini LLM initialized successfully!")
        return llm
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        return None

def create_entity_extraction_chain(llm):
    """Create a chain for extracting entities and relationships from text"""
    template = """
    Analyze the following text and extract:
    1. Key entities (people, organizations, concepts, locations, etc.)
    2. Relationships between entities
    3. Main topics or themes
    
    Text: {text}
    
    Please format your response as JSON with the following structure:
    {{
        "entities": ["entity1", "entity2", ...],
        "relationships": [
            {{"source": "entity1", "target": "entity2", "relationship": "description"}},
            ...
        ],
        "topics": ["topic1", "topic2", ...]
    }}
    
    Response:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def create_similarity_analysis_chain(llm):
    """Create a chain for analyzing text similarity and themes"""
    template = """
    Analyze the following two texts and determine:
    1. Their similarity score (0-1)
    2. Common themes or topics
    3. Key differences
    
    Text 1: {text1}
    Text 2: {text2}
    
    Please format your response as JSON:
    {{
        "similarity_score": 0.0-1.0,
        "common_themes": ["theme1", "theme2", ...],
        "differences": ["difference1", "difference2", ...]
    }}
    
    Response:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text1", "text2"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Initialize Gemini LLM
llm = initialize_gemini_llm()

if llm:
    entity_chain = create_entity_extraction_chain(llm)
    similarity_chain = create_similarity_analysis_chain(llm)
    print("LangChain pipelines created successfully!")

## Text Processing and Analysis

def extract_entities_and_relationships(text_data: List[Dict], entity_chain):
    """Extract entities and relationships from text data using Gemini"""
    processed_data = []
    
    for i, item in enumerate(text_data):
        try:
            print(f"Processing document {i+1}/{len(text_data)}: {item['id']}")
            
            # Use Gemini to extract entities and relationships
            response = entity_chain.run(text=item['text'])
            
            # Parse JSON response
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple structure
                parsed_response = {
                    "entities": [],
                    "relationships": [],
                    "topics": []
                }
            
            processed_data.append({
                'id': item['id'],
                'text': item['text'],
                'entities': parsed_response.get('entities', []),
                'relationships': parsed_response.get('relationships', []),
                'topics': parsed_response.get('topics', []),
                'original_doc': item.get('original_doc')
            })
            
        except Exception as e:
            print(f"Error processing document {item['id']}: {e}")
            # Add empty structure for failed processing
            processed_data.append({
                'id': item['id'],
                'text': item['text'],
                'entities': [],
                'relationships': [],
                'topics': [],
                'original_doc': item.get('original_doc')
            })
    
    return processed_data

def calculate_text_similarity(text_data: List[Dict]):
    """Calculate similarity between documents using TF-IDF"""
    texts = [item['text'] for item in text_data]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, vectorizer

# Process text data
if llm:
    processed_data = extract_entities_and_relationships(text_data, entity_chain)
else:
    # Fallback processing without Gemini
    processed_data = []
    for item in text_data:
        # Simple entity extraction using basic NLP
        words = re.findall(r'\b[A-Z][a-z]+\b', item['text'])
        processed_data.append({
            'id': item['id'],
            'text': item['text'],
            'entities': words[:5],  # Take first 5 capitalized words as entities
            'relationships': [],
            'topics': [],
            'original_doc': item.get('original_doc')
        })

# Calculate similarity matrix
similarity_matrix, vectorizer = calculate_text_similarity(text_data)

print(f"Processed {len(processed_data)} documents.")
print(f"Similarity matrix shape: {similarity_matrix.shape}")

## Graph Construction

def build_document_similarity_graph(text_data: List[Dict], similarity_matrix: np.ndarray, threshold: float = 0.3):
    """Build a graph based on document similarity"""
    G = nx.Graph()
    
    # Add nodes (documents)
    for i, item in enumerate(text_data):
        G.add_node(item['id'], 
                  text=item['text'][:100] + "..." if len(item['text']) > 100 else item['text'],
                  full_text=item['text'])
    
    # Add edges based on similarity
    for i in range(len(text_data)):
        for j in range(i + 1, len(text_data)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(text_data[i]['id'], text_data[j]['id'], 
                          weight=similarity,
                          similarity=similarity)
    
    return G

def build_entity_graph(processed_data: List[Dict]):
    """Build a graph based on entities and their relationships"""
    G = nx.Graph()
    
    # Add entity nodes and document-entity relationships
    for doc in processed_data:
        doc_id = doc['id']
        
        # Add document node
        G.add_node(f"doc_{doc_id}", 
                  type='document',
                  text=doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text'],
                  full_text=doc['text'])
        
        # Add entity nodes and connect to document
        for entity in doc['entities']:
            entity_clean = entity.strip().lower()
            if entity_clean:
                G.add_node(f"entity_{entity_clean}", 
                          type='entity',
                          name=entity)
                G.add_edge(f"doc_{doc_id}", f"entity_{entity_clean}", 
                          type='contains')
        
        # Add relationship edges
        for rel in doc['relationships']:
            if 'source' in rel and 'target' in rel:
                source = f"entity_{rel['source'].strip().lower()}"
                target = f"entity_{rel['target'].strip().lower()}"
                if source in G and target in G:
                    G.add_edge(source, target, 
                              type='relationship',
                              relationship=rel.get('relationship', ''))
    
    return G

def build_topic_graph(processed_data: List[Dict]):
    """Build a graph based on topics"""
    G = nx.Graph()
    
    # Add document and topic nodes
    for doc in processed_data:
        doc_id = doc['id']
        
        # Add document node
        G.add_node(f"doc_{doc_id}", 
                  type='document',
                  text=doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text'])
        
        # Add topic nodes and connect to document
        for topic in doc['topics']:
            topic_clean = topic.strip().lower()
            if topic_clean:
                G.add_node(f"topic_{topic_clean}", 
                          type='topic',
                          name=topic)
                G.add_edge(f"doc_{doc_id}", f"topic_{topic_clean}", 
                          type='discusses')
    
    return G

# Build different types of graphs
similarity_graph = build_document_similarity_graph(text_data, similarity_matrix)
#entity_graph = build_entity_graph(processed_data)
#topic_graph = build_topic_graph(processed_data)

# print(f"Similarity graph: {similarity_graph.number_of_nodes()} nodes, {similarity_graph.number_of_edges()} edges")
#print(f"Entity graph: {entity_graph.number_of_nodes()} nodes, {entity_graph.number_of_edges()} edges")
# print(f"Topic graph: {topic_graph.number_of_nodes()} nodes, {topic_graph.number_of_edges()} edges")

## Graph Visualization Functions
