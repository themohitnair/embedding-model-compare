from sentence_transformers import SentenceTransformer, util
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "color": "lightblue",
        "type": "local",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "color": "lightcoral",
        "type": "local",
    },
    "e5_large": {
        "name": "intfloat/e5-large-v2",
        "dim": 1024,
        "color": "lightgreen",
        "type": "local",
    },
    "e5_base": {
        "name": "intfloat/e5-base-v2",
        "dim": 768,
        "color": "lightyellow",
        "type": "local",
    },
    "bge_large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "color": "lightpink",
        "type": "local",
    },
    "bge_base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "color": "lightgray",
        "type": "local",
    },
    "gtr_t5": {
        "name": "sentence-transformers/gtr-t5-large",
        "dim": 768,
        "color": "lightsalmon",
        "type": "local",
    },
    "jina_v3": {
        "name": "jina-embeddings-v3",
        "dim": 1024,
        "color": "mediumpurple",
        "type": "jina_api",
    },
    "openai_large": {
        "name": "text-embedding-3-large",
        "dim": 3072,
        "color": "darkseagreen",
        "type": "openai_api",
    },
}

# Target description
desc = "breakfast"

# Topics list
topics = [
    "black coffee",
    "coffee brewing", 
    "espresso",
    "coffee beans",
    "coffee aroma",
    "caffeine",
    "coffee culture",
    "coffee preparation",
    "coffee enthusiasts",
    "coffee health benefits",
    "coffee taste",
    "coffee varieties",
    "morning coffee",
    "coffee shop",
    "coffee rituals"
]

def get_openai_embeddings(texts, model_name="text-embedding-3-large"):
    """Get embeddings from OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": texts,
        "model": model_name
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/embeddings", 
                               headers=headers, json=data)
        response.raise_for_status()
        embeddings = [item['embedding'] for item in response.json()['data']]
        return np.array(embeddings)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def get_jina_embeddings(texts, model_name="jina-embeddings-v3"):
    """Get embeddings from Jina API"""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("Jina API key not found. Set JINA_API_KEY environment variable.")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": texts,
        "model": model_name
    }
    
    try:
        response = requests.post("https://api.jina.ai/v1/embeddings", 
                               headers=headers, json=data)
        response.raise_for_status()
        embeddings = [item['embedding'] for item in response.json()['data']]
        return np.array(embeddings)
    except Exception as e:
        print(f"Jina API error: {e}")
        return None

def process_local_model(model_name):
    """Process local model using your exact method"""
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Target description
    desc_embedding = model.encode(desc)
    
    # Calculate and print similarity
    print(f"Similarity of '{desc}' with each topic:")
    for topic in topics:
        topic_embedding = model.encode(topic)
        similarity = util.cos_sim(desc_embedding, topic_embedding).item()
        print(f"{topic:25s} -> Similarity: {similarity:.4f}")

def process_api_model(embeddings_func, model_name):
    """Process API model using your method structure"""
    print(f"Getting embeddings from {model_name}...")
    
    # Get embedding for description
    desc_embedding = embeddings_func([desc], model_name)
    if desc_embedding is None:
        return
    desc_embedding = desc_embedding[0]
    
    # Calculate and print similarity
    print(f"Similarity of '{desc}' with each topic:")
    for topic in topics:
        topic_embedding = embeddings_func([topic], model_name)
        if topic_embedding is None:
            continue
        topic_embedding = topic_embedding[0]
        
        # Calculate cosine similarity manually (same as util.cos_sim)
        similarity = np.dot(desc_embedding, topic_embedding) / (
            np.linalg.norm(desc_embedding) * np.linalg.norm(topic_embedding)
        )
        print(f"{topic:25s} -> Similarity: {similarity:.4f}")

# Main execution
for model_key, model_info in MODELS.items():
    print("=" * 60)
    print(f"Model: {model_key} ({model_info['name']})")
    print("=" * 60)
    
    try:
        if model_info['type'] == 'local':
            process_local_model(model_info['name'])
        elif model_info['type'] == 'openai_api':
            process_api_model(get_openai_embeddings, model_info['name'])
        elif model_info['type'] == 'jina_api':
            process_api_model(get_jina_embeddings, model_info['name'])
    except Exception as e:
        print(f"Error processing {model_key}: {e}")
    
    print()  # Empty line for spacing

print("Done!")
