import asyncio
import orjson
import uuid
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

client = AsyncQdrantClient(host="localhost", port=6333)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

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


def get_openai_embedding(text, model_name):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding


def get_jina_embedding(text, model_name):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "model": model_name,
        "normalized": True,
        "embedding_type": "float",
        "input": [text],
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


async def setup_collections():
    for model_key, model_info in MODELS.items():
        collection_name = f"{model_key}_posts"
        try:
            await client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model_info["dim"], distance=Distance.COSINE
            ),
        )
        print(f"Created collection: {collection_name} (dim: {model_info['dim']})")


async def feed_data(json_file_path: str):
    await setup_collections()

    # Load only local models
    local_models = {}
    for model_key, model_info in MODELS.items():
        if model_info["type"] == "local":
            print(f"Loading {model_info['name']}...")
            local_models[model_key] = SentenceTransformer(model_info["name"])

    all_points = {model_key: [] for model_key in MODELS.keys()}
    total_processed = 0

    with open(json_file_path, "rb") as f:
        for line_num, line in enumerate(f):
            try:
                data = orjson.loads(line)
                post_content = data["post"]
                topic = data["topic"]
                point_id = str(uuid.uuid4())

                payload = {
                    "post": post_content,
                    "topic": topic,
                    "line_number": line_num,
                }

                for model_key, model_info in MODELS.items():
                    print(f"Processing with {model_info['name']}... (line {line_num})")

                    if model_info["type"] == "local":
                        embedding = (
                            local_models[model_key].encode(post_content).tolist()
                        )
                    elif model_info["type"] == "openai_api":
                        embedding = get_openai_embedding(
                            post_content, model_info["name"]
                        )
                    elif model_info["type"] == "jina_api":
                        embedding = get_jina_embedding(post_content, model_info["name"])

                    all_points[model_key].append(
                        PointStruct(id=point_id, vector=embedding, payload=payload)
                    )

                total_processed += 1
                if total_processed % 50 == 0:
                    for model_key in MODELS.keys():
                        if all_points[model_key]:
                            await client.upsert(
                                collection_name=f"{model_key}_posts",
                                points=all_points[model_key],
                            )
                            all_points[model_key].clear()
                    print(f"Uploaded batch: {total_processed} posts processed...")

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Upload remaining points
    for model_key in MODELS.keys():
        if all_points[model_key]:
            await client.upsert(
                collection_name=f"{model_key}_posts", points=all_points[model_key]
            )

    print(
        f"✅ Successfully processed {total_processed} posts across {len(MODELS)} models"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest.py <json_file_path>")
        sys.exit(1)
    asyncio.run(feed_data(sys.argv[1]))
