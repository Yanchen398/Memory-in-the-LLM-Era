from openai import OpenAI, AsyncOpenAI
import asyncio
from tqdm import tqdm
from . import config
import math
from openai import OpenAI
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import uuid

# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API. Model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."


class AsyncOpenAIClient:
    """Asynchronous OpenAI client."""
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API (async). Model: {model}")
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

def get_embedding(texts, batch=1):
    texts_embeddings = config.globalconfig.model.encode(texts, convert_to_tensor=True, show_progress_bar=True, device="cuda", batch_size=batch)
    texts_embeddings = texts_embeddings.cpu().numpy()
    
    if texts_embeddings.ndim == 1:
        texts_embeddings = texts_embeddings.reshape(1, -1)
    return texts_embeddings
    
def insert(data):
    config.globalconfig.client.insert(collection_name=config.globalconfig.collection_name, data=data)

def update_vector(new_data):
    config.globalconfig.client.upsert(
        collection_name=config.globalconfig.collection_name,
        data=new_data,
    )
    
def batch_insert(data, BATCH_SIZE):
    total = len(data)
    with tqdm(total=total, desc=f"Inserting into {config.globalconfig.collection_name}") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            config.globalconfig.client.insert(collection_name=config.globalconfig.collection_name, data=batch)
            pbar.update(len(batch))     
            
def search(query:list[list[float]], output_fields=None, top_k=None, filter=None):
    if filter:
        res = config.globalconfig.client.search(
            collection_name=config.globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )
    else:
        res = config.globalconfig.client.search(
            collection_name=config.globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
        )
    return res

def calculate_threshold(current_depth):
    threshold = config.globalconfig.base_threshold * math.exp(config.globalconfig.rate * current_depth / config.globalconfig.max_depth)
    return threshold

def calculate_cos(v, M):
    return cosine_similarity(v, M).flatten()
    
def retrieve(query: str, mode=None):
    """
    Retrieve relevant segments and dialogues for a single query.
    """
    query_embedding = get_embedding([query], config.globalconfig.embedding_batch_size)
    if not mode:
        relevent_contexts = search([query_embedding[0]],output_fields=["text", "type"], top_k=config.globalconfig.top_k_retrieve)
        relevent_contexts = relevent_contexts[0]
        res = list(map(lambda x: {x["entity"]["type"]: x["entity"]["text"]}, relevent_contexts))
        return res

    if mode == "seg":
        """
        Retrieve all segments with flat retrieval.
        """
        relevent_contexts_segments = search([query_embedding[0]], output_fields=["text", "type"], top_k=config.globalconfig.top_k_retrieve, filter="type == 'segment'")
        relevent_contexts_segments = relevent_contexts_segments[0]
        res = list(map(lambda x: {x["entity"]["type"]: x["entity"]["text"]}, relevent_contexts_segments))
        return res
    
    if mode == "dial":
        """
        Retrieve all dialogues with flat retrieval.
        """
        relevent_contexts_dialogues = search([query_embedding[0]], output_fields=["text", "type"], top_k=config.globalconfig.top_k_retrieve, filter="type == 'dialogue'")
        relevent_contexts_dialogues = relevent_contexts_dialogues[0]
        res = list(map(lambda x: {x["entity"]["type"]: x["entity"]["text"]}, relevent_contexts_dialogues))
        return res
    
    if mode == "beam":
        """
        Retrieve dialogues with beam search.
        """
        pass


def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    """Generate a unique ID: \"prefix + 8 char uuid4\" """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
