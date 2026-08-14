from openai import OpenAI, AsyncOpenAI
import asyncio
import threading
from tqdm import tqdm
from . import config
from .token_tracker import record_failure, record_usage
import math
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import uuid

NO_THINKING_EXTRA_BODY = {
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}

_RUNTIME_ENDPOINTS = []
_RUNTIME_SEMAPHORES = []
_RUNTIME_COUNTER = None
_RUNTIME_LOCK = None
_LOCAL_COUNTER = 0
_LOCAL_LOCK = threading.Lock()
_EMBEDDING_CLIENTS = {}
_EMBEDDING_CLIENT_LOCK = threading.Lock()


def normalize_base_urls(base_url):
    """Normalize one or more OpenAI-compatible endpoints."""
    if isinstance(base_url, (list, tuple)):
        urls = [str(url).strip() for url in base_url if str(url).strip()]
    elif isinstance(base_url, str) and "," in base_url:
        urls = [url.strip() for url in base_url.split(",") if url.strip()]
    elif base_url:
        urls = [str(base_url).strip()]
    else:
        configured = os.getenv("ABLATION_LLM_BASE_URLS") or os.getenv(
            "ABLATION_LLM_BASE_URL"
        )
        if not configured:
            raise ValueError(
                "An LLM endpoint is required; pass base_url or set "
                "ABLATION_LLM_BASE_URLS"
            )
        urls = [url.strip() for url in configured.split(",") if url.strip()]
    return urls


def configure_llm_runtime(base_urls, per_endpoint_concurrency=16,
                          semaphores=None, counter=None, lock=None):
    """Configure process-local clients with optional process-shared limits."""
    global _RUNTIME_ENDPOINTS, _RUNTIME_SEMAPHORES
    global _RUNTIME_COUNTER, _RUNTIME_LOCK, _LOCAL_COUNTER

    _RUNTIME_ENDPOINTS = normalize_base_urls(base_urls)
    _RUNTIME_SEMAPHORES = list(semaphores) if semaphores is not None else [
        threading.BoundedSemaphore(per_endpoint_concurrency)
        for _ in _RUNTIME_ENDPOINTS
    ]
    _RUNTIME_COUNTER = counter
    _RUNTIME_LOCK = lock or _LOCAL_LOCK
    _LOCAL_COUNTER = 0


def _acquire_endpoint(base_urls):
    """Round-robin across endpoints and enforce their shared concurrency caps."""
    global _LOCAL_COUNTER

    endpoints = normalize_base_urls(base_urls)
    if endpoints != _RUNTIME_ENDPOINTS or len(_RUNTIME_SEMAPHORES) != len(endpoints):
        configure_llm_runtime(endpoints)

    with _RUNTIME_LOCK:
        if _RUNTIME_COUNTER is None:
            start_index = _LOCAL_COUNTER % len(endpoints)
            _LOCAL_COUNTER += 1
        else:
            start_index = int(_RUNTIME_COUNTER.value) % len(endpoints)
            _RUNTIME_COUNTER.value = int(_RUNTIME_COUNTER.value) + 1

    # Prefer an immediately available endpoint while retaining round-robin order.
    for offset in range(len(endpoints)):
        endpoint_index = (start_index + offset) % len(endpoints)
        semaphore = _RUNTIME_SEMAPHORES[endpoint_index]
        if semaphore.acquire(False):
            return endpoint_index, endpoints[endpoint_index]

    # All endpoints are at capacity; wait for the round-robin choice.
    semaphore = _RUNTIME_SEMAPHORES[start_index]
    semaphore.acquire()
    return start_index, endpoints[start_index]


def _release_endpoint(endpoint_index):
    _RUNTIME_SEMAPHORES[endpoint_index].release()


# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_urls = normalize_base_urls(base_url)
        if not _RUNTIME_ENDPOINTS:
            configure_llm_runtime(self.base_urls)
        self.clients = [
            OpenAI(api_key=self.api_key, base_url=url)
            for url in self.base_urls
        ]
        self.client = self.clients[0]

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000,
                        response_format=None, extra_body=None, stage="llm"):
        endpoint_index, endpoint = _acquire_endpoint(self.base_urls)
        print(f"Calling OpenAI API. Model: {model}; endpoint: {endpoint}")
        try:
            request_extra_body = dict(NO_THINKING_EXTRA_BODY)
            if extra_body:
                request_extra_body.update(extra_body)
            request_kwargs = {
                "model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens,
                "extra_body": request_extra_body,
            }
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            response = self.clients[endpoint_index].chat.completions.create(**request_kwargs)
            record_usage(response.usage, stage=stage, endpoint=endpoint, model=model)
            return response.choices[0].message.content.strip()
        except Exception as e:
            record_failure(stage=stage, endpoint=endpoint)
            print(f"Error calling OpenAI API via {endpoint}: {e}")
            return "Error: Could not get response from LLM."
        finally:
            _release_endpoint(endpoint_index)


class AsyncOpenAIClient:
    """Asynchronous OpenAI client."""
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_urls = normalize_base_urls(base_url)
        if not _RUNTIME_ENDPOINTS:
            configure_llm_runtime(self.base_urls)
        self.clients = [
            AsyncOpenAI(api_key=self.api_key, base_url=url)
            for url in self.base_urls
        ]
        self.client = self.clients[0]

    async def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000,
                              response_format=None, extra_body=None, stage="llm"):
        endpoint_index, endpoint = await asyncio.to_thread(
            _acquire_endpoint, self.base_urls
        )
        print(f"Calling OpenAI API (async). Model: {model}; endpoint: {endpoint}")
        try:
            request_extra_body = dict(NO_THINKING_EXTRA_BODY)
            if extra_body:
                request_extra_body.update(extra_body)
            request_kwargs = {
                "model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens,
                "extra_body": request_extra_body,
            }
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            response = await self.clients[endpoint_index].chat.completions.create(**request_kwargs)
            record_usage(response.usage, stage=stage, endpoint=endpoint, model=model)
            return response.choices[0].message.content.strip()
        except Exception as e:
            record_failure(stage=stage, endpoint=endpoint)
            print(f"Error calling OpenAI API via {endpoint}: {e}")
            return None
        finally:
            _release_endpoint(endpoint_index)

    async def close(self):
        await asyncio.gather(*(client.close() for client in self.clients))

def _get_embedding_client():
    base_url = config.globalconfig.embedding_base_url
    api_key = getattr(config.globalconfig, "embedding_api_key", "EMPTY")
    cache_key = (base_url, api_key)
    with _EMBEDDING_CLIENT_LOCK:
        client = _EMBEDDING_CLIENTS.get(cache_key)
        if client is None:
            client = OpenAI(api_key=api_key, base_url=base_url)
            _EMBEDDING_CLIENTS[cache_key] = client
    return client


def get_embedding(texts, batch=1):
    del batch  # The remote service batches all supplied texts in one request.
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    response = _get_embedding_client().embeddings.create(
        model=config.globalconfig.embedding_model_name,
        input=texts,
        extra_body={"truncate_prompt_tokens": 256},
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    embeddings = np.asarray(
        [item.embedding for item in ordered],
        dtype=np.float32,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings
    
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
    
def retrieve(query: str, mode=None, top_k=None):
    """
    Retrieve relevant segments and dialogues for a single query.
    """
    resolved_top_k = int(top_k or config.globalconfig.top_k_retrieve)
    query_embedding = get_embedding([query], config.globalconfig.embedding_batch_size)
    if not mode:
        relevent_contexts = search([query_embedding[0]],output_fields=["text", "type"], top_k=resolved_top_k)
        relevent_contexts = relevent_contexts[0]
        res = list(map(lambda x: {x["entity"]["type"]: x["entity"]["text"]}, relevent_contexts))
        return res

    if mode == "seg":
        """
        Retrieve all segments with flat retrieval.
        """
        relevent_contexts_segments = search([query_embedding[0]], output_fields=["text", "type"], top_k=resolved_top_k, filter="type == 'segment'")
        relevent_contexts_segments = relevent_contexts_segments[0]
        res = list(map(lambda x: {x["entity"]["type"]: x["entity"]["text"]}, relevent_contexts_segments))
        return res
    
    if mode == "dial":
        """
        Retrieve all dialogues with flat retrieval.
        """
        relevent_contexts_dialogues = search([query_embedding[0]], output_fields=["text", "type"], top_k=resolved_top_k, filter="type == 'dialogue'")
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
