from tqdm import tqdm
from .config import globalconfig
import re
import math
import os
from collections import deque
from openai import OpenAI
from hashlib import md5
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Optional, Set, Union, Tuple
from .token_tracker import TokenTracker

def get_embedding(texts, batch=1):
    embedding_device = getattr(globalconfig, "embedding_device", "cpu")
    texts_embeddings = globalconfig.model.encode(texts, convert_to_tensor=True, show_progress_bar=True, device=embedding_device, batch_size=batch)
    texts_embeddings = texts_embeddings.cpu().numpy()
    
    if texts_embeddings.ndim == 1:
        texts_embeddings = texts_embeddings.reshape(1, -1)
    return texts_embeddings
    
def insert(data):
    globalconfig.client.insert(collection_name=globalconfig.collection_name, data=data)
    
def batch_insert(data, BATCH_SIZE):
    total = len(data)
    with tqdm(total=total, desc=f"Inserting into {globalconfig.collection_name}") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            globalconfig.client.insert(collection_name=globalconfig.collection_name, data=batch)
            pbar.update(len(batch))     
            
def search(query:list[list[float]], output_fields=None, top_k=None, filter=None):
    if filter:
        res = globalconfig.client.search(
            collection_name=globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )
    else:
        res = globalconfig.client.search(
            collection_name=globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
        )
    return res

def calculate_threshold(current_depth):
    threshold = globalconfig.base_threshold * math.exp(globalconfig.rate * current_depth / globalconfig.max_depth)
    return threshold

from sklearn.metrics.pairwise import cosine_similarity
def calculate_cos(v, M):
    return cosine_similarity(v, M).flatten()


from pathlib import Path
from tqdm import tqdm

# def worker_ollama(prompt):
#     """ 包装函数用于处理异常 """
#     # question_id, prompt = args[0], args[1]
#     try:
#         client = ollama.Client(host="http://localhost:5001/forward")
#         res = client.chat(
#             model="llama3.1:8b4k",
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": 0}
#         )
#         return res.message.content
#     except Exception as e:
#         print(f"Error processing {str(e)}")
#         return None
def worker_ollama(prompt):
    try:
        llm_base_url = getattr(globalconfig, "llm_base_url", "http://localhost:8000/v1")
        llm_api_key = getattr(globalconfig, "llm_api_key", "EMPTY")
        llm_model = getattr(globalconfig, "llm_model", "Qwen3.5-9B")
        client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        res = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None
worker_openai = worker_ollama
def update_vector(new_data):
    globalconfig.client.upsert(
        collection_name=globalconfig.collection_name,
        data=new_data,
    )
    
def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def get_question_text(query_item):
    if isinstance(query_item, dict):
        return str(query_item.get("question", ""))
    return str(query_item)
    
def retrieve_single(args: Tuple[str, List[np.ndarray], int, float]):
    query_id, query_emb, top_k, embedding_latency_ms = args
    retrieval_start = time.perf_counter()
    relevent_contexts = search(query_emb, top_k=top_k)
    search_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
    relevent_contexts = relevent_contexts[0]
    res = list(map(lambda x: x["id"], relevent_contexts))
    return query_id, res, embedding_latency_ms + search_latency_ms

def retrieve(query: list[str], i, token_file: str, top_k: Optional[int] = None):
    max_top_k = int(top_k or getattr(globalconfig, "top_k_retrieve", 10))
    query_texts = [get_question_text(item) for item in query]
    embedding_start = time.perf_counter()
    query_embeddings = get_embedding(query_texts, globalconfig.embedding_batch_size)
    embedding_latency_ms = (time.perf_counter() - embedding_start) * 1000.0
    per_query_embedding_latency_ms = embedding_latency_ms / len(query_texts) if query_texts else 0.0

    total_tasks = len(query)
    tasks = [(query[idx], [query_embeddings[idx]], max_top_k, per_query_embedding_latency_ms) for idx in range(total_tasks)]
    update = []
    for task in tasks:
        update.append(retrieve_single(task))
    return update
    
# import pickle
# def save_tree(tree, filename='memtree.pkl'):
#     with open(filename, 'wb') as f:
#         pickle.dump(tree, f)

# def load_tree(filename='memtree.pkl'):
#     if os.path.exists(filename):
#         with open(filename, 'rb') as f:
#             return pickle.load(f)
#     return None

from .prompt import ANSWER_PROMPT
def generation(tree, retrieve_results, i, token_file: str):
    def generate_one(index_and_item):
        qa_idx, item = index_and_item
        if len(item) == 3:
            que, contexts_id, retrieval_latency_ms = item
        else:
            que, contexts_id = item
            retrieval_latency_ms = 0.0
        contexts = list(map(lambda x: tree.nodes[x].cv, contexts_id))
        contexts = "\n\n".join(contexts)
        query_text = get_question_text(que)
        prompt = ANSWER_PROMPT.format(query=query_text, retrieved_content=contexts)
        output = worker_ollama(prompt)
        return qa_idx, (que, contexts, output, retrieval_latency_ms)

    indexed_items = list(enumerate(retrieve_results))
    if not indexed_items:
        return []

    max_workers = int(getattr(globalconfig, "answer_parallel_nums", getattr(globalconfig, "llm_parallel_nums", 1)) or 1)
    max_workers = max(1, min(max_workers, len(indexed_items)))
    results = [None] * len(indexed_items)

    if max_workers == 1:
        for indexed_item in indexed_items:
            qa_idx, result = generate_one(indexed_item)
            results[qa_idx] = result
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_one, indexed_item) for indexed_item in indexed_items]
            for future in as_completed(futures):
                qa_idx, result = future.result()
                results[qa_idx] = result

    return [result for result in results if result is not None]


        
        



        
    
