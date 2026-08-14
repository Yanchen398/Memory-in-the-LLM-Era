import os
import json
import numpy as np
import faiss
import openai
import tiktoken
from tqdm import tqdm
from multiprocessing import Process

# ========================================
# Config
# ========================================
SHARD_GPU_MAP = {
    0: "0",
    1: "4",
    2: "5",
    3: "7"
}
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CORPUS_PATH = "corpus.jsonl"
INDEX_OUTPUT_DIR = "embeddings"
os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)

MAX_TOKENS_PER_BATCH = 8000
BATCH_SIZE = 50 
MAX_DOCS = None 
NUM_PROCESSES = 4  

tokenizer = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)


def encode(text):
    return tokenizer.encode(text, disallowed_special=())


client = openai.OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(texts):
    """Call OpenAI embedding API"""
    response = client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def process_shard(start: int, stride: int):
    shard_ids = []
    shard_texts = []

    gpu_id = SHARD_GPU_MAP[start]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[Shard {start}] Running on GPU {gpu_id}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if MAX_DOCS and i >= MAX_DOCS:
                break
            if i % stride != start:
                continue
            obj = json.loads(line)
            shard_ids.append(obj["docid"])
            shard_texts.append(obj["text"])

    print(f"[Shard {start}] Total docs: {len(shard_ids)}")

    # Batch embeddings
    all_embeddings = []
    batch_texts = []
    batch_ids = []

    for doc_id, doc_text in tqdm(zip(shard_ids, shard_texts), total=len(shard_ids), desc=f"Shard {start}"):
        # truncate
        tokens = encode(doc_text)
        if len(tokens) > MAX_TOKENS_PER_BATCH:
            tokens = tokens[:MAX_TOKENS_PER_BATCH]
            doc_text = tokenizer.decode(tokens)

        batch_texts.append(doc_text)
        batch_ids.append(doc_id)

        if len(batch_texts) >= BATCH_SIZE:
            embs = get_embeddings(batch_texts)
            all_embeddings.extend(embs)

        
            np.save(os.path.join(INDEX_OUTPUT_DIR, f"shard{start}_batch{batch_ids[0]}.npy"), np.array(embs, dtype="float32"))
            batch_texts, batch_ids = [], []

    if batch_texts:
        embs = get_embeddings(batch_texts)
        all_embeddings.extend(embs)
        np.save(os.path.join(INDEX_OUTPUT_DIR, f"shard{start}_batch{batch_ids[0]}.npy"), np.array(embs, dtype="float32"))


    if all_embeddings:
        emb_matrix = np.array(all_embeddings, dtype="float32")
        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb_matrix)
        faiss.write_index(index, os.path.join(INDEX_OUTPUT_DIR, f"shard{start}.index"))


        with open(os.path.join(INDEX_OUTPUT_DIR, f"shard{start}_id_map.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": shard_ids}, f, ensure_ascii=False, indent=2)

    print(f"[Shard {start}] Done!")


if __name__ == "__main__":
    processes = []
    for start in range(NUM_PROCESSES):
        p = Process(target=process_shard, args=(start, NUM_PROCESSES))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All shards processed. Index files are in:", INDEX_OUTPUT_DIR)
