from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time
from collections import Counter
try:
    import demjson3
except Exception:
    demjson3 = None
def limit_and_deduplicate_tags(tag_list, max_len=100):
    counter = Counter(tag_list)
    unique_sorted = [tag for tag, _ in counter.most_common()]
    return unique_sorted[:max_len]

def simple_tokenize(text):
    return word_tokenize(text)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(
        self,
        model: str = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1",
        )
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be either 'openai' or 'ollama'")

class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):
        
        self.content = content
        
        # Generate metadata using LLM if not provided and controller is available
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            # analysis = self.analyze_content(content, llm_controller)
            # print("analysis", analysis)
            # keywords = keywords or analysis["keywords"]
            # context = context or analysis["context"]
            # tags = tags or analysis["tags"]
            analysis = self.analyze_content(content, llm_controller) if llm_controller and any(param is None for param in [keywords, context, category, tags]) else {}
            if isinstance(analysis, dict):
                keywords = keywords or analysis.get("keywords", []) or []
                context = context or analysis.get("context", "") or ""
                tags = tags or analysis.get("tags", []) or []
            else:
                # 如果 analysis 不是 dict（例如字符串），退回默认值并继续
                print("⚠️ Warning: analysis returned non-dict, falling back to defaults.")
                keywords = keywords or []
                context = context or ""
                tags = tags or []
        
        # Set default values for optional parameters
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Handle context that can be either string or list
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)  # Convert list to string by joining
            
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content_old(content: str, llm_controller: LLMController) -> Dict:            
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = llm_controller.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            })
            
            try:
                analysis = json.loads(response)
            except:
                analysis = response
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }
    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:
        """Robustly analyze content and return dict {'keywords':[], 'context':'', 'tags':[]}."""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = llm_controller.llm.get_completion(prompt, response_format={
                "type": "json_schema", "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["keywords", "context", "tags"],
                        "additionalProperties": True
                    },
                    "strict": False
                }
            })
        except Exception as e:
            print(f"Error calling LLM for analyze_content: {e}")
            # 无安全 response 可用，直接返回默认
            return {"keywords": [], "context": "General", "tags": []}

        # 下面尝试把 response 解析为 dict，多个回退方案
        try:
            if isinstance(response, dict):
                analysis = response
            else:
                s = str(response).strip()
                # 去掉 markdown code fences
                import re
                s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
                # 尝试 json.loads
                try:
                    analysis = json.loads(s)
                except Exception:
                    # 尝试 demjson3（如果可用）
                    if demjson3 is not None:
                        try:
                            analysis = demjson3.decode(s)
                        except Exception:
                            analysis = {}
                    else:
                        # 尝试提取第一个 JSON 子串
                        m = re.search(r"\{.*\}", s, flags=re.S)
                        if m:
                            try:
                                analysis = json.loads(m.group(0))
                            except Exception:
                                analysis = {}
                        else:
                            analysis = {}
        except Exception as e:
            print(f"Unexpected parse error in analyze_content: {e}")
            analysis = {}

        # 规范化字段类型
        if not isinstance(analysis, dict):
            analysis = {}
        keywords = analysis.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        if not isinstance(keywords, list):
            keywords = []
        context = analysis.get("context", "") or ""
        if not isinstance(context, str):
            context = str(context)
        tags = analysis.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if not isinstance(tags, list):
            tags = []
        return {"keywords": keywords, "context": context, "tags": tags}
class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize the hybrid retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            alpha: Weight for combining BM25 and semantic scores (0 = only BM25, 1 = only semantic)
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
            
        # Save everything else using pickle
        state = {
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'all-MiniLM-L6-v2'  # Default value for model name
        }
        
        # Try to get the actual model name if possible
        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass
            
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        # Load the pickled state
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)
            
        # Create new instance
        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})
        
        # Load embeddings from numpy file if it exists
        if retriever_cache_embeddings_file.exists():
            retriever.embeddings = np.load(retriever_cache_embeddings_file)
            
        return retriever
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float) -> bool:
        """Load retriever state from memory"""
        all_docs = [", ".join(m.keywords) for m in memories.values()] #[m.content for m in memories.values()]
        retriever = cls(model_name, alpha)
        retriever.add_documents(all_docs)
        return retriever
    
    def add_documents(self, documents: List[str]) -> bool:
        """One-time Add documents to both BM25 and semantic index"""
        if not documents:
            return
            
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create embeddings
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        doc_idx = 0
        for document in documents:
            self.document_ids[document] = doc_idx
            doc_idx += 1

        return True

    def add_document(self, document: str) -> bool:
        """Add a single document to the retriever.
        
        Args:
            document: Text content to add
            
        Returns:
            bool: True if document was added, False if it was already present
        """
        # Check if document already exists
        if document in self.document_ids:
            return False
            
        # Add to corpus and get index
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # Update BM25
        if self.bm25 is None:
            # First document, initialize BM25
            tokenized_corpus = [simple_tokenize(document)]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            # Add to existing BM25
            tokenized_doc = simple_tokenize(document)
            self.bm25.add_document(tokenized_doc)
        
        # Update embeddings
        doc_embedding = self.model.encode([document], convert_to_tensor=True)
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = torch.cat([self.embeddings, doc_embedding])
            
        return True
        
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """Retrieve documents using hybrid scoring"""
        if not self.corpus:
            return []
            
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize BM25 scores if they exist
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Get semantic scores
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Combine scores
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
            
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # Create and initialize retriever
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval"""
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.memories = {}  # id -> MemoryNote
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url)

        # self.evolution_system_prompt = '''
        #                         You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
        #                         Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
        #                         Make decisions about its evolution.  
        #                         Please ONLY output a complete and valid JSON object exactly matching the schema, no extra text. Don't return more than 20000 tokens

        #                         The new memory context:
        #                         {context}
        #                         content: {content}
        #                         keywords: {keywords}

        #                         The nearest neighbors memories:
        #                         {nearest_neighbors_memories}

        #                         Based on this information, determine:
        #                         1. Should this memory be evolved? Consider its relationships with other memories.
        #                         2. What specific actions should be taken (strengthen, update_neighbor)?
        #                            2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
        #                            2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
        #                         Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
        #                         Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
        #                         The number of neighbors is {neighbor_number}.
        #                         Return your decision in JSON format with the following structure:
        #                         {{
        #                             "should_evolve": True or False,
        #                             "actions": ["strengthen", "update_neighbor"],
        #                             "suggested_connections": ["neighbor_memory_ids"],
        #                             "tags_to_update": ["tag_1",..."tag_n"], 
        #                             "new_context_neighborhood": ["new context",...,"new context"],
        #                             "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
        #                         }}
        #                         '''
        self.evolution_system_prompt = '''
        You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
        Analyze the new memory note according to keywords and context, along with its several nearest neighbor memories.
        Make decisions about its evolution.

        IMPORTANT: You MUST respond with ONLY a complete and valid JSON object exactly matching the schema below.
        Do NOT include any explanations, comments, or extra text before or after the JSON.
        If you cannot provide a valid JSON, respond exactly with the following JSON:
        {{
          "should_evolve": false,
          "actions": [],
          "suggested_connections": [],
          "tags_to_update": [],
          "new_context_neighborhood": [],
          "new_tags_neighborhood": []
        }}

        The new memory context:
        {context}
        content: {content}
        keywords: {keywords}

        The nearest neighbors memories:
        {nearest_neighbors_memories}

        Based on this information, determine:
        1. Should this memory be evolved? Consider its relationships with other memories.
        2. What specific actions should be taken (strengthen, update_neighbor)?
           2.1 If choose to strengthen the connection, which memory should it be connected to? Provide the updated tags of this memory.
           2.2 If choose to update_neighbor, update the context and tags of these memories based on the understanding of these memories.
               If the context and the tags are not updated, they should be the same as the originals.
               Generate the new context and tags in the sequential order of the input neighbors.
        Tags should be determined by the content characteristics of these memories, which can be used to retrieve and categorize them later.
        Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
        The number of neighbors is {neighbor_number}.
        
        Return your decision in JSON format with the following structure:

        {{
          "should_evolve": true or false,
          "actions": ["strengthen", "update_neighbor"],
          "suggested_connections": ["neighbor_memory_ids"],
          "tags_to_update": ["tag_1", ..., "tag_n"],
          "new_context_neighborhood": ["new context", ..., "new context"],
          "new_tags_neighborhood": [
            ["tag_1", ..., "tag_n"],
            ["tag_1", ..., "tag_n"],
            ["tag_1", ..., "tag_n"],
            ["tag_1", ..., "tag_n"],
            ["tag_1", ..., "tag_n"]
          ]
        }}
        
        Example of valid output:

        {{
          "should_evolve": true,
          "actions": ["strengthen"],
          "suggested_connections": [1, 3],
          "tags_to_update": ["tag1", "tag2"],
          "new_context_neighborhood": ["context1", "context2", "context3", "context4", "context5"],
          "new_tags_neighborhood": [
            ["tag1", "tag2"],
            ["tag3"],
            ["tag4", "tag5"],
            ["tag6"],
            ["tag7", "tag8"]
          ]
        }}
        Do not repeat tags in tags_to_update. List each tag only once.
        '''

        self.evo_cnt = 0 
        self.evo_threshold = evo_threshold

    def add_note_old(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        
        # Update retriever with all documents
        # all_docs = [m.content for m in self.memories.values()]
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([note.context + " keywords: " + ", ".join(note.keywords)])
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note with robust error handling."""
        try:
            note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        except Exception as e:
            print(f"⚠️ Error creating MemoryNote: {e}. Creating minimal note and continuing.")
            try:
                # fallback 不使用 llm_controller，使用最小元数据，保证不会抛出
                note = MemoryNote(content=content, llm_controller=None, timestamp=time, keywords=[], tags=[], context="General")
            except Exception as e2:
                print(f"❌ Fatal: failed to create fallback MemoryNote: {e2}")
                raise

        # process_memory 的异常不要抛出到上层，视为 no-evolution 并继续
        try:
            evo_label, note = self.process_memory(note)
        except Exception as e:
            print(f"⚠️ Error during process_memory: {e}. Skipping evolution for this note.")
            evo_label = False

        # 保存并尝试更新 retriever（失败时记录警告但继续）
        self.memories[note.id] = note
        try:
            self.retriever.add_documents([note.context + " keywords: " + ", ".join(note.keywords)])
        except Exception as e:
            print(f"⚠️ Warning: retriever.add_documents failed: {e}")

        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                try:
                    self.consolidate_memories()
                except Exception as e:
                    print(f"⚠️ consolidate_memories failed: {e}")
        return note.id
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents
        
        This function re-initializes the retriever and updates it with all memory documents,
        including their context, keywords, and tags to ensure the retrieval system has the
        latest state of all memories.
        """
        # Reset the retriever with the same model
        try:
            # Try to get model name through get_config_dict if available
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            # Fallback: use the model name from the class initialization
            model_name = 'all-MiniLM-L6-v2'
        
        self.retriever = SimpleEmbeddingRetriever(model_name)
        
        # Re-add all memory documents with their metadata
        for memory in self.memories.values():
            # Combine memory metadata into a single searchable document
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            # Add both the content and metadata as separate documents for better retrieval
            self.retriever.add_documents([memory.content + " , " + metadata_text])



    def process_memory(self, note: MemoryNote) -> bool:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
        )
        print("prompt_memory", prompt_memory)
    
        # 1) 安全调用 LLM（出错时不抛出，视为 no-evolution 并继续）
        try:
            response = self.llm_controller.llm.get_completion(
                prompt_memory,
                # max_tokens=4000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "integer"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                            "required": [
                                "should_evolve",
                                "actions",
                                "suggested_connections",
                                "tags_to_update",
                                "new_context_neighborhood",
                                "new_tags_neighborhood",
                            ],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
        except Exception as e:
            print(f"⚠️ LLM evolution call failed: {e}. Treating as no-evolution and continuing.")
            return False, note
    
        # 2) 解析 response（稳健回退，不依赖 demjson3 必须存在）
        import json, re, ast
        try:
            # 把 response 规范为字符串进行解析（LLM 返回 dict 也能处理）
            resp_text = response if isinstance(response, str) else str(response)
            # 去掉可能的 markdown code fence
            resp_text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", resp_text, flags=re.MULTILINE).strip()
    
            # 尝试标准 json.loads
            try:
                response_json = json.loads(resp_text)
            except Exception:
                # 若 demjson3 可用，尝试 demjson3.decode（更宽松）
                if demjson3 is not None:
                    try:
                        response_json = demjson3.decode(resp_text)
                    except Exception:
                        response_json = {}
                else:
                    # 尝试抽取第一个 JSON 子串再解析
                    m = re.search(r"\{.*\}", resp_text, flags=re.S)
                    if m:
                        try:
                            response_json = json.loads(m.group(0))
                        except Exception:
                            response_json = {}
                    else:
                        response_json = {}
        except Exception as e:
            print(f"⚠️ Failed to parse LLM response: {e}. Treating as no-evolution.")
            response_json = {}
    
        # 小修复：若解析结果不是 dict ，回退为空 dict
        if not isinstance(response_json, dict):
            response_json = {}
    
        # 3) 修复字符串化列表字段（保持原逻辑）
        def fix_stringified_lists(response_json: dict):
            for key in ["tags_to_update", "new_tags_neighborhood"]:
                if key in response_json:
                    new_list = []
                    for item in response_json[key]:
                        try:
                            if isinstance(item, str) and item.strip().startswith("["):
                                parsed = ast.literal_eval(item)
                                new_list.append(parsed)
                            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                                parsed = ast.literal_eval(item[0])
                                new_list.append(parsed)
                            else:
                                new_list.append(item)
                        except Exception as e:
                            print(f"⚠️ Failed to parse item in {key}: {item} → {e}")
                            new_list.append([])
                    response_json[key] = new_list
    
        try:
            fix_stringified_lists(response_json)
        except Exception as e:
            print(f"⚠️ fix_stringified_lists failed: {e}")
    
        # 4) 最低保证：若不能解析到 should_evolve，视为 False
        should_evolve = response_json.get("should_evolve", False)
    
        # 5) 如果需要演化，安全地执行 action（每步都包 try/except）
        if should_evolve:
            actions = response_json.get("actions", [])
            for action in actions:
                try:
                    if action == "strengthen":
                        suggest_connections = response_json.get("suggested_connections", [])
                        new_tags = response_json.get("tags_to_update", [])
                        # normalize and dedupe
                        def flatten_to_str_list(x):
                            if not isinstance(x, list):
                                return [str(x)]
                            flat = []
                            for item in x:
                                if isinstance(item, list):
                                    flat.extend([str(i) for i in item])
                                else:
                                    flat.append(str(item))
                            return flat
                        flat_tags = flatten_to_str_list(new_tags)
                        flat_tags = limit_and_deduplicate_tags(flat_tags, max_len=20)
                        note.links.extend(suggest_connections)
                        note.tags = flat_tags
                    elif action == "update_neighbor":
                        new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                        new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                        noteslist = list(self.memories.values())
                        notes_id = list(self.memories.keys())
                        for i in range(min(len(indices), len(new_tags_neighborhood))):
                            tag = new_tags_neighborhood[i]
                            if i < len(new_context_neighborhood):
                                context = new_context_neighborhood[i]
                            else:
                                # 防御式：如 indices[i] 越界则跳过
                                if indices[i] < len(noteslist):
                                    context = noteslist[indices[i]].context
                                else:
                                    context = ""
                            memorytmp_idx = indices[i]
                            try:
                                if memorytmp_idx < 0 or memorytmp_idx >= len(noteslist):
                                    print(f"⚠️ neighbor index {memorytmp_idx} out of range, skipping")
                                    continue
                                notetmp = noteslist[memorytmp_idx]
                                # 规范 tag 为字符串列表
                                if isinstance(tag, list):
                                    notetmp.tags = [str(x) for x in tag]
                                else:
                                    notetmp.tags = [str(tag)]
                                notetmp.context = context
                                self.memories[notes_id[memorytmp_idx]] = notetmp
                            except Exception as e:
                                print(f"⚠️ Failed to update neighbor memory index {memorytmp_idx}: {e}")
                except Exception as e:
                    print(f"⚠️ Error applying action {action}: {e}")
    
        return should_evolve, note
    def process_memory_old(self, note: MemoryNote) -> bool:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
        )
        print("prompt_memory", prompt_memory)

        # 1) 安全调用 LLM（出错时不抛出，视为 no-evolution 并继续）
        try:
            response = self.llm_controller.llm.get_completion(
                prompt_memory,
                # max_tokens=4000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "integer"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                            "required": [
                                "should_evolve",
                                "actions",
                                "suggested_connections",
                                "tags_to_update",
                                "new_context_neighborhood",
                                "new_tags_neighborhood",
                            ],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
        except Exception as e:
            print(f"⚠️ LLM evolution call failed: {e}. Treating as no-evolution and continuing.")
            return False, note

        # 2) 解析 response（稳健回退，不依赖 demjson3 必须存在）
        import json, re, ast
        try:
            # 把 response 规范为字符串进行解析（LLM 返回 dict 也能处理）
            resp_text = response if isinstance(response, str) else str(response)
            # 去掉可能的 markdown code fence
            resp_text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", resp_text, flags=re.MULTILINE).strip()

            # 尝试标准 json.loads
            try:
                response_json = json.loads(resp_text)
            except Exception:
                # 若 demjson3 可用，尝试 demjson3.decode（更宽松）
                if demjson3 is not None:
                    try:
                        response_json = demjson3.decode(resp_text)
                    except Exception:
                        response_json = {}
                else:
                    # 尝试抽取第一个 JSON 子串再解析
                    m = re.search(r"\{.*\}", resp_text, flags=re.S)
                    if m:
                        try:
                            response_json = json.loads(m.group(0))
                        except Exception:
                            response_json = {}
                    else:
                        response_json = {}
        except Exception as e:
            print(f"⚠️ Failed to parse LLM response: {e}. Treating as no-evolution.")
            response_json = {}

        # 小修复：若解析结果不是 dict ，回退为空 dict
        if not isinstance(response_json, dict):
            response_json = {}

        # 3) 修复字符串化列表字段（保持原逻辑）
        def fix_stringified_lists(response_json: dict):
            for key in ["tags_to_update", "new_tags_neighborhood"]:
                if key in response_json:
                    new_list = []
                    for item in response_json[key]:
                        try:
                            if isinstance(item, str) and item.strip().startswith("["):
                                parsed = ast.literal_eval(item)
                                new_list.append(parsed)
                            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                                parsed = ast.literal_eval(item[0])
                                new_list.append(parsed)
                            else:
                                new_list.append(item)
                        except Exception as e:
                            print(f"⚠️ Failed to parse item in {key}: {item} → {e}")
                            new_list.append([])
                    response_json[key] = new_list

        try:
            fix_stringified_lists(response_json)
        except Exception as e:
            print(f"⚠️ fix_stringified_lists failed: {e}")

        # 4) 最低保证：若不能解析到 should_evolve，视为 False
        should_evolve = response_json.get("should_evolve", False)

        # 5) 如果需要演化，安全地执行 action（每步都包 try/except）
        if should_evolve:
            actions = response_json.get("actions", [])
            for action in actions:
                try:
                    if action == "strengthen":
                        suggest_connections = response_json.get("suggested_connections", [])
                        new_tags = response_json.get("tags_to_update", [])
                        # normalize and dedupe
                        def flatten_to_str_list(x):
                            if not isinstance(x, list):
                                return [str(x)]
                            flat = []
                            for item in x:
                                if isinstance(item, list):
                                    flat.extend([str(i) for i in item])
                                else:
                                    flat.append(str(item))
                            return flat
                        flat_tags = flatten_to_str_list(new_tags)
                        flat_tags = limit_and_deduplicate_tags(flat_tags, max_len=20)
                        note.links.extend(suggest_connections)
                        note.tags = flat_tags
                    elif action == "update_neighbor":
                        new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                        new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                        noteslist = list(self.memories.values())
                        notes_id = list(self.memories.keys())
                        for i in range(min(len(indices), len(new_tags_neighborhood))):
                            tag = new_tags_neighborhood[i]
                            if i < len(new_context_neighborhood):
                                context = new_context_neighborhood[i]
                            else:
                                # 防御式：如 indices[i] 越界则跳过
                                if indices[i] < len(noteslist):
                                    context = noteslist[indices[i]].context
                                else:
                                    context = ""
                            memorytmp_idx = indices[i]
                            try:
                                if memorytmp_idx < 0 or memorytmp_idx >= len(noteslist):
                                    print(f"⚠️ neighbor index {memorytmp_idx} out of range, skipping")
                                    continue
                                notetmp = noteslist[memorytmp_idx]
                                # 规范 tag 为字符串列表
                                if isinstance(tag, list):
                                    notetmp.tags = [str(x) for x in tag]
                                else:
                                    notetmp.tags = [str(tag)]
                                notetmp.context = context
                                self.memories[notes_id[memorytmp_idx]] = notetmp
                            except Exception as e:
                                print(f"⚠️ Failed to update neighbor memory index {memorytmp_idx}: {e}")
                except Exception as e:
                    print(f"⚠️ Error applying action {action}: {e}")

        return should_evolve, note

    def process_memory_old(self, note: MemoryNote) -> bool:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
        )
        print("prompt_memory", prompt_memory)

        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            # max_tokens=4000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_evolve": {"type": "boolean"},
                            "actions": {"type": "array", "items": {"type": "string"}},
                            "suggested_connections": {"type": "array", "items": {"type": "integer"}},
                            "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                            "tags_to_update": {"type": "array", "items": {"type": "string"}},
                            "new_tags_neighborhood": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                        "required": [
                            "should_evolve",
                            "actions",
                            "suggested_connections",
                            "tags_to_update",
                            "new_context_neighborhood",
                            "new_tags_neighborhood",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )

        import json
        import demjson3
        import ast

        # def safe_parse_response(response: str):
        #     try:
        #         return json.loads(response)
        #     except json.JSONDecodeError as e:
        #         print("[JSONDecodeError] json.loads failed:", e)
        #         with open("bad_response.json", "w", encoding="utf-8") as f:
        #             f.write(response)
        #         try:
        #             response_json = demjson3.decode(response)
        #             print("[Recovered] using demjson3")
        #             return response_json
        #         except Exception as e2:
        #             print("❌ demjson3 also failed:", e2)
        #             raise RuntimeError("Failed to decode LLM response")


        import json
        import re
        import demjson3

        def safe_parse_response(response: str):
            """
            尝试更健壮地将LLM response解析为JSON：
            1. 清理markdown代码块头尾
            2. 尝试标准json
            3. 尝试demjson3
            4. 全部失败则保存bad_response并报错
            """
            # 1. strip markdown代码块和多余字符
            _raw = response
            response = response.strip()
            # 去除 markdown 代码块 ```json ... ```
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            # 也可用正则更强力去掉所有代码块
            response = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", response, flags=re.MULTILINE).strip()

            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                print("[JSONDecodeError] json.loads failed:", e)
                with open("bad_response.json", "w", encoding="utf-8") as f:
                    f.write(_raw)
                try:
                    response_json = demjson3.decode(response)
                    print("[Recovered] using demjson3")
                    return response_json
                except Exception as e2:
                    print("❌ demjson3 also failed:", e2)
                    # 保存demjson3失败后的内容
                    with open("bad_response_demjson3.json", "w", encoding="utf-8") as f:
                        f.write(_raw)
                    # 最终抛出
                    # raise RuntimeError("Failed to decode LLM response")
                    # return {
                    #     "should_evolve": None,
                    #     "actions": None,
                    #     "suggested_connections": None,
                    #     "tags_to_update": None,
                    #     "new_context_neighborhood": None,
                    #     "new_tags_neighborhood": None
                    # }
                    return {
                        "should_evolve": False,
                        "actions": [],
                        "suggested_connections": [],
                        "tags_to_update": [],
                        "new_context_neighborhood": [],
                        "new_tags_neighborhood": []
                    }


        def fix_stringified_lists(response_json: dict):
            for key in ["tags_to_update", "new_tags_neighborhood"]:
                if key in response_json:
                    new_list = []
                    for item in response_json[key]:
                        try:
                            if isinstance(item, str) and item.strip().startswith("["):
                                parsed = ast.literal_eval(item)
                                new_list.append(parsed)
                            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                                parsed = ast.literal_eval(item[0])
                                new_list.append(parsed)
                            else:
                                new_list.append(item)
                        except Exception as e:
                            print(f"⚠️ Failed to parse item in {key}: {item} → {e}")
                            new_list.append([])
                    response_json[key] = new_list

        def flatten_and_strlist(x):
            # 扁平化任意多层嵌套，全部转换为字符串列表
            if not isinstance(x, list):
                return [str(x)]
            flat = []
            for item in x:
                if isinstance(item, list):
                    flat.extend(flatten_and_strlist(item))
                else:
                    flat.append(str(item))
            return flat

        # Robust JSON parse + fix
        response_json = safe_parse_response(response)
        fix_stringified_lists(response_json)

        print("✅ Parsed response_json:", type(response_json))
        should_evolve = response_json.get("should_evolve", False)

        if should_evolve:
            actions = response_json.get("actions", [])
            for action in actions:
                # if action == "strengthen":
                #     suggest_connections = response_json.get("suggested_connections", [])
                #     new_tags = response_json.get("tags_to_update", [])
                #     note.links.extend(suggest_connections)
                #     note.tags = flatten_and_strlist(new_tags)
                if action == "strengthen":
                    suggest_connections = response_json.get("suggested_connections", [])
                    new_tags = response_json.get("tags_to_update", [])
                
                    new_tags = flatten_and_strlist(new_tags)
                    new_tags = limit_and_deduplicate_tags(new_tags, max_len=20)
                
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                    new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        notetmp.tags = flatten_and_strlist(tag)
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp

        return should_evolve, note


    def process_memory_dropped(self, note: MemoryNote) -> bool:
        """Process a memory note and return an evolution label"""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        print("prompt_memory", prompt_memory)
        response = self.llm_controller.llm.get_completion(
            prompt_memory,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean",
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
        )


        import json
        



        print("=== RAW response from LLM ===")
        print(response)

        import json
        import demjson3
        import ast

        # def safe_parse_response(response: str):
        #     try:
        #         return json.loads(response)
        #     except json.JSONDecodeError as e:
        #         print("[JSONDecodeError] json.loads failed:", e)
        #         with open("bad_response.json", "w", encoding="utf-8") as f:
        #             f.write(response)
        #         try:
        #             response_json = demjson3.decode(response)
        #             print("[Recovered] using demjson3")
        #             return response_json
        #         except Exception as e2:
        #             print("❌ demjson3 also failed:", e2)
        #             raise RuntimeError("Failed to decode LLM response")
        import json
        import re
        import demjson3

        def safe_parse_response(response: str):
            """
            尝试更健壮地将LLM response解析为JSON：
            1. 清理markdown代码块头尾
            2. 尝试标准json
            3. 尝试demjson3
            4. 全部失败则保存bad_response并报错
            """
            # 1. strip markdown代码块和多余字符
            _raw = response
            response = response.strip()
            # 去除 markdown 代码块 ```json ... ```
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            # 也可用正则更强力去掉所有代码块
            response = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", response, flags=re.MULTILINE).strip()

            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                print("[JSONDecodeError] json.loads failed:", e)
                with open("bad_response.json", "w", encoding="utf-8") as f:
                    f.write(_raw)
                try:
                    response_json = demjson3.decode(response)
                    print("[Recovered] using demjson3")
                    return response_json
                except Exception as e2:
                    print("❌ demjson3 also failed:", e2)
                    # 保存demjson3失败后的内容
                    # with open("bad_response_demjson3.json", "w", encoding="utf-8") as f:
                    #     f.write(_raw)
                    # # 最终抛出
                    # raise RuntimeError("Failed to decode LLM response")
                    return {
                        "should_evolve": False,
                        "actions": [],
                        "suggested_connections": [],
                        "tags_to_update": [],
                        "new_context_neighborhood": [],
                        "new_tags_neighborhood": []
                    }

        def fix_stringified_lists(response_json: dict):
            for key in ["tags_to_update", "new_tags_neighborhood"]:
                if key in response_json:
                    new_list = []
                    for item in response_json[key]:
                        try:
                            if isinstance(item, str) and item.strip().startswith("["):
                                parsed = ast.literal_eval(item)
                                new_list.append(parsed)
                            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                                parsed = ast.literal_eval(item[0])
                                new_list.append(parsed)
                            else:
                                new_list.append(item)
                        except Exception as e:
                            print(f"⚠️ Failed to parse item in {key}: {item} → {e}")
                            new_list.append([])
                    response_json[key] = new_list

        # === Robust JSON parse + repair ===
        response_json = safe_parse_response(response)
        fix_stringified_lists(response_json)

        # === Safe access ===
        print("✅ Parsed response_json:", type(response_json))
        should_evolve = response_json.get("should_evolve", False)

        if should_evolve:
            actions = response_json.get("actions", [])
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json.get("suggested_connections", [])
                    new_tags = response_json.get("tags_to_update", [])
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                    new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # notetmp.tags = tag
                        # notetmp.context = context
                        def flatten_and_strlist(x):
                            if isinstance(x, list):
                                flat = []
                                for item in x:
                                    if isinstance(item, list):
                                        flat.extend(item)
                                    else:
                                        flat.append(item)
                                return [str(i) for i in flat]
                            return [str(x)]
                        notetmp.tags = flatten_and_strlist(tag)
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        
        return should_evolve, note




    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "",[]
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        for i in indices:
            memory_str += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return []
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        for i in indices:
            memory_str +=  "talk start time:" + all_memories[i].timestamp + "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += "talk start time:" + all_memories[neighbor].timestamp + "memory content: " + all_memories[neighbor].content + "memory context: " + all_memories[neighbor].context + "memory keywords: " + str(all_memories[neighbor].keywords) + "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                if j >=k:
                    break
                j += 1
        return memory_str

def run_tests():
    """Run system tests"""
    print("Starting Memory System Tests...")
    
    # Initialize memory system with OpenAI backend
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model='/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft'
    )
    
    print("\nAdding test memories...")
    
    # Add test memories - only content is required
    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))
    
    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))
    
    print("\nQuerying for related memories...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )
    
    related = memory_system.find_related_memories(query.content, k=2)
    print("related", related)
    print("\nResults:")
    for i, memory in enumerate(related, 1):
        print(f"\n{i}. Memory:")
        print(f"Content: {memory.content}")
        print(f"Category: {memory.category}")
        print(f"Keywords: {memory.keywords}")
        print(f"Tags: {memory.tags}")
        print(f"Context: {memory.context}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
