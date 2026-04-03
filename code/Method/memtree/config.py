import argparse
import os
from pymilvus import MilvusClient
import html
import re

def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def get_embedding_model(config):
    from sentence_transformers import SentenceTransformer
    model_name = getattr(config, "embedding_model_name", "/home/docker/Model/bge-m3")
    embedding_aliases = {
        "minilm": "/home/docker/Model/all-MiniLM-L6-v2",
        "all-minilm-l6-v2": "/home/docker/Model/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2": "/home/docker/Model/all-MiniLM-L6-v2",
        "bge-m3": "/home/docker/Model/bge-m3",
        "models--BAAI--bge-m3": "/home/docker/Model/bge-m3",
    }
    resolved_model_name = embedding_aliases.get(model_name, model_name)
    model = SentenceTransformer(resolved_model_name)

    # if config.embedding_model_name == "models--BAAI--bge-m3":
    #     model = SentenceTransformer("/home/yaodong/codes/GNNRAG/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
        
    # elif config.embedding_model_name == "nvidia/NV-Embed-v2":
    #     model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}, device="cuda")
    #     model.max_seq_length = 4096
    #     model.tokenizer.padding_side="right"
    # else:
    #     pass
    
    return model

def create_collections(client, collection_name, dimension=1024):
    # if client.has_collection(collection_name=collection_name):
    #     client.drop_collection(collection_name=collection_name)
    if client.has_collection(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists, using existing collection")
        return 

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )
    return
    
# def create_collections(client, collection_name, dimension=1024):
#     from pymilvus import MilvusClient, DataType
    
#     # 检查集合是否已存在
#     if client.has_collection(collection_name=collection_name):
#         print(f"Collection '{collection_name}' already exists, using existing collection")
#         return 
    
#     # 创建新集合
#     schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    
#     schema.add_field(
#         field_name="id",
#         datatype=DataType.VARCHAR,
#         is_primary=True,
#         auto_id=True,
#         max_length=100
#     )
    
#     schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    
#     index_params = MilvusClient.prepare_index_params()
#     index_params.add_index(
#         field_name="vector",
#         index_type="AUTOINDEX", 
#         metric_type="COSINE"
#     )
    
#     # 创建集合和索引
#     client.create_collection(
#         collection_name=collection_name,
#         dimension=dimension,
#         schema=schema,
#     )
#     client.create_index(collection_name=collection_name, index_params=index_params)
    
#     print(f"Created new collection '{collection_name}' with dimension {dimension}")
#     return 

def load_config(config_path, args):
    import yaml
    from types import SimpleNamespace
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update keys from CLI args when explicit values are provided.
    for key, value in vars(args).items():
        if value is not None and key != "config_path":
            config[key] = value

    config = resolve_memtree_config_dict(config, config_path)
        
    config = SimpleNamespace(**config)
    
    return config


def resolve_default_config_path():
    """Prefer the repo-level memtree config and fall back to the legacy internal config."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(current_dir))
    candidate_paths = [
        os.path.join(repo_root, "Config", "memtree.yaml"),
        os.path.join(current_dir, "config", "config.yaml"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return candidate_paths[0]


def resolve_memtree_config_dict(config, config_path=None):
    """Normalize memtree config values while keeping backward compatibility."""
    config = dict(config or {})

    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    def resolve_path(path_value):
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(base_dir, path_value))

    dataset_path = resolve_path(config.get("dataset_path"))
    if dataset_path:
        config["dataset_path"] = dataset_path

    if not config.get("dataset_name"):
        if dataset_path:
            config["dataset_name"] = os.path.splitext(os.path.basename(dataset_path))[0]
        else:
            raise ValueError("MemoryTree config requires either 'dataset_name' or 'dataset_path'.")

    for key in ("output_path", "token_file"):
        if config.get(key):
            config[key] = resolve_path(config.get(key))

    return config

class GlobalConfig:
    def __init__(self, config):
        config_values = resolve_memtree_config_dict(vars(config), getattr(config, "config_path", None))

        # Bind all config key-value pairs to this instance.
        for key, value in config_values.items():
            setattr(self, key, value)
           
        self.model= get_embedding_model(config)
        
        # Build the method-local storage directory from the dataset namespace.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data", self.dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        
        if not getattr(self, "dataset_path", None):
            self.dataset_path = os.path.join(data_dir, "data.json")

        self.db_name = os.path.join(data_dir, f'{clean_str(self.embedding_model_name).replace(" ", "")}_{self.vdb_name}')
        
        if self.vdb_name != "milvus.db":
            self.client = MilvusClient(self.db_name)
            create_collections(self.client, self.collection_name, self.dimension)
        
        self.save_path = os.path.join(data_dir, self.save_name)


def initialize_global_config():
    """Initialize the global configuration."""
    parser = argparse.ArgumentParser(description='Path to config')
    parser.add_argument('--config_path', type=str, default=resolve_default_config_path())
    args = parser.parse_args()
    config_yaml = load_config(config_path=args.config_path, args=args)
    return GlobalConfig(config_yaml)

# Initialize the global configuration only when this file is run directly.
if __name__ == "__main__":
    globalconfig = initialize_global_config()
else:
    # When imported as a module, try to initialize from the repo-level config first.
    try:
        default_config_path = resolve_default_config_path()
        default_args = argparse.Namespace(config_path=default_config_path)
        config_yaml = load_config(config_path=default_config_path, args=default_args)
        globalconfig = GlobalConfig(config_yaml)
    except Exception as e:
        print(f"Warning: Could not initialize globalconfig: {e}")
        globalconfig = None
        
        
        
