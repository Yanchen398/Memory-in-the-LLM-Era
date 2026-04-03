import argparse
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
    model_name = getattr(config, "embedding_model_name", "/home/docker/Model/all-MiniLM-L6-v2")
    embedding_aliases = {
        "minilm": "/home/docker/Model/all-MiniLM-L6-v2",
        "all-minilm-l6-v2": "/home/docker/Model/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2": "/home/docker/Model/all-MiniLM-L6-v2",
        "bge-m3": "/home/docker/Model/bge-m3",
    }
    resolved_model_name = embedding_aliases.get(model_name, model_name)
    model = SentenceTransformer(resolved_model_name)
    return model

def create_collections(client, collection_name, dimension=384):
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

def load_config(config_path, args):
    import yaml
    from types import SimpleNamespace
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    f.close()
    
    # updata key
    for key, value in vars(args).items():
        if key not in config: 
            config[key] = value
        
    config = SimpleNamespace(**config)
    
    return config

class GlobalConfig:
    def __init__(self, config):
        # Dynamically bind all config key-value pairs to this instance.
        for key, value in vars(config).items():
            setattr(self, key, value)
           
        self.model= get_embedding_model(config)
        
        # Use absolute paths derived from the current output location.
        import os
        output_dir = os.path.dirname(config.output_path)
        data_dir = os.path.join(output_dir, "database")
        
        self.db_name = os.path.join(data_dir, f'{clean_str(config.embedding_model_name).replace(" ", "")}_{config.vdb_name}')
        
        if config.vdb_name != "milvus.db":
            self.client = MilvusClient(self.db_name)
            create_collections(self.client, config.collection_name, config.dimension)
        
        self.save_path = os.path.join(data_dir, config.save_name)


def initialize_global_config():
    """Initialize the global configuration."""
    parser = argparse.ArgumentParser(description='Path to config')
    parser.add_argument('--config_path', type=str,default="./config/memtree.yaml")
    args = parser.parse_args()
    config_yaml = load_config(config_path=args.config_path, args=args)
    return GlobalConfig(config_yaml)

# Initialize global configuration only when this file is run directly.
if __name__ == "__main__":
    globalconfig = initialize_global_config()
else:
    globalconfig = None        
