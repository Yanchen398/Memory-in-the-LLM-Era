"""
Unified runner with config-file support and CLI overrides.

Supported commands:
1. memtree - Run the MemoryTree method
   Basic usage (config file required):
   python run.py memtree --config_file <config.yaml>

   Override values from the config file:
   python run.py memtree --config_file <config.yaml> --dataset_name <name> --output_path <path>

2. memoryos - Run the MemoryOS method
   Basic usage (config file required):
   python run.py memoryos --config_file <config.yaml>

   Override values from the config file:
   python run.py memoryos --config_file <config.yaml> --dataset_path <path> --output_path <path>

3. amem - Run the Agentic Memory method
   Config-file mode:
   python run.py amem --config_file <config.yaml>

   Mixed mode (use a config file and override selected values):
   python run.py amem --config_file <config.yaml> --dataset_path <path> --output_path <path>

4. zep - Run the Zep (Graphiti) method
   Basic usage (config file required):
   python run.py zep --config_file <config.yaml>

   Override values from the config file:
   python run.py zep --config_file <config.yaml> --dataset_path <path> --output_path <path>

5. memochat - Run the MemoChat method
   Basic usage (config file required):
   python run.py memochat --config_file <config.yaml>

   Override values from the config file:
   python run.py memochat --config_file <config.yaml> --input_data <path> --results_output_path <path>

6. memos - Run the Memos method
   Basic usage (config file required):
   python run.py memos --config_file <config.yaml>

   Override values from the config file:
   python run.py memos --config_file <config.yaml> --version <version> --num_workers <workers> --top_k <k> --dataset_path <path>

7. sota - Run the SOTA method
   Basic usage (config file required):
   python run.py sota --config_file <config.yaml>

   Override values from the config file:
   python run.py sota --config_file <config.yaml> --dataset_path <path> --memory_path <path> --output_path <path>

Note: `memtree`, `memoryos`, `zep`, `memochat`, `memos`, and `sota` now require a config file and no longer support pure CLI-only mode.
"""

import argparse
import sys
import os

# Add the project root to the Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import utility helpers and the configuration manager.
from utils import ConfigManager


def str_to_bool(value):
    """Parse common string forms into booleans for CLI overrides."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def run_memory_tree(config_manager: ConfigManager):
    """Run the MemoryTree method."""
    print("Running MemoryTree...")
    
    try:
        print("Importing the MemoryTree module...")
        from Method.memtree import run_memtree
        print("MemoryTree module imported successfully.")
        
        config = config_manager.get_all()
        
        # Read values directly from the config file.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")
        
        # Run MemoryTree.
        result = run_memtree(
            config_path=config_path,
            dataset_name=config.get('dataset_name'),
            dataset_path=config.get('dataset_path'),
            output_path=config.get('output_path'),
            token_file=config.get('token_file'),
            batch_size=config.get('batch_size'),
            num_processes=config.get('num_processes'),
        )
        
        return result
        
    except ImportError as e:
        print(f"Error: failed to import the MemoryTree module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemoryTree raised an exception: {e}")
        return False
    

def run_zep(config_manager: 'ConfigManager'):
    """Run the Zep (Graphiti) method."""
    print("Running Zep (Graphiti)...")
    
    try:
        print("Importing the Zep module...")
        from Method.zep import run_zep
        print("Zep module imported successfully.")
        
        config = config_manager.get_all()
        
        # Read values directly from the config file.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")
        
        # Run Zep.
        import asyncio
        result = asyncio.run(run_zep(
            dataset_path=config.get('dataset_path'),
            output_path=config.get('output_path'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model'),
            llm_small_model=config.get('llm_small_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
            embedding_api_key=config.get('embedding_api_key'),
            embedding_base_url=config.get('embedding_base_url'),
            embedding_dim=config.get('embedding_dim'),
            neo4j_uri=config.get('neo4j_uri'),
            neo4j_user=config.get('neo4j_user'),
            neo4j_password=config.get('neo4j_password'),
            answer_model=config.get('answer_model'),
            answer_api_key=config.get('answer_api_key'),
            answer_base_url=config.get('answer_base_url'),
            answer_temperature=config.get('answer_temperature'),
            answer_max_tokens=config.get('answer_max_tokens'),
        ))
        
        return result
        
    except ImportError as e:
        print(f"Error: failed to import the Zep module: {e}")
        return False
    except Exception as e:
        print(f"Error: Zep raised an exception: {e}")
        return False

def run_memoryos(config_manager: ConfigManager):
    """Run the MemoryOS method."""
    print("Running MemoryOS...")
    
    try:
        print("Importing the MemoryOS module...")
        from Method.memoryos import run_memoryos
        print("MemoryOS module imported successfully.")
        
        config = config_manager.get_all()
        
        # Read values directly from the config file.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")
        
        # Run MemoryOS.
        result = run_memoryos(
            dataset_path=config.get('dataset_path'),
            output_path=config.get('output_path'),
            memory_path=config.get('memory_path'),
            llm_model=config.get('llm_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
            token_file=config.get('token_file'),
        )
        
        return result
        
    except ImportError as e:
        print(f"Error: failed to import the MemoryOS module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemoryOS raised an exception: {e}")
        return False

def run_memochat(config_manager: ConfigManager):
    """Run the MemoChat method."""
    print("Running MemoChat...")
    
    try:
        print("Importing the MemoChat module...")
        from Method.memochat import run_memochat
        print("MemoChat module imported successfully.")
        
        config = config_manager.get_all()
        
        # Read values from the config file, with CLI overrides applied by ConfigManager.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")

        # Run MemoChat.
        results, memory = run_memochat(
            input_data=config.get('input_data'),
            results_output_path=config.get('results_output_path'),
            memory_output_path=config.get('memory_output_path'),
            prompt_path=config.get('prompt_path'),
            openai_modelid=config.get('openai_modelid'),
            base_url=config.get('base_url'),
            api_key=config.get('api_key'),
            token_file=config.get('token_file')
        )
        
        return results
        
    except ImportError as e:
        print(f"Error: failed to import the MemoChat module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemoChat raised an exception: {e}")
        return False


def run_memos(config_manager: ConfigManager):
    """Run the Memos method."""
    print("Running Memos...")
    
    try:
        print("Importing the Memos module...")
        from Method.memos.main import run_memos
        print("Memos module imported successfully.")
        
        config = config_manager.get_all()
        
        # Read values directly from the config file.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")

        # Run Memos.
        result = run_memos(config=config, config_path=config_path)
        
        print("Memos completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error: failed to import the Memos module: {e}")
        return False
    except Exception as e:
        print(f"Error: Memos raised an exception: {e}")
        return False


def run_sota(config_manager: ConfigManager):
    """Run the SOTA method."""
    print("Running SOTA...")

    try:
        print("Importing the SOTA module...")
        from Method.sota import run_sota
        print("SOTA module imported successfully.")

        config = config_manager.get_all()

        # Read values directly from the config file.
        config_path = config_manager.args.config_file
        print(f"Using config file: {config_path}")

        # Run SOTA.
        result = run_sota(
            dataset_path=config.get('dataset_path'),
            output_path=config.get('output_path'),
            memory_path=config.get('memory_path'),
            config_path=config_path,
            llm_model=config.get('llm_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
        )

        return result

    except ImportError as e:
        print(f"Error: failed to import the SOTA module: {e}")
        return False
    except Exception as e:
        print(f"Error: SOTA raised an exception: {e}")
        return False


def run_amem(config_manager: ConfigManager):
    """Run the Agentic Memory method."""
    print("Running Agentic Memory...")
    
    try:
        print("Importing the AgenticMemory module...")
        from Method.amem import run_amem as run_amem_method
        print("AgenticMemory module imported successfully.")
        
        config = config_manager.get_all()
        config_path = config_manager.args.config_file
        
        if config_path:
            print(f"Using config file: {config_path}")

        results = run_amem_method(
            dataset_path=config.get('dataset_path') or config.get('dataset'),
            output_path=config.get('output_path') or config.get('output'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model') or config.get('model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
            backend=config.get('backend', 'openai'),
            retrieve_k=config.get('retrieve_k', 10),
            ratio=config.get('ratio', 1.0),
            start_idx=config.get('start_idx', 0),
            end_idx=config.get('end_idx'),
            config_path=config_path,
            modelname=config.get('modelname'),
        )
        
        sample_count = len(results) if isinstance(results, list) else 0
        print(f"Agentic Memory completed successfully. Aggregated {sample_count} sample results.")
        return True
        
    except ImportError as e:
        print(f"Error: failed to import the AgenticMemory module: {e}")
        return False
    except Exception as e:
        print(f"Error: Agentic Memory raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False



def main():
    parser = argparse.ArgumentParser(description='Unified runner with config-file support and CLI overrides')
    
    parser.add_argument('--config_file', type=str, help='Path to the config file (YAML or JSON)')
    
    subparsers = parser.add_subparsers(dest='command', help='Select a command to run')
    
    # MemoryTree method
    parser_memtree = subparsers.add_parser('memtree', help='Run the MemoryTree method')
    parser_memtree.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memtree.add_argument('--dataset_name', type=str, help='Dataset name')
    parser_memtree.add_argument('--dataset_path', type=str, help='Dataset JSON file path')
    parser_memtree.add_argument('--mode', type=str, help='Run mode')
    parser_memtree.add_argument('--embedding_model_name', type=str, help='Embedding model name')
    parser_memtree.add_argument('--vdb_name', type=str, help='Vector database name')
    parser_memtree.add_argument('--batch_size', type=int, help='Batch size')
    parser_memtree.add_argument('--num_processes', type=int, help='Number of worker processes')
    parser_memtree.add_argument('--dimension', type=int, help='Vector dimension')
    parser_memtree.add_argument('--collection_name', type=str, help='Collection name')
    parser_memtree.add_argument('--base_threshold', type=float, help='Base threshold')
    parser_memtree.add_argument('--rate', type=float, help='Rate')
    parser_memtree.add_argument('--max_depth', type=int, help='Maximum depth')
    parser_memtree.add_argument('--retrieve_threshold', type=float, help='Retrieval threshold')
    parser_memtree.add_argument('--llm_parallel_nums', type=int, help='Number of parallel LLM workers')
    parser_memtree.add_argument('--embedding_batch_size', type=int, help='Embedding batch size')
    parser_memtree.add_argument('--top_k_retrieve', type=int, help='Top-K retrieval count')
    parser_memtree.add_argument('--save_name', type=str, help='Output file name')
    parser_memtree.add_argument('--output_path', type=str, help='Output file path')
    parser_memtree.add_argument('--token_file', type=str, help='Token tracking output file path')
    
    # MemoryOS method
    parser_memoryos = subparsers.add_parser('memoryos', help='Run the MemoryOS method')
    parser_memoryos.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memoryos.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_memoryos.add_argument('--output_path', type=str, help='Output path')
    parser_memoryos.add_argument('--memory_path', type=str, help='Memory path')
    parser_memoryos.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_memoryos.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_memoryos.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_memoryos.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_memoryos.add_argument('--token_file', type=str, help='Optional token tracking output path')
    
    # Agentic Memory method
    parser_amem = subparsers.add_parser('amem', help='Run the Agentic Memory method')
    parser_amem.add_argument('--config_file', type=str, help='Path to the config file')
    parser_amem.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_amem.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output file path')
    parser_amem.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_amem.add_argument('--llm_model', '--model', dest='llm_model', type=str, help='LLM model path or ID')
    parser_amem.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_amem.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_amem.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_amem.add_argument('--modelname', type=str, help='Model name')
    parser_amem.add_argument('--ratio', type=float, help='Dataset processing ratio (0.0 to 1.0)')
    parser_amem.add_argument('--backend', type=str, help='Backend type (openai or ollama)')
    parser_amem.add_argument('--retrieve_k', type=int, help='Number of memories to retrieve')
    parser_amem.add_argument('--start_idx', type=int, help='Start sample index (inclusive)')
    parser_amem.add_argument('--end_idx', type=int, help='End sample index (exclusive)')
    
    # Zep (Graphiti) method
    parser_zep = subparsers.add_parser('zep', help='Run the Zep (Graphiti) method')
    parser_zep.add_argument('--config_file', type=str, help='Path to the config file')
    parser_zep.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_zep.add_argument('--output_path', type=str, help='Output path')
    parser_zep.add_argument('--token_file', type=str, help='Optional token tracking output path')
    parser_zep.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_zep.add_argument('--llm_small_model', type=str, help='Small LLM model ID or path')
    parser_zep.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_zep.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_zep.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_zep.add_argument('--embedding_api_key', type=str, help='Embedding API key')
    parser_zep.add_argument('--embedding_base_url', type=str, help='Embedding API base URL')
    parser_zep.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser_zep.add_argument('--neo4j_uri', type=str, help='Neo4j URI')
    parser_zep.add_argument('--neo4j_user', type=str, help='Neo4j username')
    parser_zep.add_argument('--neo4j_password', type=str, help='Neo4j password')
    parser_zep.add_argument('--answer_model', type=str, help='Answer generation model ID or path')
    parser_zep.add_argument('--answer_api_key', type=str, help='Answer generation API key')
    parser_zep.add_argument('--answer_base_url', type=str, help='Answer generation API base URL')
    parser_zep.add_argument('--answer_temperature', type=float, help='Answer generation temperature')
    parser_zep.add_argument('--answer_max_tokens', type=int, help='Answer generation max tokens')
    
    # MemoChat method
    parser_memochat = subparsers.add_parser('memochat', help='Run the MemoChat method')
    parser_memochat.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memochat.add_argument('--input_data', type=str, help='Path to the input data file')
    parser_memochat.add_argument('--results_output_path', type=str, help='Path to the results output file')
    parser_memochat.add_argument('--memory_output_path', type=str, help='Path to the memory output file')
    parser_memochat.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser_memochat.add_argument('--openai_modelid', type=str, help='LLM model ID or path')
    parser_memochat.add_argument('--base_url', type=str, help='LLM API base URL')
    parser_memochat.add_argument('--api_key', type=str, help='LLM API key')
    parser_memochat.add_argument('--token_file', type=str, help='Optional token tracking output path')
    
    # Memos method
    parser_memos = subparsers.add_parser('memos', help='Run the Memos method')
    parser_memos.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memos.add_argument('--version', type=str, help='Version identifier')
    parser_memos.add_argument('--num_workers', type=int, help='Number of parallel worker processes')
    parser_memos.add_argument('--top_k', type=int, help='Number of results retrieved during search')
    parser_memos.add_argument('--dataset_path', type=str, help='Path to the LOCOMO dataset JSON file')
    parser_memos.add_argument('--result_dir', type=str, help='Root directory for Memos outputs')
    parser_memos.add_argument('--storage_dir', type=str, help='Directory for storage dumps')
    parser_memos.add_argument('--tmp_dir', type=str, help='Directory for intermediate search outputs')
    parser_memos.add_argument('--search_results_path', type=str, help='Path for aggregated search results')
    parser_memos.add_argument('--response_results_path', type=str, help='Path for raw response results')
    parser_memos.add_argument('--formatted_results_path', type=str, help='Path for final formatted results')
    parser_memos.add_argument('--token_file', type=str, help='Token tracking output file')
    parser_memos.add_argument('--llm_model', type=str, help='Shared LLM model path or ID')
    parser_memos.add_argument('--llm_api_key', type=str, help='Shared LLM API key')
    parser_memos.add_argument('--llm_base_url', type=str, help='Shared LLM base URL')
    parser_memos.add_argument('--embedding_model_name', type=str, help='Shared embedding model path or ID')
    parser_memos.add_argument('--response_model', type=str, help='Response-stage LLM model path or ID')
    parser_memos.add_argument('--response_api_key', type=str, help='Response-stage LLM API key')
    parser_memos.add_argument('--response_base_url', type=str, help='Response-stage LLM base URL')
    parser_memos.add_argument('--ingestion_top_k', type=int, help='Top-K used during ingestion client setup')
    parser_memos.add_argument('--mos_config_path', type=str, help='Optional custom MOS config template JSON')
    parser_memos.add_argument('--mem_cube_config_path', type=str, help='Optional custom MemCube config template JSON')
    parser_memos.add_argument('--graph_db_uri', type=str, help='Graph database URI')
    parser_memos.add_argument('--graph_db_user', type=str, help='Graph database user')
    parser_memos.add_argument('--graph_db_password', type=str, help='Graph database password')
    parser_memos.add_argument('--graph_db_auto_create', type=str_to_bool, help='Whether to auto-create the graph database')

    # SOTA method
    parser_sota = subparsers.add_parser('sota', help='Run the SOTA method')
    parser_sota.add_argument('--config_file', type=str, help='Path to the config file')
    parser_sota.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_sota.add_argument('--memory_path', type=str, help='Memory path')
    parser_sota.add_argument('--output_path', type=str, help='Output path')
    parser_sota.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_sota.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_sota.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_sota.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    
    args = parser.parse_args()
    
    # Validate that a command was provided.
    if not args.command:
        print("Error: please specify a command to run.")
        parser.print_help()
        return
    
    # Create the configuration manager.
    try:
        config_manager = ConfigManager(args)
    except Exception as e:
        print(f"Error: failed to initialize configuration: {e}")
        return
    
    # Dispatch to the selected method.
    success = False
    
    if args.command == 'memtree':
        success = run_memory_tree(config_manager)
    elif args.command == 'memoryos':
        success = run_memoryos(config_manager)
    elif args.command == 'amem':
        success = run_amem(config_manager)
    elif args.command == 'zep':
        success = run_zep(config_manager)
    elif args.command == 'memochat':
        success = run_memochat(config_manager)
    elif args.command == 'memos':
        success = run_memos(config_manager)
    elif args.command == 'sota':
        success = run_sota(config_manager)
    else:
        print(f"Error: unknown command '{args.command}'")
        parser.print_help()
        return
    
    # Print the final execution status.
    if success:
        print(f"{args.command} completed successfully!")
    else:
        print(f"{args.command} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
