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

8. memgpt - Run the MemGPT (Letta) method
   Basic usage (config file required):
   python run.py memgpt --config_file <config.yaml>

   Override values from the config file:
   python run.py memgpt --config_file <config.yaml> --dataset_path <path> --output_path <path>

9. mem0 - Run the Mem0 method
   Basic usage (config file required):
   python run.py mem0 --config_file <config.yaml>

   Override values from the config file:
   python run.py mem0 --config_file <config.yaml> --dataset_path <path> --output_path <path>

10. mem0g - Run the Mem0$^g$ method
   Basic usage (config file required):
   python run.py mem0g --config_file <config.yaml>

   Override values from the config file:
   python run.py mem0g --config_file <config.yaml> --dataset_path <path> --output_path <path>

11. memorybank - Run the MemoryBank method
   Basic usage (config file required):
   python run.py memorybank --config_file <config.yaml>

   Override values from the config file:
   python run.py memorybank --config_file <config.yaml> --dataset_path <path> --output_path <path>

12. lightmem - Run the LightMem method
   Basic usage (config file required):
   python run.py lightmem --config_file <config.yaml>

   Override values from the config file:
   python run.py lightmem --config_file <config.yaml> --dataset_path <path> --output_path <path>

13. memgas - Run the MemGAS method
   Basic usage (config file required):
   python run.py memgas --config_file <config.yaml>

   Override values from the config file:
   python run.py memgas --config_file <config.yaml> --dataset_path <path> --output_path <path>

Note: all methods should normally be launched with their YAML config under `Config/`.
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


def run_memgpt(config_manager: ConfigManager):
    """Run the MemGPT (Letta) method."""
    print("Running MemGPT (Letta)...")

    try:
        print("Importing the MemGPT module...")
        from Method.memgpt import run_memgpt as run_memgpt_method
        print("MemGPT module imported successfully.")

        config = config_manager.get_all()
        config_path = config_manager.args.config_file

        if config_path:
            print(f"Using config file: {config_path}")

        results = run_memgpt_method(
            dataset_path=config.get('dataset_path'),
            output_path=config.get('output_path'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
            embedding_api_key=config.get('embedding_api_key'),
            embedding_base_url=config.get('embedding_base_url'),
            embedding_endpoint_type=config.get('embedding_endpoint_type'),
            embedding_dim=config.get('embedding_dim'),
            letta_base_url=config.get('letta_base_url'),
            letta_pg_uri=config.get('letta_pg_uri'),
            letta_server_command=config.get('letta_server_command'),
            letta_init_command=config.get('letta_init_command'),
            letta_startup_timeout=config.get('letta_startup_timeout'),
            retrieve_k=config.get('retrieve_k', 10),
            start_idx=config.get('start_idx', 0),
            end_idx=config.get('end_idx'),
            config_path=config_path,
        )

        sample_count = len(results) if isinstance(results, list) else 0
        print(f"MemGPT completed successfully. Aggregated {sample_count} sample results.")
        return True

    except ImportError as e:
        print(f"Error: failed to import the MemGPT module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemGPT raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def _run_mem0_variant(config_manager: ConfigManager, graph_mode: bool):
    method_label = "Mem0^g" if graph_mode else "Mem0"
    print(f"Running {method_label}...")

    try:
        print(f"Importing the {method_label} module...")
        from Method.mem0 import run_mem0, run_mem0g
        print(f"{method_label} module imported successfully.")

        runner = run_mem0g if graph_mode else run_mem0
        config = config_manager.get_all()
        config_path = config_manager.args.config_file

        if config_path:
            print(f"Using config file: {config_path}")

        results = runner(
            dataset_path=config.get('dataset_path') or config.get('dataset'),
            output_path=config.get('output_path') or config.get('output'),
            memory_path=config.get('memory_path'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model') or config.get('model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            llm_provider=config.get('llm_provider'),
            embedding_model_name=config.get('embedding_model_name'),
            embedding_api_key=config.get('embedding_api_key'),
            embedding_base_url=config.get('embedding_base_url'),
            embedding_dim=config.get('embedding_dim'),
            retrieve_k=config.get('retrieve_k', 10),
            batch_size=config.get('batch_size', 2),
            ratio=config.get('ratio', 1.0),
            start_idx=config.get('start_idx', 0),
            end_idx=config.get('end_idx'),
            neo4j_uri=config.get('neo4j_uri'),
            neo4j_user=config.get('neo4j_user'),
            neo4j_password=config.get('neo4j_password'),
            config_path=config_path,
        )

        sample_count = len(results) if isinstance(results, list) else 0
        print(f"{method_label} completed successfully. Aggregated {sample_count} sample results.")
        return True

    except ImportError as e:
        print(f"Error: failed to import the {method_label} module: {e}")
        return False
    except Exception as e:
        print(f"Error: {method_label} raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mem0(config_manager: ConfigManager):
    return _run_mem0_variant(config_manager, graph_mode=False)


def run_mem0g(config_manager: ConfigManager):
    return _run_mem0_variant(config_manager, graph_mode=True)


def run_memorybank(config_manager: ConfigManager):
    """Run the MemoryBank method."""
    print("Running MemoryBank...")

    try:
        print("Importing the MemoryBank module...")
        from Method.memorybank import run_memorybank as run_memorybank_method
        print("MemoryBank module imported successfully.")

        config = config_manager.get_all()
        config_path = config_manager.args.config_file

        if config_path:
            print(f"Using config file: {config_path}")

        results = run_memorybank_method(
            dataset_path=config.get('dataset_path') or config.get('dataset'),
            output_path=config.get('output_path') or config.get('output'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model') or config.get('model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            embedding_model_name=config.get('embedding_model_name'),
            retrieve_k=config.get('retrieve_k', 10),
            ratio=config.get('ratio', 1.0),
            start_idx=config.get('start_idx', 0) if config.get('start_idx') is not None else config.get('start_id', 0),
            end_idx=config.get('end_idx') if config.get('end_idx') is not None else config.get('end_id'),
            config_path=config_path,
        )

        sample_count = len(results) if isinstance(results, list) else 0
        print(f"MemoryBank completed successfully. Aggregated {sample_count} sample results.")
        return True

    except ImportError as e:
        print(f"Error: failed to import the MemoryBank module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemoryBank raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_lightmem(config_manager: ConfigManager):
    """Run LightMem through the repository's unified dataset interface."""
    print("Running LightMem...")

    try:
        print("Importing the LightMem module...")
        from Method.lightmem import run_lightmem as run_lightmem_method
        print("LightMem module imported successfully.")

        config = config_manager.get_all()
        config_path = config_manager.args.config_file
        if config_path:
            print(f"Using config file: {config_path}")

        results = run_lightmem_method(
            dataset_path=config.get('dataset_path') or config.get('dataset'),
            output_path=config.get('output_path') or config.get('output'),
            memory_path=config.get('memory_path'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            llm_provider=config.get('llm_provider', 'openai'),
            embedding_model_name=config.get('embedding_model_name'),
            embedding_dim=config.get('embedding_dim', 384),
            embedding_device=config.get('embedding_device', 'cpu'),
            retrieve_k=config.get('retrieve_k'),
            retrieve_top_ks=config.get('retrieve_top_ks'),
            ratio=config.get('ratio', 1.0),
            start_idx=config.get('start_idx', 0),
            end_idx=config.get('end_idx'),
            pre_compress=config.get('pre_compress', False),
            llmlingua_model_path=config.get('llmlingua_model_path'),
            compression_rate=config.get('compression_rate', 0.6),
            topic_segment=config.get('topic_segment', True),
            metadata_generate=config.get('metadata_generate', True),
            text_summary=config.get('text_summary', True),
            extraction_mode=config.get('extraction_mode', 'flat'),
            enable_update=config.get('enable_update', True),
            update_top_k=config.get('update_top_k', 20),
            update_keep_top_n=config.get('update_keep_top_n', 10),
            update_score_threshold=config.get('update_score_threshold', 0.9),
            update_workers=config.get('update_workers', 5),
            enable_summary=config.get('enable_summary', False),
            summary_time_window=config.get('summary_time_window', 3600),
            summary_top_k_seeds=config.get('summary_top_k_seeds', 15),
            auto_eval=config.get('auto_eval'),
            eval_embedding_model=config.get('eval_embedding_model'),
            config_path=config_path,
            config_overrides=config,
        )

        if isinstance(results, dict):
            result_count = sum(len(items) for items in results.values() if isinstance(items, list))
        else:
            result_count = len(results) if isinstance(results, list) else 0
        print(f"LightMem completed successfully. Aggregated {result_count} result entries.")
        return True

    except ImportError as e:
        print(f"Error: failed to import the LightMem module: {e}")
        return False
    except Exception as e:
        print(f"Error: LightMem raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_memgas(config_manager: ConfigManager):
    """Run MemGAS through the repository's unified dataset interface."""
    print("Running MemGAS...")

    try:
        print("Importing the MemGAS module...")
        from Method.memgas import run_memgas as run_memgas_method
        print("MemGAS module imported successfully.")

        config = config_manager.get_all()
        config_path = config_manager.args.config_file
        if config_path:
            print(f"Using config file: {config_path}")

        results = run_memgas_method(
            dataset_path=config.get('dataset_path') or config.get('dataset'),
            output_path=config.get('output_path') or config.get('output'),
            memory_path=config.get('memory_path'),
            token_file=config.get('token_file'),
            llm_model=config.get('llm_model'),
            llm_api_key=config.get('llm_api_key'),
            llm_base_url=config.get('llm_base_url'),
            llm_provider=config.get('llm_provider', 'vllm'),
            embedder=config.get('embedder', 'contriever'),
            device=config.get('device'),
            batch_size=config.get('batch_size', 64),
            embedder_api_key=config.get('embedder_api_key', 'EMPTY'),
            embedder_base_url=config.get('embedder_base_url'),
            embedder_model=config.get('embedder_model'),
            embedder_max_tokens=config.get('embedder_max_tokens', 256),
            llm_max_tokens=config.get('llm_max_tokens', 500),
            llm_temperature=config.get('llm_temperature', 0.0),
            llm_max_retries=config.get('llm_max_retries', 3),
            llm_retry_wait_sec=config.get('llm_retry_wait_sec', 2.0),
            llm_context_window=config.get('llm_context_window', 16384),
            llm_prompt_token_buffer=config.get('llm_prompt_token_buffer', 128),
            llm_use_qwen_thinking_control=config.get('llm_use_qwen_thinking_control', True),
            retrieve_k=config.get('retrieve_k', 3),
            retrieve_top_ks=config.get('retrieve_top_ks'),
            auto_eval=config.get('auto_eval', False),
            eval_dataset=config.get('eval_dataset', 'loco'),
            eval_embedding_model=config.get('eval_embedding_model'),
            ratio=config.get('ratio', 1.0),
            start_idx=config.get('start_idx', 0),
            end_idx=config.get('end_idx'),
            mode=config.get('mode', 'memgas'),
            mem_threshold=config.get('mem_threshold', 30),
            n_components=config.get('n_components', 2),
            num_seednodes=config.get('num_seednodes', 15),
            damping=config.get('damping', 0.1),
            router_temp=config.get('router_temp', 0.2),
            num_workers=config.get('num_workers', 1),
            track_tokens=config.get('track_tokens', True),
            record_latency=config.get('record_latency', True),
            config_path=config_path,
        )

        result_count = sum(len(items) for items in results.values() if isinstance(items, list)) if isinstance(results, dict) else 0
        print(f"MemGAS completed successfully. Aggregated {result_count} result entries.")
        return True

    except ImportError as e:
        print(f"Error: failed to import the MemGAS module: {e}")
        return False
    except Exception as e:
        print(f"Error: MemGAS raised an exception: {e}")
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

    # MemGPT (Letta) method
    parser_memgpt = subparsers.add_parser('memgpt', help='Run the MemGPT (Letta) method')
    parser_memgpt.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memgpt.add_argument('--dataset_path', type=str, help='Dataset path')
    parser_memgpt.add_argument('--output_path', type=str, help='Output path')
    parser_memgpt.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_memgpt.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_memgpt.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_memgpt.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_memgpt.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_memgpt.add_argument('--embedding_api_key', type=str, help='Embedding API key')
    parser_memgpt.add_argument('--embedding_base_url', type=str, help='Embedding API base URL')
    parser_memgpt.add_argument('--embedding_endpoint_type', type=str, help='Embedding endpoint type')
    parser_memgpt.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser_memgpt.add_argument('--letta_base_url', type=str, help='Letta server base URL')
    parser_memgpt.add_argument('--letta_pg_uri', type=str, help='Letta PostgreSQL URI')
    parser_memgpt.add_argument('--letta_server_command', type=str, help='Command used to start the Letta server')
    parser_memgpt.add_argument('--letta_init_command', type=str, help='Optional Letta initialization command')
    parser_memgpt.add_argument('--letta_startup_timeout', type=int, help='Letta server startup timeout in seconds')
    parser_memgpt.add_argument('--retrieve_k', type=int, help='Number of memories to retrieve')
    parser_memgpt.add_argument('--start_idx', type=int, help='Start sample index (inclusive)')
    parser_memgpt.add_argument('--end_idx', type=int, help='End sample index (exclusive)')

    # Mem0 method
    parser_mem0 = subparsers.add_parser('mem0', help='Run the Mem0 method')
    parser_mem0.add_argument('--config_file', type=str, help='Path to the config file')
    parser_mem0.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_mem0.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output result path')
    parser_mem0.add_argument('--memory_path', type=str, help='Runtime memory directory')
    parser_mem0.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_mem0.add_argument('--llm_model', '--model', dest='llm_model', type=str, help='LLM model ID or path')
    parser_mem0.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_mem0.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_mem0.add_argument('--llm_provider', type=str, help='Mem0 LLM provider name')
    parser_mem0.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_mem0.add_argument('--embedding_api_key', type=str, help='Embedding API key')
    parser_mem0.add_argument('--embedding_base_url', type=str, help='Embedding API base URL')
    parser_mem0.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser_mem0.add_argument('--retrieve_k', type=int, help='Number of memories to retrieve')
    parser_mem0.add_argument('--batch_size', type=int, help='Dialogue ingestion batch size')
    parser_mem0.add_argument('--ratio', type=float, help='Dataset processing ratio')
    parser_mem0.add_argument('--start_idx', '--sample_start', dest='start_idx', type=int, help='Start sample index (inclusive)')
    parser_mem0.add_argument('--end_idx', '--sample_end', dest='end_idx', type=int, help='End sample index (exclusive)')

    # Mem0^g method
    parser_mem0g = subparsers.add_parser('mem0g', help='Run the Mem0^g method')
    parser_mem0g.add_argument('--config_file', type=str, help='Path to the config file')
    parser_mem0g.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_mem0g.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output result path')
    parser_mem0g.add_argument('--memory_path', type=str, help='Runtime memory directory')
    parser_mem0g.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_mem0g.add_argument('--llm_model', '--model', dest='llm_model', type=str, help='LLM model ID or path')
    parser_mem0g.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_mem0g.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_mem0g.add_argument('--llm_provider', type=str, help='Mem0 LLM provider name')
    parser_mem0g.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_mem0g.add_argument('--embedding_api_key', type=str, help='Embedding API key')
    parser_mem0g.add_argument('--embedding_base_url', type=str, help='Embedding API base URL')
    parser_mem0g.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser_mem0g.add_argument('--retrieve_k', type=int, help='Number of memories to retrieve')
    parser_mem0g.add_argument('--batch_size', type=int, help='Dialogue ingestion batch size')
    parser_mem0g.add_argument('--ratio', type=float, help='Dataset processing ratio')
    parser_mem0g.add_argument('--start_idx', '--sample_start', dest='start_idx', type=int, help='Start sample index (inclusive)')
    parser_mem0g.add_argument('--end_idx', '--sample_end', dest='end_idx', type=int, help='End sample index (exclusive)')
    parser_mem0g.add_argument('--neo4j_uri', type=str, help='Neo4j URI')
    parser_mem0g.add_argument('--neo4j_user', type=str, help='Neo4j username')
    parser_mem0g.add_argument('--neo4j_password', type=str, help='Neo4j password')

    # MemoryBank method
    parser_memorybank = subparsers.add_parser('memorybank', help='Run the MemoryBank method')
    parser_memorybank.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memorybank.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_memorybank.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output result path')
    parser_memorybank.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_memorybank.add_argument('--llm_model', '--model', dest='llm_model', type=str, help='LLM model ID or path')
    parser_memorybank.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_memorybank.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_memorybank.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_memorybank.add_argument('--retrieve_k', type=int, help='Number of memory chunks to retrieve')
    parser_memorybank.add_argument('--ratio', type=float, help='Dataset processing ratio')
    parser_memorybank.add_argument('--start_idx', '--start_id', dest='start_idx', type=int, help='Start sample index (inclusive)')
    parser_memorybank.add_argument('--end_idx', '--end_id', dest='end_idx', type=int, help='End sample index (exclusive)')

    # LightMem method
    parser_lightmem = subparsers.add_parser('lightmem', help='Run the LightMem method')
    parser_lightmem.add_argument('--config_file', type=str, help='Path to the config file')
    parser_lightmem.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_lightmem.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output result path')
    parser_lightmem.add_argument('--memory_path', type=str, help='Runtime memory directory')
    parser_lightmem.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_lightmem.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_lightmem.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_lightmem.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_lightmem.add_argument('--llm_provider', type=str, help='LightMem memory-manager provider')
    parser_lightmem.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
    parser_lightmem.add_argument('--embedding_api_key', type=str, help='Embedding API key')
    parser_lightmem.add_argument('--embedding_base_url', type=str, help='Embedding API base URL')
    parser_lightmem.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser_lightmem.add_argument('--embedding_device', type=str, help='Embedding and LLMLingua device')
    parser_lightmem.add_argument('--retrieve_k', type=int, help='Number of memories to retrieve')
    parser_lightmem.add_argument('--retrieve_top_ks', type=str, help='Comma-separated LOCOMO retrieval cutoffs')
    parser_lightmem.add_argument('--ratio', type=float, help='Dataset processing ratio')
    parser_lightmem.add_argument('--start_idx', type=int, help='Start sample index (inclusive)')
    parser_lightmem.add_argument('--end_idx', type=int, help='End sample index (exclusive)')
    parser_lightmem.add_argument('--pre_compress', type=str_to_bool, help='Enable LLMLingua pre-compression')
    parser_lightmem.add_argument('--llmlingua_model_path', type=str, help='LLMLingua-2 model name or path')
    parser_lightmem.add_argument('--llmlingua_service_url', type=str, help='Optional remote LLMLingua service URL')
    parser_lightmem.add_argument('--compression_rate', type=float, help='LLMLingua compression rate')
    parser_lightmem.add_argument('--topic_segment', type=str_to_bool, help='Enable topic segmentation')
    parser_lightmem.add_argument('--metadata_generate', type=str_to_bool, help='Enable metadata generation')
    parser_lightmem.add_argument('--text_summary', type=str_to_bool, help='Enable memory text summaries')
    parser_lightmem.add_argument('--extraction_mode', choices=['flat', 'event'], help='Memory extraction granularity')
    parser_lightmem.add_argument('--enable_update', type=str_to_bool, help='Enable offline memory updates')
    parser_lightmem.add_argument('--update_top_k', type=int, help='Candidate count for memory updates')
    parser_lightmem.add_argument('--update_keep_top_n', type=int, help='Candidates retained for memory updates')
    parser_lightmem.add_argument('--update_score_threshold', type=float, help='Memory update score threshold')
    parser_lightmem.add_argument('--update_workers', type=int, help='Memory update worker count')
    parser_lightmem.add_argument('--enable_summary', type=str_to_bool, help='Enable global memory summarization')
    parser_lightmem.add_argument('--summary_time_window', type=int, help='Summary time window in seconds')
    parser_lightmem.add_argument('--summary_top_k_seeds', type=int, help='Seed count for summarization')
    parser_lightmem.add_argument('--num_processes', type=int, help='Parallel sample worker count')
    parser_lightmem.add_argument('--worker_cuda_device_count', type=int, help='CUDA devices assigned across workers')
    parser_lightmem.add_argument('--force_rebuild', type=str_to_bool, help='Ignore cached sample results')
    parser_lightmem.add_argument('--track_tokens', type=str_to_bool, help='Write token usage reports')
    parser_lightmem.add_argument('--auto_eval', type=str_to_bool, help='Run unified evaluation after inference')
    parser_lightmem.add_argument('--eval_embedding_model', type=str, help='Embedding model used by evaluation')

    # MemGAS method
    parser_memgas = subparsers.add_parser('memgas', help='Run the MemGAS method')
    parser_memgas.add_argument('--config_file', type=str, help='Path to the config file')
    parser_memgas.add_argument('--dataset_path', '--dataset', dest='dataset_path', type=str, help='Dataset path')
    parser_memgas.add_argument('--output_path', '--output', dest='output_path', type=str, help='Output result path')
    parser_memgas.add_argument('--memory_path', type=str, help='Runtime memory directory')
    parser_memgas.add_argument('--token_file', type=str, help='Token tracking output file path')
    parser_memgas.add_argument('--llm_model', type=str, help='LLM model ID or path')
    parser_memgas.add_argument('--llm_api_key', type=str, help='LLM API key')
    parser_memgas.add_argument('--llm_base_url', type=str, help='LLM API base URL')
    parser_memgas.add_argument('--llm_provider', choices=['openai', 'vllm'], help='LLM provider')
    parser_memgas.add_argument('--llm_max_tokens', type=int, help='Maximum generated tokens per LLM call')
    parser_memgas.add_argument('--llm_temperature', type=float, help='LLM sampling temperature')
    parser_memgas.add_argument('--llm_max_retries', type=int, help='Maximum LLM retry attempts')
    parser_memgas.add_argument('--llm_retry_wait_sec', type=float, help='Delay between LLM retries')
    parser_memgas.add_argument('--llm_context_window', type=int, help='LLM context-window size')
    parser_memgas.add_argument('--llm_prompt_token_buffer', type=int, help='Reserved prompt token buffer')
    parser_memgas.add_argument('--llm_use_qwen_thinking_control', type=str_to_bool, help='Disable Qwen thinking through request metadata')
    parser_memgas.add_argument('--embedder', choices=['contriever', 'mpnet', 'minilm', 'qaminilm', 'openai'], help='Embedding backend')
    parser_memgas.add_argument('--device', type=str, help='Embedding device, such as cpu or cuda')
    parser_memgas.add_argument('--batch_size', type=int, help='Embedding batch size')
    parser_memgas.add_argument('--embedder_api_key', type=str, help='Embedding API key')
    parser_memgas.add_argument('--embedder_base_url', type=str, help='Embedding API base URL')
    parser_memgas.add_argument('--embedder_model', type=str, help='Embedding API model name')
    parser_memgas.add_argument('--embedder_max_tokens', type=int, help='Embedding input truncation limit')
    parser_memgas.add_argument('--retrieve_k', type=int, help='Default number of memories to retrieve')
    parser_memgas.add_argument('--retrieve_top_ks', type=str, help='Comma-separated retrieval cutoffs')
    parser_memgas.add_argument('--mode', choices=['session_level', 'turn_level', 'summary_level', 'keyword_level', 'hybrid_level', 'lite', 'memgas'], help='Retrieval/fusion mode')
    parser_memgas.add_argument('--mem_threshold', type=int, help='Maximum graph edge candidates')
    parser_memgas.add_argument('--n_components', type=int, help='Gaussian-mixture component count')
    parser_memgas.add_argument('--num_seednodes', type=int, help='Personalized PageRank seed count')
    parser_memgas.add_argument('--damping', type=float, help='Personalized PageRank damping factor')
    parser_memgas.add_argument('--router_temp', type=float, help='Granularity router temperature')
    parser_memgas.add_argument('--ratio', type=float, help='Dataset processing ratio')
    parser_memgas.add_argument('--start_idx', type=int, help='Start sample index (inclusive)')
    parser_memgas.add_argument('--end_idx', type=int, help='End sample index (exclusive)')
    parser_memgas.add_argument('--num_workers', type=int, help='Parallel sample worker count')
    parser_memgas.add_argument('--track_tokens', type=str_to_bool, help='Write token usage reports')
    parser_memgas.add_argument('--record_latency', type=str_to_bool, help='Record per-query retrieval latency')
    parser_memgas.add_argument('--auto_eval', type=str_to_bool, help='Run unified evaluation after inference')
    parser_memgas.add_argument('--eval_dataset', choices=['loco', 'lme'], help='Unified evaluation dataset name')
    parser_memgas.add_argument('--eval_embedding_model', type=str, help='Embedding model used by evaluation')
    
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
    elif args.command == 'memgpt':
        success = run_memgpt(config_manager)
    elif args.command == 'mem0':
        success = run_mem0(config_manager)
    elif args.command == 'mem0g':
        success = run_mem0g(config_manager)
    elif args.command == 'memorybank':
        success = run_memorybank(config_manager)
    elif args.command == 'lightmem':
        success = run_lightmem(config_manager)
    elif args.command == 'memgas':
        success = run_memgas(config_manager)
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
