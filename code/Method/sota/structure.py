import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
from .utils import get_embedding, insert, calculate_cos, calculate_threshold, OpenAIClient, AsyncOpenAIClient, update_vector
from . import config
from tqdm import tqdm
import asyncio
from typing import List, Dict

from .prompt import AGGREGATE_PROMPT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MemTreeNode:
    """Tree Node Structure: content value(str), parent value(int), depth value(int)"""
    def __init__(self, content: str = ""):
        self.cv = content
        self.pv: Optional[int] = None            # Parent node ID to avoid circular references.
        self.dv: int = 0   # Depth
        self.heat: dict = {"R_recency": 1, "N_visit": 0}

class MemTree:
    def __init__(self, root_content: str = "Root", api_key: str = None, base_url: str = None, model: str = None, mode: str = "default"):
        self.root = MemTreeNode(root_content)
        self.nodes: Dict[int, MemTreeNode] = {id(self.root): self.root}  # Node ID -> node
        """Node ID (int) -> node (MemTreeNode)."""
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)           # Adjacency map: parent ID -> child ID set
        self.size = 1
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.mode = mode
    
    def add_node_single(self, content: str, ev: np.ndarray, current_parent_id: int):
        """Given the parent id(the exact position), add a single node to the tree."""
        current_parent = self.nodes[current_parent_id]
        
        new_node = MemTreeNode(content)
        new_node.pv = current_parent_id
        new_node.dv = current_parent.dv + 1
        new_id = id(new_node)
        
        # insert into vdb
        ev = ev.flatten().tolist()
        insert([{"id": new_id, "vector": ev, "text": content, "type": "segment"}])

        self.nodes[new_id] = new_node
        self.adjacency[current_parent_id].add(new_id)
        self.size += 1
        
        return new_id
    
    def add_node(self, content: str, parent_id: Optional[int] = None) -> int:
        """Add a node to the tree based on content similarity traversal."""
        current_parent_id = parent_id if parent_id else id(self.root)
        ev = get_embedding(content) # 1*D
        parent_ids = []
        
        is_continue_traversal = True
        while is_continue_traversal:
            
            current_parent = self.nodes[current_parent_id]
            current_depth = current_parent.dv + 1
            # list of node_id
            node_ids_in_next_layer = list(map(lambda x: x, self.adjacency[current_parent_id]))
            #breakpoint()
            
            if not node_ids_in_next_layer:
                break
            # list of dict_keys(['id', 'vector'])
            if hasattr(config.globalconfig, 'client') and hasattr(config.globalconfig, 'collection_name'):
                node_results_in_next_layer = config.globalconfig.client.get(
                    collection_name=config.globalconfig.collection_name,
                    ids=node_ids_in_next_layer,
                )
            else:
                print("Error: globalconfig missing client or collection_name")
                raise AttributeError("globalconfig not properly initialized")
            
            # shape: node_nums * dimension
            node_embs_in_next_layer = np.array(
                list(map(lambda x: x["vector"], node_results_in_next_layer))
            )
            
            try:
                cos = calculate_cos(v=ev, M=node_embs_in_next_layer)
            except:
                breakpoint()
            current_threshold = calculate_threshold(current_depth=current_depth)
            cos = (cos > current_threshold).astype(int) * cos
            max_cos = np.max(cos)
            # if content == "computer":
            #     breakpoint()
        
            if max_cos:
                # Continue to traverse
                max_index = int(np.argmax(cos))
                current_parent_id = node_ids_in_next_layer[max_index]
                parent_ids.append(current_parent_id)
            else:
                # all child nodes' similarities are below the threshold
                # v_new is directly attached as a new leaf node under the current node.
                is_continue_traversal = False
        
        new_id = self.add_node_single(content, ev, current_parent_id)
        
        # if content == "computer":
        #     breakpoint()
        self.modify_nodes(new_content=content, node_ids=parent_ids)
        
        return new_id
        # return new_id, current_parent_id

    # def get_children(self, node_id: int) -> List[MemTreeNode]:
    #     """
    #     Get the list of child node objects from the adjacency map.
    #     """
    #     return [self.nodes[child_id] for child_id in self.adjacency.get(node_id)]

    # def traverse_from_root(self) -> List[MemTreeNode]:
    #     """
    #     Iterative traversal (BFS).
    #     """
    #     from collections import deque
    #     visited = []
    #     queue = deque([id(self.root)])
    #     while queue:
    #         node_id = queue.popleft()
    #         visited.append(self.nodes[node_id])
    #         queue.extend(list(self.adjacency.get(node_id)))
    #     return visited
    
    def print_tree_terminal(self, max_depth: int = 3):
        """Print the tree hierarchy in the terminal."""
        if not self.nodes:
            print("(empty tree)")
            return
        
        # Use a BFS queue: (node ID, node object)
        queue = deque([(id(self.root), self.root)])
        
        while queue:
            node_id, node = queue.popleft()
            
            # Print the current node.
            indent = "    " * node.dv
            parent_info = f" -> parent[{node.pv}]" if node.pv else ""
            print(f"{indent}├─ ID:{node_id} content:'{node.cv}' depth:{node.dv}{parent_info}")
            
            # # Stop if the maximum depth is reached.
            # if node.dv >= max_depth:
            #     continue
                
            # Add child nodes to the queue.
            for child_id in self.adjacency[node_id]:
                if child_id in self.nodes:
                    child = self.nodes[child_id]
                    #child.dv = node.dv + 1  # Update child depth.
                    queue.append((child_id, child))

    def modify_nodes(self, new_content: str, node_ids: List[int]):
        if node_ids:
            # Prepare bookkeeping for batched updates.
            total_tasks = len(node_ids)
            # print(f"Starting modification of {total_tasks} nodes")
            
            # Validate that all node IDs exist.
            valid_node_ids = []
            for node_id in node_ids:
                if node_id in self.nodes:
                    valid_node_ids.append(node_id)
                else:
                    print(f"Warning: Node ID {node_id} not found in tree")
            
            if not valid_node_ids:
                print("No valid nodes to update")
                return
            
            # Update the task count to the number of valid nodes.
            total_tasks = len(valid_node_ids)
            # print(f"Processing {total_tasks} valid nodes")
            
            tasks = [(node_id, self.nodes[node_id].cv, len(self.adjacency[node_id]), new_content) for node_id in valid_node_ids]
            
            update_nodes = deque()
            successful_updates = 0
            failed_updates = 0
            
            if self.mode == 'default':
                # Call the LLM sequentially with a for-loop.
                with tqdm(total=total_tasks, desc="Processing updation of parent nodes traversed along the path...") as pbar:
                    for node_id, current_content, len_children, new_content in tasks:
                        try:
                            result = self._modify_sync(
                                node_id, current_content, len_children, new_content,
                                self.api_key, self.base_url, self.model
                            )
                            node_id, output = result
                            if output is not None:
                                update_nodes.append((node_id, output))
                                successful_updates += 1
                            else:
                                failed_updates += 1
                                print(f"Warning: LLM processing failed for node {node_id}")
                        except Exception as e:
                            failed_updates += 1
                            print(f"Warning: LLM processing failed with error: {e}")
                        pbar.update(1)
                
                print(f"LLM processing completed: {successful_updates} successful, {failed_updates} failed")

            elif self.mode == 'async':
                # Create a shared client.
                shared_client = AsyncOpenAIClient(self.api_key, self.base_url)
                # Call the LLM concurrently with async tasks.
                async def run_all_tasks():
                    try:
                        async_tasks = [
                            self._modify_async(
                                node_id, current_content, len_children, new_content,
                                shared_client, self.model
                            )
                            for node_id, current_content, len_children, new_content in tasks
                        ]
                        return await asyncio.gather(*async_tasks, return_exceptions=True)
                    finally:
                        await shared_client.client.close()

                results = asyncio.run(run_all_tasks())
                
                # Process results.
                with tqdm(total=total_tasks, desc="Processing updation of parent nodes traversed along the path...") as pbar:
                    for result in results:
                        if isinstance(result, Exception):
                            failed_updates += 1
                            print(f"Warning: LLM processing failed with error: {result}")
                        else:
                            node_id, output = result
                            if output is not None:
                                update_nodes.append((node_id, output))
                                successful_updates += 1
                            else:
                                failed_updates += 1
                                print(f"Warning: LLM processing failed for node {node_id}")
                        pbar.update(1)
                
                # print(f"LLM processing completed: {successful_updates} successful, {failed_updates} failed")
            
            # Before inserting the original node back, keep the old content as a child.
            # Example: banana + apple -> fruit, and apple is attached under fruit.
            
            # Check whether any nodes were updated successfully.
            if not update_nodes:
                print("Warning: No nodes were successfully updated by LLM")
                return
            
            content_from_origin_node = []
            content_from_current_node = []
            for item in update_nodes:
                node_id, update_content = item
                # save content from original nodes and current nodes
                content_from_origin_node.append(self.nodes[node_id].cv)
                content_from_current_node.append(update_content)
                self.nodes[node_id].cv = update_content
            
            # Track the actual number of successful updates.
            actual_update_count = len(update_nodes)
            # print(f"Successfully updated {actual_update_count} out of {total_tasks} nodes")
                
            # Batch embedding for the actual number of updated nodes.
            batch_size = getattr(config.globalconfig, 'embedding_batch_size', 256)  # Default to 256.
            evs_from_origin_node = get_embedding(content_from_origin_node, batch=batch_size)
            evs_from_current_node = get_embedding(content_from_current_node, batch=batch_size)
            evs_from_current_node = evs_from_current_node.tolist()
            
            # Update current nodes using the actual number of updated entries.
            update_data = [
                {"id": update_nodes[i][0], "vector": evs_from_current_node[i], "text": content_from_current_node[i], "type": "segment"} for i in range(actual_update_count)
            ]
            
            # print(f"Updating {len(update_data)} vectors in database")
            update_vector(new_data=update_data)
            
            # Insert the original node only when updates succeeded.
            if content_from_origin_node and evs_from_origin_node.size > 0 and update_nodes:
                # print(f"Adding original node back to tree")
                self.add_node_single(
                    content=content_from_origin_node[-1], 
                    ev=evs_from_origin_node[-1], 
                    current_parent_id=update_nodes[-1][0]
                )
    
    @staticmethod
    def _modify_sync(node_id, current_content, len_children, new_content, api_key, base_url, model):
        """Synchronously call the LLM to update a node."""
        input_prompt = AGGREGATE_PROMPT.format(
            new_content=new_content,
            n_children=str(len_children), 
            current_content=current_content,
        )
        client = OpenAIClient(api_key=api_key, base_url=base_url)
        output = client.chat_completion(model, messages=[{"role": "user", "content": input_prompt}])
        return node_id, output

    @staticmethod
    async def _modify_async(node_id, current_content, len_children, new_content, client, model):
        """Asynchronously call the LLM to update a node."""
        input_prompt = AGGREGATE_PROMPT.format(
            new_content=new_content,
            n_children=str(len_children), 
            current_content=current_content,
        )
        output = await client.chat_completion(model, messages=[{"role": "user", "content": input_prompt}])
        return node_id, output

import pickle
def save_tree(tree, filepath='memtree.pkl', i=None):
    """
    Save the tree structure to a file.
    Args:
        tree: The tree object to save.
        filepath: Full file path or file name.
        i: Sample index.
    """
    if i is not None:
        # Extract the directory and file name.
        if os.path.dirname(filepath):
            # If this is a full path, split it into directory and file name.
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            # Keep the original naming format: sample_{i}_{filename}
            sample_filename = f"sample_{i}_{filename}"
            final_path = os.path.join(directory, sample_filename)
        else:
            # If this is only a file name, keep the original naming format.
            final_path = f"sample_{i}_{filepath}"
    else:
        final_path = filepath
    
    # Ensure the target directory exists.
    directory = os.path.dirname(final_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(final_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Tree saved to: {final_path}")

def load_tree(filepath='memtree.pkl', i=None):
    """
    Load the tree structure from a file.
    Args:
        filepath: Full file path or file name.
        i: Sample index.
    Returns:
        The loaded tree object, or None if the file does not exist.
    """
    if i is not None:
        # Extract the directory and file name.
        if os.path.dirname(filepath):
            # If this is a full path, split it into directory and file name.
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            # Keep the original naming format: sample_{i}_{filename}
            sample_filename = f"sample_{i}_{filename}"
            final_path = os.path.join(directory, sample_filename)
        else:
            # If this is only a file name, keep the original naming format.
            final_path = f"sample_{i}_{filepath}"
    else:
        final_path = filepath
    
    if os.path.exists(final_path):
        with open(final_path, 'rb') as f:
            print(f"Tree loaded from: {final_path}")
            return pickle.load(f)
    else:
        print(f"Tree file not found: {final_path}")
    return None

def build_tree(data, i):
    tree = load_tree(config.globalconfig.save_path, i)
    questions, sessions = data.data[i]
    
    if tree is None:
        tree = MemTree("")
        root_id = id(tree.root)
        all_sessions = []
        for session_id, session in sessions.items():
            all_sessions.extend(session)
            # break  # Temporary single-session testing.
            dial_id = 0
            for dial in session:
                tree.add_node(dial, root_id)
                dial_id += 1

        save_tree(tree, config.globalconfig.save_path, i)
        
    return tree
