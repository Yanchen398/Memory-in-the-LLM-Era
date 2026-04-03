import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
from .utils import get_embedding, insert, calculate_cos, calculate_threshold, worker_ollama, update_vector, worker_openai
from .config import globalconfig
from tqdm import tqdm
import multiprocessing

from multiprocessing import Pool, Array, Value
import ctypes
from typing import List, Dict

from .prompt import AGGREGATE_PROMPT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MemTreeNode:
    def __init__(self, content: str = ""):
        self.cv = content
        self.pv: Optional[int] = None            # 父节点ID（非对象，减少循环引用）
        self.dv: int = 0                         # 深度

class MemTree:
    def __init__(self, root_content: str = "Root"):
        self.root = MemTreeNode(root_content)
        self.nodes: Dict[int, MemTreeNode] = {id(self.root): self.root}  # 节点ID → 节点 
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)           # 邻接表：父ID → 子ID-set
        self.size = 1

    # def add_node(self, content: str, parent_id: Optional[int] = None) -> int:
    #     """
    #     添加节点并返回新节点ID
    #     """
    #     parent_id = parent_id if parent_id else id(self.root)
    #     parent = self.nodes[parent_id]
        
    #     # node_ids_in_next_layer = list(map(lambda x: x, self.adjacency[parent_id]))
        
    #     # node_embs_in_next_layer = globalconfig.client .get(
    #     #     collection_name=globalconfig.collection_name,
    #     #     ids=node_ids_in_next_layer,
    #     #     output_fields=["vector"]
    #     # )
    #     # breakpoint()

    #     new_node = MemTreeNode(content)
    #     new_node.pv = parent_id
    #     new_node.dv = parent.dv + 1
    #     new_id = id(new_node)
        
    #     # breakpoint()
    #     # embdding & insert into vdb
    #     if content:
    #         ev = get_embedding(content)
    #         ev = ev.flatten().tolist()
    #         # breakpoint()
    #         insert([{"id": new_id, "vector": ev}])

    #     self.nodes[new_id] = new_node
    #     self.adjacency[parent_id].add(new_id) # 更新父节点的邻接表
    #     self.size += 1
    #     return new_id
    
    def add_node_single(self, content: str, ev: np.ndarray, current_parent_id: int):
        current_parent = self.nodes[current_parent_id]
        
        new_node = MemTreeNode(content)
        new_node.pv = current_parent_id
        new_node.dv = current_parent.dv + 1
        new_id = id(new_node)
        
        # insert into vdb
        ev = ev.flatten().tolist()
        insert([{"id": new_id, "vector": ev}])

        self.nodes[new_id] = new_node
        self.adjacency[current_parent_id].add(new_id)
        self.size += 1
        
        return new_id
    
    def add_node(self, content: str, parent_id: Optional[int] = None) -> int:
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
            if hasattr(globalconfig, 'client') and hasattr(globalconfig, 'collection_name'):
                node_results_in_next_layer = globalconfig.client.get(
                    collection_name=globalconfig.collection_name,
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
    #     根据邻接表获取子节点对象列表
    #     """
    #     return [self.nodes[child_id] for child_id in self.adjacency.get(node_id)]

    # def traverse_from_root(self) -> List[MemTreeNode]:
    #     """
    #     迭代遍历（BFS）
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
        """在终端按层级打印树结构"""
        if not self.nodes:
            print("(空树)")
            return
        
        # 使用BFS队列: (节点ID, 节点对象)
        queue = deque([(id(self.root), self.root)])
        
        while queue:
            node_id, node = queue.popleft()
            
            # 打印当前节点
            indent = "    " * node.dv
            parent_info = f" → 父[{node.pv}]" if node.pv else ""
            print(f"{indent}├─ ID:{node_id} 内容:'{node.cv}' 深度:{node.dv}{parent_info}")
            
            # # 如果达到最大深度则停止
            # if node.dv >= max_depth:
            #     continue
                
            # 添加子节点到队列
            for child_id in self.adjacency[node_id]:
                if child_id in self.nodes:
                    child = self.nodes[child_id]
                    #child.dv = node.dv + 1  # 更新子节点深度
                    queue.append((child_id, child))

    def modify_nodes(self, new_content: str, node_ids: List[int]):
        if node_ids:
            # 预先分配共享内存
            total_tasks = len(node_ids)
            print(f"Starting modification of {total_tasks} nodes")
            
            # 验证所有节点ID都存在
            valid_node_ids = []
            for node_id in node_ids:
                if node_id in self.nodes:
                    valid_node_ids.append(node_id)
                else:
                    print(f"Warning: Node ID {node_id} not found in tree")
            
            if not valid_node_ids:
                print("No valid nodes to update")
                return
            
            # 更新任务数量为有效节点数量
            total_tasks = len(valid_node_ids)
            print(f"Processing {total_tasks} valid nodes")
            
            tasks = [(node_id, self.nodes[node_id].cv, len(self.adjacency[node_id]), new_content) for node_id in valid_node_ids]
            
            update_nodes = deque()
            successful_updates = 0
            failed_updates = 0
            
            with tqdm(total=total_tasks, desc="Processing updation of parent nodes traversed along the path...") as pbar:
                parallel_nums = getattr(globalconfig, 'llm_parallel_nums', 1)  # 使用默认值1
                with multiprocessing.Pool(processes=parallel_nums) as pool:
                    for result in pool.imap_unordered(self._modify_shared_mem, tasks):
                        node_id, output = result
                        if output is not None:
                            update_nodes.append((node_id, output))
                            successful_updates += 1
                        else:
                            failed_updates += 1
                            print(f"Warning: LLM processing failed for node {node_id}")
                        pbar.update(1)
            
            print(f"LLM processing completed: {successful_updates} successful, {failed_updates} failed")
            
            #更新之前，你讲之前的node信息，加入到当前节点中
            #banana+apple->fruit, apple挂到ftuit下。
            
            # 检查是否有成功更新的节点
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
            
            # 检查列表长度一致性
            actual_update_count = len(update_nodes)
            print(f"Successfully updated {actual_update_count} out of {total_tasks} nodes")
                
            # Batch embedding - 使用实际更新的节点数量
            batch_size = getattr(globalconfig, 'embedding_batch_size', 256)  # 使用默认值256
            evs_from_origin_node = get_embedding(content_from_origin_node, batch=batch_size)
            evs_from_current_node = get_embedding(content_from_current_node, batch=batch_size)
            evs_from_current_node = evs_from_current_node.tolist()
            
            # Update current node - 使用实际更新的节点数量
            update_data = [
                {"id": update_nodes[i][0], "vector": evs_from_current_node[i]} for i in range(actual_update_count)
            ]
            
            print(f"Updating {len(update_data)} vectors in database")
            update_vector(new_data=update_data)
            
            # Insert original node - 只有在有更新节点时才执行
            if content_from_origin_node and evs_from_origin_node.size > 0 and update_nodes:
                print(f"Adding original node back to tree")
                self.add_node_single(
                    content=content_from_origin_node[-1], 
                    ev=evs_from_origin_node[-1], 
                    current_parent_id=update_nodes[-1][0]
                )
    
    @staticmethod
    def _modify_shared_mem(args):
        node_id, current_content, len_children, new_content = args
        # current_content = cv.value.decode()
        
        input_prompt = AGGREGATE_PROMPT.format(
            new_content=new_content,
            n_children=str(len_children), 
            current_content=current_content,
        )
        
        output = worker_ollama(input_prompt)
        
        return node_id, output
        # cv.value = (current + " modified").encode()

import pickle
def save_tree(tree, filepath='memtree.pkl', i=None):
    """
    保存树结构到文件
    Args:
        tree: 要保存的树对象
        filepath: 完整的文件路径或文件名
        i: 样本索引
    """
    if i is not None:
        # 提取目录和文件名
        if os.path.dirname(filepath):
            # 如果是完整路径，分离目录和文件名
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            # 按照原始格式：sample_{i}_{filename}
            sample_filename = f"sample_{i}_{filename}"
            final_path = os.path.join(directory, sample_filename)
        else:
            # 如果只是文件名，按照原始格式
            final_path = f"sample_{i}_{filepath}"
    else:
        final_path = filepath
    
    # 确保目录存在
    directory = os.path.dirname(final_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(final_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Tree saved to: {final_path}")

def load_tree(filepath='memtree.pkl', i=None):
    """
    从文件加载树结构
    Args:
        filepath: 完整的文件路径或文件名
        i: 样本索引
    Returns:
        加载的树对象，如果文件不存在则返回None
    """
    if i is not None:
        # 提取目录和文件名
        if os.path.dirname(filepath):
            # 如果是完整路径，分离目录和文件名
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            # 按照原始格式：sample_{i}_{filename}
            sample_filename = f"sample_{i}_{filename}"
            final_path = os.path.join(directory, sample_filename)
        else:
            # 如果只是文件名，按照原始格式
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

from .token_tracker import TokenTracker

def build_tree(data, i, tracker: TokenTracker):
    tree = load_tree(globalconfig.save_path, i)
    questions, sessions = data.data[i]
    
    if tree is None:
        tree = MemTree("")
        root_id = id(tree.root)
        all_sessions = []
        for session_id, session in sessions.items():
            with tracker.stage(f"Session {session_id}"):
                all_sessions.extend(session)
                # break # 目前单session测试
                dial_id = 0
                for dial in session:
                    with tracker.stage(f"Dialog {dial_id}"):
                        tree.add_node(dial, root_id)
                    dial_id += 1

        save_tree(tree, globalconfig.save_path, i)
        
    return tree