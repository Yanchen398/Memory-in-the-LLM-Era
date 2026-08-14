from .config import globalconfig
import json
import os
from collections import defaultdict

from ..dataset_hygiene import format_turn_text, natural_session_keys

class Dataloder:
    def __init__(self, globalconfig):
        self.dataset_name = globalconfig.dataset_name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_data_path = os.path.join(current_dir, "data", globalconfig.dataset_name, "data.json")
        self.data_path = getattr(globalconfig, "dataset_path", default_data_path)
        
        self.data, self.sample_ids = self.read_data()
    
    def read_data(self):
        data = []
        sample_ids = []
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            questions = item["qa"]
            conversation = item["conversation"]
            sample_id = item["sample_id"]
            data_now = defaultdict(list)
            
            for session_id_key in natural_session_keys(conversation):
                session_time = str(conversation.get(f"{session_id_key}_date_time") or "")
                conversation_data = [
                    f"{session_time}:{turn.get('speaker', 'Unknown')}:{format_turn_text(turn)}"
                    for turn in conversation.get(session_id_key, [])
                ]
                data_now[session_id_key] = conversation_data
            
            data.append((questions, data_now))
            sample_ids.append(sample_id)
        return data, sample_ids
    
    def update_config(self, new_config):
        """
        更新数据加载器的配置
        
        Args:
            new_config: 新的配置对象
        """
        self.config = new_config
        # 如果需要的话，可以在这里添加其他配置更新逻辑
    
# 只有在 globalconfig 可用时才创建全局 dataloader 实例
if globalconfig is not None:
    dataloader = Dataloder(globalconfig)
else:
    dataloader = None
