# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from typing import Dict, List, Optional


class SLO:
    """服务水平目标(SLO)定义"""
    def __init__(self, TTFT: float, TPOT: float):
        self.TTFT = TTFT  # Target Time to First Token
        self.TPOT = TPOT  # Target Time to Output Token


class PerformanceProfile:
    """性能参数配置类,用于存储从profiler文件加载的性能数据"""
    def __init__(self):
        # 数据结构: {model_name: {tp: {"prefill": [a, b, c], "decode": [a, b, c]}}}
        # prefill/decode参数含义: [a, b, c] 对应线性回归参数
        # 对于prefill: latency = a + b * input_tokens + c * input_tokens^2
        # 对于decode: latency = a + b * output_tokens + c (第三个参数通常是常数项)
        self.profiles: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
        
    def load_from_file(self, file_path: str) -> bool:
        """从JSON文件加载性能参数
        
        Args:
            file_path: profiler JSON文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Performance profile file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 解析JSON数据
            for model_name, tp_data in data.items():
                if model_name not in self.profiles:
                    self.profiles[model_name] = {}
                
                for tp_str, phase_data in tp_data.items():
                    tp = int(tp_str)  # TP度: 1, 2, 4等
                    self.profiles[model_name][tp] = {
                        "prefill": phase_data.get("prefill", [0, 0, 0]),
                        "decode": phase_data.get("decode", [0, 0, 0])
                    }
            
            print(f"Successfully loaded performance profiles from {file_path}")
            print(f"Models: {list(self.profiles.keys())}")
            return True
            
        except Exception as e:
            print(f"Error loading performance profile: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_prefill_latency(self, model_name: str, tp: int, input_tokens: int) -> Optional[float]:
        """计算prefill阶段的延迟(毫秒)
        
        Args:
            model_name: 模型名称
            tp: Tensor Parallelism度
            input_tokens: 输入token数量
            
        Returns:
            预估延迟(毫秒),如果找不到参数则返回None
        """
        if model_name not in self.profiles or tp not in self.profiles[model_name]:
            return None
        
        params = self.profiles[model_name][tp]["prefill"]
        # latency = a + b * input_tokens + c * input_tokens^2
        latency = params[0] + params[1] * input_tokens + params[2] * (input_tokens ** 2)
        return max(0, latency)  # 确保非负
    
    def get_decode_latency(self, model_name: str, tp: int,input_tokens, output_tokens: int) -> Optional[float]:
        """计算decode阶段的延迟(毫秒)——TPOT
        
        Args:
            model_name: 模型名称
            tp: Tensor Parallelism度
            output_tokens: 输出token数量
            
        Returns:
            预估延迟(毫秒),如果找不到参数则返回None
        """
        if model_name not in self.profiles or tp not in self.profiles[model_name]:
            return None
        
        params = self.profiles[model_name][tp]["decode"]
        # decode通常是线性关系: latency = a + b * output_tokens
        # 第三个参数通常是额外的常数项
        latency = params[0] + params[1] * output_tokens + params[2]
        return max(0, latency)  # 确保非负
    
    def has_profile(self, model_name: str, tp: int) -> bool:
        """检查是否存在指定模型和TP度的性能参数"""
        return model_name in self.profiles and tp in self.profiles[model_name]


class Instance:
    """实例类，包含实例的所有配置信息"""
    def __init__(self, http_address: str, zmq_address: str = None, 
                 tp: int = 1, min_token: int = 0, max_token: int = float('inf'),
                 instance_type: str = "general"):
        """
        Args:
            http_address: HTTP地址 (ip:port)
            zmq_address: ZMQ地址 (ip:port)
            tp: 并行度配置 (tensor parallelism)
            min_token: 处理的最小token长度
            max_token: 处理的最大token长度
            instance_type: 实例类型 ("prefill", "decode", "general")
        """
        self.http_address = http_address
        self.zmq_address = zmq_address
        self.tp = tp  # 张量并行度配置
        self.min_token = min_token # 当前实例处理的最小token
        self.max_token = max_token # 当前实例处理的最大token
        self.instance_type = instance_type # 当前
        self.last_ping_time = time.time()
        
    def __eq__(self, other):
        # == 和 in时执行
        if not isinstance(other, Instance):
            return False
        return self.http_address == other.http_address
    
    def __hash__(self):
        # 用作字典键或集合元素时调用
        return hash(self.http_address)
    
    def __repr__(self):
        # 打印或字符串转换时调用
        return (f"Instance(http={self.http_address}, zmq={self.zmq_address}, "
                f"tp={self.tp}, tokens=[{self.min_token}, {self.max_token}], "
                f"type={self.instance_type})")
