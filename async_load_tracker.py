"""
==================================================================================
动态区间自适应调度器 (SLO-Aware Dynamic Interval Adjustment Scheduler)
==================================================================================

核心功能：
1. **区间管理**：每个实例负责一个请求长度区间 [min_length, max_length]
2. **SLO感知**：基于TTFT/TPOT约束选择实例
3. **动态调整**：根据负载自适应扩张/收缩区间
4. **稳定性保证**：通过滞回机制、速率限制和成本建模避免频繁调整

调度流程：
阶段1：查找负责当前请求长度的实例
阶段2：检查负责实例是否满足SLO → 满足则直接返回
阶段3：搜索相邻区间和非相邻区间的候选实例
阶段4：评估候选实例（性能收益 vs 状态跃迁成本）
阶段5：应用滞回机制，连续投票后才执行区间调整
阶段6：非相邻实例通过"软分配"临时处理
阶段7：所有策略失败则 fail fast（拒绝请求）

关键参数：
- min_stable_interval: 最小稳定区间宽度（默认50 tokens）
- hysteresis_threshold: 滞回阈值（默认3次连续投票）
- adjustment_cooldown: 区间调整冷却时间（默认10秒）
- max_interval_change_rate: 单次最大变化幅度（默认100 tokens）
- prefill_transition_cost: prefill实例惩罚系数（默认0.3）
- decode_transition_cost: decode实例惩罚系数（默认0.6）
- soft_allocation_enabled: 是否允许软分配（默认True）

使用方法：
```python
# 配置参数
tracker.configure_interval_adjustment(
    hysteresis_threshold=5,
    prefill_transition_cost=0.25
)

# 调度请求
selected_instance = tracker.get_slo_aware_loaded_instance(
    instances=instance_list,
    input_tokens=100,
    expected_output_tokens=50
)

if selected_instance is None:
    # 请求被拒绝（fail fast）
    handle_rejection()

# 查看统计
stats = tracker.get_interval_statistics()
```

==================================================================================
"""

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from utils import random_uuid, generate_split_request_id
from instance import Instance, SLO, PerformanceProfile
import re


class AsyncLoadTracker:
    """异步负载追踪器,支持基于请求ID的乱序完成"""
    
    def __init__(self, profiler_file_path: str = None):
        self.instance_loads: Dict[str, Dict] = {}  # http_address -> load_info
        self.request_contexts: Dict[str, Dict] = {}  # request_id -> context
        self.zmq_to_http_mapping: Dict[str, str] = {}  # zmq_address -> http_address
        self.lock = threading.RLock()  # 使用可重入锁
        self.slo = SLO(TTFT=1000, TPOT=200)  # 统一使用小写 slo
        
        # 性能参数配置
        self.perf_profile = PerformanceProfile()
        if profiler_file_path:
            self.perf_profile.load_from_file(profiler_file_path)
        
        # 默认模型名称（可以通过set_model_name修改）
        self.model_name = None
        
        # -----------------------SLO感知调度-------------------------------
        # 动态区间管理相关状态（使用 instance.min_token 和 instance.max_token）
        self.interval_adjustment_history: Dict[str, List[Dict]] = {}  # http_address -> [last_change_time，old_interval,new_interval]
        self.last_interval_change: Dict[str, float] = {}  # http_address -> timestamp 上一次区间调整时间
        self.adjustment_votes: Dict[str, Dict] = {}  # http_address -> {direction: count} {"expand_left": 0, "expand_right": 0, "shrink": 0}
        
        # 区间调整参数（可配置）
        self.max_token_num=8192 #一个批次最大处理的token数量
        self.min_stable_interval = 50  # 最小稳定区间宽度（token数）
        self.hysteresis_threshold = 5  # 滞回阈值：连续N次相同方向才触发
        self.adjustment_cooldown = 0.2  # 区间调整冷却时间（秒）
        self.max_interval_change_rate = 500  # 单次最大区间变化幅度（token数）
        self.prefill_transition_cost = 0.8  # prefill实例状态跃迁惩罚系数(暂定)
        self.decode_transition_cost = 0.8  # decode实例状态跃迁惩罚系数(暂定后续调整)
        self.soft_allocation_penalty = 1  # 软分配惩罚系数（高于区间调整惩罚，避免过度使用软分配）
        self.soft_allocation_enabled = True  # 是否允许软分配（临时跨区间调度）
        #----------------------------------------------------------------
    
    def get_penalty_config(self) -> Dict[str, float]:
        """获取当前惩罚系数配置"""
        return {
            "prefill_transition_cost": self.prefill_transition_cost,
            "decode_transition_cost": self.decode_transition_cost,
            "soft_allocation_penalty": self.soft_allocation_penalty
        }
    
    def set_penalty_config(self, prefill: float = None, decode: float = None, soft: float = None) -> Dict[str, float]:
        """设置惩罚系数配置，返回更新后的配置"""
        if prefill is not None:
            self.prefill_transition_cost = prefill
        if decode is not None:
            self.decode_transition_cost = decode
        if soft is not None:
            self.soft_allocation_penalty = soft
        return self.get_penalty_config()

    def set_slo(self, ttft: float, tpot: float):
        """设置 SLO 目标"""
        with self.lock:
            self.slo = SLO(TTFT=ttft, TPOT=tpot)
            print(f"SLO updated: TTFT={ttft}s, TPOT={tpot}s")
    
    def get_slo(self) -> SLO:
        """获取当前 SLO"""
        with self.lock:
            return self.slo
    
    def load_performance_profile(self, file_path: str) -> bool:
        """加载或重新加载性能参数配置文件(暂未使用)
        
        Args:
            file_path: profiler JSON文件路径
            
        Returns:
            bool: 是否加载成功
        """
        return self.perf_profile.load_from_file(file_path)
    
    def set_model_name(self, model_name: str):
        """设置模型名称用于性能估算
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        #print(f"Model name set to: {model_name}")
    
    def get_model_name(self) -> Optional[str]:
        """获取当前模型名称(暂未使用)"""
        return self.model_name
    # -----------------------SLO感知调度-------------------------------
    def configure_interval_adjustment(self, 
                                     min_stable_interval: int = None,
                                     hysteresis_threshold: int = None,
                                     adjustment_cooldown: float = None,
                                     max_interval_change_rate: int = None,
                                     prefill_transition_cost: float = None,
                                     decode_transition_cost: float = None,
                                     soft_allocation_penalty: float = None,
                                     soft_allocation_enabled: bool = None):
        """配置动态区间调整参数
        
        Args:
            min_stable_interval: 最小稳定区间宽度（token数）
            hysteresis_threshold: 滞回阈值（连续投票次数）
            adjustment_cooldown: 区间调整冷却时间（秒）
            max_interval_change_rate: 单次最大区间变化幅度（token数）
            prefill_transition_cost: prefill实例状态跃迁惩罚系数
            decode_transition_cost: decode实例状态跃迁惩罚系数
            soft_allocation_penalty: 软分配惩罚系数
            soft_allocation_enabled: 是否允许软分配
        """
        # 设置SLO感知调度的相关参数
        with self.lock:
            if min_stable_interval is not None:
                self.min_stable_interval = min_stable_interval
            if hysteresis_threshold is not None:
                self.hysteresis_threshold = hysteresis_threshold
            if adjustment_cooldown is not None:
                self.adjustment_cooldown = adjustment_cooldown
            if max_interval_change_rate is not None:
                self.max_interval_change_rate = max_interval_change_rate
            if prefill_transition_cost is not None:
                self.prefill_transition_cost = prefill_transition_cost
            if decode_transition_cost is not None:
                self.decode_transition_cost = decode_transition_cost
            if soft_allocation_penalty is not None:
                self.soft_allocation_penalty = soft_allocation_penalty
            if soft_allocation_enabled is not None:
                self.soft_allocation_enabled = soft_allocation_enabled
            
            print(f"Interval adjustment config updated: "
                  f"min_stable={self.min_stable_interval}, "
                  f"hysteresis={self.hysteresis_threshold}, "
                  f"cooldown={self.adjustment_cooldown}s, "
                  f"max_change={self.max_interval_change_rate}")
    
    def get_interval_statistics(self) -> Dict:
        """获取区间调整统计信息(暂未使用)
        
        Returns:
            包含所有实例区间状态的字典
        """
        with self.lock:
            stats = {}
            for http_addr in self.interval_adjustment_history:
                stats[http_addr] = {
                    "last_change_time": self.last_interval_change.get(http_addr, 0),
                    "adjustment_votes": self.adjustment_votes.get(http_addr, {}),
                    "adjustment_count": len(self.interval_adjustment_history.get(http_addr, []))
                }
            return stats
    
    def get_interval_change_history(self) -> Dict:
        """获取所有实例的区间变化历史记录（测试接口）
        
        Returns:
            格式化的区间变化历史，格式为 {实例地址: [{new_interval, old_interval, timestamp}, ...]}
        """
        with self.lock:
            result = {
                "total_adjustments": 0,
                "instances": {}
            }
            
            for http_addr, history in self.interval_adjustment_history.items():
                if history:
                    result["instances"][http_addr] = {
                        "adjustment_count": len(history),
                        "changes": [
                            {
                                "old_interval": f"[{record['old_interval'][0]}, {record['old_interval'][1]}]",
                                "new_interval": f"[{record['new_interval'][0]}, {record['new_interval'][1]}]",
                                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record['timestamp']))
                            }
                            for record in history
                        ],
                        # 简化格式：实例:旧区间->新区间
                        "formatted": [
                            f"{http_addr}: [{record['old_interval'][0]},{record['old_interval'][1]}] -> [{record['new_interval'][0]},{record['new_interval'][1]}]"
                            for record in history
                        ]
                    }
                    result["total_adjustments"] += len(history)
            
            return result
    
    # -------------------------------------------------------------------------------------------
    def initialize_instance(self, http_address: str, zmq_address: str = None):
        """初始化实例负载记录"""
        with self.lock:
            if http_address not in self.instance_loads:
                self.instance_loads[http_address] = {
                    "total_requests": 0,    # 当前总请求数
                    "total_tokens": 0,      # 当前总token数
                    "total_processed_requests": 0,  # 历史处理请求总数
                    "total_processed_tokens": 0,    # 历史处理token总数
                    "last_update": time.time(),
                    "creation_time": time.time(),
                    "pending_requests": {}  # request_id -> request_info
                }
            
            # 建立zmq到http地址的映射
            # 维护一张zmq地址与http地址的映射表
            if zmq_address:
                self.zmq_to_http_mapping[zmq_address] = http_address

    def extract_instance_info_from_request_id(self, request_id: str) -> Dict[str, str]:
        """从请求ID中提取实例信息
        
        ID格式: ___prefill_addr_{prefill_zmq}_tp_{prefill_tp}___decode_addr_{decode_zmq}_tp_{decode_tp}_{uuid}
        
        Returns:
            Dict包含prefill_zmq, decode_zmq等信息
        """
        pattern = r"___prefill_addr_(.+?)_tp_(\d+)___decode_addr_(.+?)_tp_(\d+)_(.+)"
        match = re.match(pattern, request_id)
        
        if match:
            return {
                "prefill_zmq": match.group(1),
                "prefill_tp": match.group(2),
                "decode_zmq": match.group(3),
                "decode_tp": match.group(4),
                "uuid": match.group(5)
            }
        return {}

    def get_http_address_from_zmq(self, zmq_address: str) -> Optional[str]:
        """根据ZMQ地址获取HTTP地址"""
        return self.zmq_to_http_mapping.get(zmq_address)

    def add_request_split(self, prefill_instance, decode_instance, input_tokens: int, 
                         expected_output_tokens: int, request_type: str, shared_request_id: str = None) -> str:
        """添加分离式处理的请求负载"""
        
        # 确保实例已初始化并建立映射
        self.initialize_instance(prefill_instance.http_address, prefill_instance.zmq_address)
        self.initialize_instance(decode_instance.http_address, decode_instance.zmq_address)
        
        if request_type == "P":
            http_address = prefill_instance.http_address
            total_tokens = input_tokens
        elif request_type == "D":
            http_address = decode_instance.http_address
            total_tokens=expected_output_tokens - 1
        else:
            raise ValueError(f"Unknown request type: {request_type}")
        
        # 使用共享的request_id或生成新的
        if shared_request_id:
            request_id = shared_request_id
        else:
            request_id = generate_split_request_id(prefill_instance, decode_instance)
        
        # total_tokens = input_tokens + expected_output_tokens
        
        with self.lock:
            load_info = self.instance_loads[http_address]
            
            # 增加总请求数和token数
            load_info["total_requests"] += 1
            load_info["total_tokens"] += total_tokens
            load_info["last_update"] = time.time()
            
            # 存储请求信息
            load_info["pending_requests"][request_id] = {
                "input_tokens": input_tokens,
                "expected_output_tokens": expected_output_tokens,
                "total_tokens": total_tokens,
                "start_time": time.time(),
                "request_type": "split",
                "sub_type": request_type  # "P" or "D"
            }
            
            # 存储请求上下文（用于跟踪整个分离式请求）
            if request_id not in self.request_contexts:
                self.request_contexts[request_id] = {
                    "prefill_instance_http": prefill_instance.http_address,
                    "decode_instance_http": decode_instance.http_address,
                    "prefill_completed": False,
                    "decode_completed": False,
                    "start_time": time.time()
                }
            
        return request_id
    
    def complete_request_by_id(self, request_id: str, actual_output_tokens: Optional[int] = None) -> bool:
        """根据请求ID完成请求处理（异步方式）
        
        Args:
            request_id: 完整的请求ID
            actual_output_tokens: 实际输出token数
            
        Returns:
            bool: 是否成功处理
        """
        # 从请求ID中提取实例信息
        instance_info = self.extract_instance_info_from_request_id(request_id)
        if not instance_info:
            print(f"Warning: Cannot extract instance info from request_id: {request_id}")
            return False
        
        prefill_zmq = instance_info["prefill_zmq"]
        decode_zmq = instance_info["decode_zmq"]
        
        # 获取HTTP地址
        prefill_http = self.get_http_address_from_zmq(prefill_zmq)
        decode_http = self.get_http_address_from_zmq(decode_zmq)
        
        if not prefill_http or not decode_http:
            print(f"Warning: Cannot find HTTP addresses for zmq: {prefill_zmq}, {decode_zmq}")
            return False
        
        with self.lock:
            success = False
            
            # 处理prefill实例
            if prefill_http in self.instance_loads:
                prefill_load = self.instance_loads[prefill_http]
                if request_id in prefill_load["pending_requests"]:
                    request_info = prefill_load["pending_requests"][request_id]
                    
                    # 完成prefill请求（通常是1个token）
                    actual_tokens = 1
                    prefill_load["total_requests"] -= 1
                    prefill_load["total_tokens"] -= (request_info["input_tokens"] + actual_tokens)
                    prefill_load["total_processed_requests"] += 1
                    prefill_load["total_processed_tokens"] += (request_info["input_tokens"] + actual_tokens)
                    
                    del prefill_load["pending_requests"][request_id]
                    prefill_load["last_update"] = time.time()
                    
                    if request_id in self.request_contexts:
                        self.request_contexts[request_id]["prefill_completed"] = True
                    
                    success = True
                    print(f"Completed prefill request {request_id} for instance {prefill_http}")
            
            # 处理decode实例
            if decode_http in self.instance_loads:
                decode_load = self.instance_loads[decode_http]
                if request_id in decode_load["pending_requests"]:
                    request_info = decode_load["pending_requests"][request_id]
                    
                    # 完成decode请求
                    actual_decode_tokens = actual_output_tokens or request_info["expected_output_tokens"]
                    decode_load["total_requests"] -= 1
                    decode_load["total_tokens"] -= (request_info["input_tokens"] + actual_decode_tokens)
                    decode_load["total_processed_requests"] += 1
                    decode_load["total_processed_tokens"] += (request_info["input_tokens"] + actual_decode_tokens)
                    
                    del decode_load["pending_requests"][request_id]
                    decode_load["last_update"] = time.time()
                    
                    if request_id in self.request_contexts:
                        self.request_contexts[request_id]["decode_completed"] = True
                    
                    success = True
                    print(f"Completed decode request {request_id} for instance {decode_http}")
            
            # 清理已完成的请求上下文
            if request_id in self.request_contexts:
                context = self.request_contexts[request_id]
                if context["prefill_completed"] and context["decode_completed"]:
                    del self.request_contexts[request_id]
                    print(f"Cleaned up request context for {request_id}")
            
            return success
    
    def complete_prefill_request(self, http_address: str, request_id: str, 
                                  input_tokens: int = 0) -> bool:
        """处理prefill完成通知，仅更新prefill实例的负载状态
        
        当prefill实例完成计算并即将发送KV cache时调用此方法。
        与complete_request_by_id不同，此方法仅更新prefill实例的负载，
        不影响decode实例的负载状态。
        
        Args:
            http_address: prefill实例的HTTP地址
            request_id: 请求ID
            input_tokens: 输入token数（用于验证）
            
        Returns:
            bool: 是否成功处理
        """
        with self.lock:
            # 检查实例是否存在
            if http_address not in self.instance_loads:
                print(f"Warning: Instance {http_address} not found in load tracker")
                return False
            
            prefill_load = self.instance_loads[http_address]
            
            # 检查请求是否存在
            if request_id not in prefill_load["pending_requests"]:
                # 请求可能已经被处理过（通过complete_request_by_id）
                # 这种情况下静默返回成功
                return True
            
            request_info = prefill_load["pending_requests"][request_id]
            
            # 仅处理prefill类型的子请求
            if request_info.get("sub_type") != "P":
                print(f"Warning: Request {request_id} is not a prefill request")
                return False
            
            # 完成prefill请求
            actual_tokens = 1  # prefill输出1个token
            prefill_load["total_requests"] -= 1
            prefill_load["total_tokens"] -= (request_info["input_tokens"] + actual_tokens)
            prefill_load["total_processed_requests"] += 1
            prefill_load["total_processed_tokens"] += (request_info["input_tokens"] + actual_tokens)
            
            del prefill_load["pending_requests"][request_id]
            prefill_load["last_update"] = time.time()
            
            # 更新请求上下文
            if request_id in self.request_contexts:
                self.request_contexts[request_id]["prefill_completed"] = True
            
            print(f"Prefill complete (via notify): instance={http_address}, "
                  f"request_id={request_id}, input_tokens={input_tokens}")
            
            return True
    
    def fail_request_by_id(self, request_id: str) -> bool:
        """标记请求失败并清理负载
        
        Args:
            request_id: 完整的请求ID
            
        Returns:
            bool: 是否成功处理
        """
        # 从请求ID中提取实例信息
        instance_info = self.extract_instance_info_from_request_id(request_id)
        if not instance_info:
            return False
        
        prefill_zmq = instance_info["prefill_zmq"]
        decode_zmq = instance_info["decode_zmq"]
        
        # 获取HTTP地址
        prefill_http = self.get_http_address_from_zmq(prefill_zmq)
        decode_http = self.get_http_address_from_zmq(decode_zmq)
        
        if not prefill_http or not decode_http:
            return False
        
        with self.lock:
            success = False
            
            # 清理prefill实例负载
            if prefill_http in self.instance_loads:
                prefill_load = self.instance_loads[prefill_http]
                if request_id in prefill_load["pending_requests"]:
                    request_info = prefill_load["pending_requests"][request_id]
                    
                    prefill_load["total_requests"] -= 1
                    prefill_load["total_tokens"] -= request_info["total_tokens"]
                    del prefill_load["pending_requests"][request_id]
                    prefill_load["last_update"] = time.time()
                    success = True
            
            # 清理decode实例负载
            if decode_http in self.instance_loads:
                decode_load = self.instance_loads[decode_http]
                if request_id in decode_load["pending_requests"]:
                    request_info = decode_load["pending_requests"][request_id]
                    
                    decode_load["total_requests"] -= 1
                    decode_load["total_tokens"] -= request_info["total_tokens"]
                    del decode_load["pending_requests"][request_id]
                    decode_load["last_update"] = time.time()
                    success = True
            
            # 清理请求上下文
            if request_id in self.request_contexts:
                del self.request_contexts[request_id]
            
            return success
    
    def get_load(self, http_address: str) -> Optional[Dict]:
        """获取指定实例的负载信息"""
        with self.lock:
            return self.instance_loads.get(http_address)
    
    def get_all_loads(self) -> Dict[str, Dict]:
        """获取所有实例的负载信息(暂未使用)"""
        with self.lock:
            return self.instance_loads.copy()
    
    def remove_instance(self, http_address: str):
        """移除实例的负载记录"""
        with self.lock:
            self.instance_loads.pop(http_address, None)
            # 同时清理ZMQ映射
            zmq_to_remove = None
            for zmq_addr, http_addr in self.zmq_to_http_mapping.items():
                if http_addr == http_address:
                    zmq_to_remove = zmq_addr
                    break
            if zmq_to_remove:
                del self.zmq_to_http_mapping[zmq_to_remove]
    
    def estimate_processing_time(self, instance: Instance, input_tokens: int, expected_output_tokens: int, 
                                model_name: str = None) -> float:
        """预估请求处理时间（基于性能参数和当前负载）
        
        Args:
            instance: 实例对象（从中获取TP、类型等信息）
            input_tokens: 输入token数量
            expected_output_tokens: 预期输出token数量
            model_name: 模型名称（用于查询性能参数），如果未提供则使用self.model_name
            
        Returns:
            预估处理时间（毫秒），如果无法计算则返回一个基于负载的估算值
        """
        http_address = instance.http_address
        instance_type = instance.instance_type
        tp = instance.tp
        
        # 如果未提供model_name，使用实例变量中的model_name
        if model_name is None:
            model_name = self.model_name
        
        with self.lock:
            load_info = self.instance_loads.get(http_address)
            if not load_info:
                return float('inf')
            
            # 基础处理延迟估算
            base_latency = 0.0
            
            # 如果有性能配置文件且提供了模型名称，使用精确计算,计算当前请求的处理时间
            if model_name and self.perf_profile.has_profile(model_name, tp):
                if instance_type == "prefill":
                    # prefill实例：计算TTFT（首token延迟） 一批请求的时间
                    base_latency = self.perf_profile.get_prefill_latency(model_name, tp, input_tokens) or 0
                    
                elif instance_type == "decode":
                    # decode实例：计算TPOT（每token延迟）  单轮的时间
                    # decode阶段也需要考虑input_tokens的影响（KV cache）
                    # 假设decode的延迟公式中也包含了input的影响
                    decode_latency = self.perf_profile.get_decode_latency(model_name, tp, input_tokens, expected_output_tokens) or 0

                    # decode阶段需要考虑KV cache的input部分影响
                    # 这里可以根据实际情况调整，简化处理：decode延迟主要由output决定
                    base_latency = decode_latency               
            else:
                # 没有 profiler 配置或模型名称时报错
                if not model_name:
                    raise ValueError("Model name is required for performance estimation. "
                                   "Please set model name via set_model_name() or pass --model argument.")
                else:
                    raise ValueError(f"Performance profile not found for model '{model_name}' with TP={tp}. "
                                   "Please provide a valid profiler JSON file via --profiler argument.")

            # 计算排队延迟（基于连续批处理机制）
            # 考虑 max_token_num 限制，计算需要处理的批数
            queue_latency = 0.0
            pending_requests = load_info.get("pending_requests", {})
            
            if pending_requests:
                if instance_type == "prefill":
                    # prefill 实例：每批处理多个请求的 input_tokens
                    # 计算所有 pending 请求的总 input_tokens
                    total_pending_tokens = sum(
                        req_info.get("input_tokens", 0) 
                        for req_info in pending_requests.values()
                    )
                    # 计算需要等待的批数（向下取整）
                    # 原因：pending tokens 都在当前批次中处理，新请求只需等当前批次完成
                    # 例如：15000 tokens / 8192 = 1.83 → 等待 1 批（当前正在处理的批次）
                    num_batches = max(1, math.floor(total_pending_tokens / self.max_token_num)) if self.max_token_num > 0 else 1
                    # 每批的处理时间近似为 base_latency
                    queue_latency = num_batches * base_latency
                    
                elif instance_type == "decode":
                    # decode 实例：每轮迭代中，每个请求只生成 1 个 token
                    # 所以一批最多处理 max_token_num 个请求（每个请求 1 token）
                    num_pending_requests = len(pending_requests)
                    # 计算需要等待的批数（向下取整）
                    # 原因：pending 请求都在当前批次中，新请求只需等当前批次完成
                    num_batches = max(1, math.floor(num_pending_requests / self.max_token_num)) if self.max_token_num > 0 else 1
                    
                    # 每批的处理时间：一轮 decode 迭代的时间
                    # 但需要考虑每个请求还需要生成多少个 token
                    # 简化估算：取 pending 请求的平均剩余 output_tokens
                    total_remaining_tokens = sum(
                        req_info.get("expected_output_tokens", 0) 
                        for req_info in pending_requests.values()
                    )
                    avg_remaining_tokens = total_remaining_tokens / num_pending_requests if num_pending_requests > 0 else 0
                    
                    # 排队延迟 = 批数 × 平均剩余迭代轮数 × 单轮 decode 时间
                    # 单轮 decode 时间 ≈ base_latency / expected_output_tokens（如果 base_latency 是总 decode 时间）
                    # 这里 base_latency 已经是当前请求的总 decode 时间
                    single_iteration_latency = base_latency
                    queue_latency = num_batches * avg_remaining_tokens * single_iteration_latency
                else:
                    raise ValueError(f"Unknown instance type: {instance_type}")
            # 总时延=排队时延+请求处理时延(prefill阶段包含传输延迟)
            return max(0, base_latency + queue_latency)

    def get_least_loaded_instance(self, instances: List[Instance], input_tokens: int = 0, expected_output_tokens: int = 0) -> Optional[Instance]:
        "选取最小负载的实例"
        if not instances:
            return None
        
        min_estimated_time = float('inf')
        best_instance = None
        
        for instance in instances:
            # 确保实例已初始化
            self.initialize_instance(instance.http_address, instance.zmq_address)
            
            # 预估处理时间（传入实例对象而非http_address）
            estimated_time = self.estimate_processing_time(
                instance, input_tokens, expected_output_tokens
            )
            
            if estimated_time < min_estimated_time:
                min_estimated_time = estimated_time
                best_instance = instance
        
        return best_instance if best_instance else instances[0]


    #----------------------------------SLO感知调度------------------------------------------
    def _initialize_instance_interval(self, instance: Instance):
        """初始化实例的调整历史和投票记录（不修改区间）
        
        Args:
            instance: 实例对象
        """
        http_addr = instance.http_address
        
        # 仅初始化调整历史和投票记录
        if http_addr not in self.interval_adjustment_history:
            self.interval_adjustment_history[http_addr] = []
            self.last_interval_change[http_addr] = 0
            self.adjustment_votes[http_addr] = {"expand_left": 0, "expand_right": 0, "shrink": 0}
    
    def _check_slo_satisfaction(self, instance: Instance, input_tokens: int, expected_output_tokens: int, 
                               transition_penalty: float = 0.0, cached_time: float = None) -> Tuple[bool, float, float]:
        """检查实例是否满足SLO约束（包含跃迁成本惩罚）
        
        Args:
            instance: 实例对象
            input_tokens: 输入token数
            expected_output_tokens: 预期输出token数
            transition_penalty: 状态跃迁惩罚系数（0表示无跃迁）
            cached_time: 缓存的预估处理时间（避免重复计算）
            
        Returns:
            (是否满足SLO, 预估TTFT(含惩罚), 预估TPOT(含惩罚))
        """
        # 预估处理时间（优先使用缓存值）
        estimated_time = cached_time if cached_time is not None else self.estimate_processing_time(instance, input_tokens, expected_output_tokens)
        
        # 根据实例类型计算TTFT和TPOT
        if instance.instance_type == "prefill":
            # prefill实例主要影响TTFT
            estimated_ttft = estimated_time
            estimated_tpot = 0  # prefill不负责decode
        elif instance.instance_type == "decode":
            # decode实例主要影响TPOT
            estimated_ttft = 0  # decode不负责prefill
            estimated_tpot = estimated_time / expected_output_tokens if expected_output_tokens > 0 else 0
        else:  # general
            # general实例同时影响TTFT和TPOT
            estimated_ttft = estimated_time * 0.3  # 假设30%时间用于prefill
            estimated_tpot = (estimated_time * 0.7) / expected_output_tokens if expected_output_tokens > 0 else 0
        
        # 应用跃迁惩罚：惩罚 = 惩罚因子 * 计算所得延迟
        if transition_penalty > 0:
            estimated_ttft_with_penalty = estimated_ttft * (1 + transition_penalty)
            estimated_tpot_with_penalty = estimated_tpot * (1 + transition_penalty)
        else:
            estimated_ttft_with_penalty = estimated_ttft
            estimated_tpot_with_penalty = estimated_tpot
        
        # 检查是否满足SLO（使用统一的 self.slo）
        ttft_satisfied = estimated_ttft_with_penalty <= self.slo.TTFT
        tpot_satisfied = estimated_tpot_with_penalty <= self.slo.TPOT
        
        return (ttft_satisfied and tpot_satisfied, estimated_ttft_with_penalty, estimated_tpot_with_penalty)
    
    def _calculate_transition_cost(self, instance: Instance, new_min: int, new_max: int) -> float:
        """计算区间变化的状态跃迁成本（暂未使用，未来需要使用）
        
        Args:
            instance: 实例对象
            new_min: 新的最小token数
            new_max: 新的最大token数
            
        Returns:
            跃迁成本（无量纲，用于与性能收益比较）
        """
        current_min = instance.min_token
        current_max = instance.max_token
        
        if current_min == new_min and current_max == new_max:
            return 0.0
        
        # 计算区间变化幅度
        left_change = abs(new_min - current_min)
        right_change = abs(new_max - current_max)
        total_change = left_change + right_change
        
        # 根据实例类型选择惩罚系数
        if instance.instance_type == "prefill":
            penalty_factor = self.prefill_transition_cost
        elif instance.instance_type == "decode":
            penalty_factor = self.decode_transition_cost
        else:
            penalty_factor = (self.prefill_transition_cost + self.decode_transition_cost) / 2
        
        # 成本与变化幅度成正比(惩罚=惩罚因子*总改变长度)
        return penalty_factor * total_change
    
    def _check_adjustment_constraints(self, instance: Instance, proposed_min: int, proposed_max: int) -> bool:
        """检查区间调整是否满足约束条件
            1.最小区间 2.冷却时间 3. 单此变动最大幅度
        
        Args:
            instance: 实例对象
            proposed_min: 提议的新最小token数
            proposed_max: 提议的新最大token数
            
        Returns:
            是否允许调整
        """
        http_addr = instance.http_address
        current_time = time.time()
        
        # 1. 检查最小稳定区间约束
        interval_width = proposed_max - proposed_min
        if interval_width < self.min_stable_interval:
            return False
        
        # 2. 检查冷却时间约束
        last_change = self.last_interval_change.get(http_addr, 0)
        if current_time - last_change < self.adjustment_cooldown:
            return False
        
        # 3. 检查单次最大变化幅度约束
        left_change = abs(proposed_min - instance.min_token)
        right_change = abs(proposed_max - instance.max_token)
        if left_change > self.max_interval_change_rate or right_change > self.max_interval_change_rate:
            return False
        
        return True
    
    def _vote_for_adjustment(self, instance: Instance, direction: str) -> bool:
        """为区间调整投票，实现滞回机制
        
        滞回机制的设计原理：
        - 只有连续N次（hysteresis_threshold）相同方向的调整请求才会触发实际的区间变化
        - 当出现不同方向的请求时，重置其他方向的投票计数，避免边界震荡
        
        示例：
            请求序列: 右->右->左->右->右->右
            投票计数: 1->2->1(重置)->1->2->3 ✅触发
            
            如果不重置: 右3次+左1次都累计，会导致逻辑混乱
        
        Args:
            instance: 实例对象
            direction: 调整方向 ("expand_left", "expand_right", "shrink")
            
        Returns:
            是否达到调整阈值
        """
        http_addr = instance.http_address
        
        # 重置其他方向的投票
        # 原因：确保只有"连续相同方向"的调整才能累积投票
        # 防止请求长度在区间边界附近来回波动时造成频繁调整
        for d in self.adjustment_votes[http_addr]:
            if d != direction:
                self.adjustment_votes[http_addr][d] = 0
        
        # 当前方向投票+1
        self.adjustment_votes[http_addr][direction] += 1
        
        # 检查是否达到滞回阈值
        return self.adjustment_votes[http_addr][direction] >= self.hysteresis_threshold
    
    def _apply_interval_adjustment(self, instance: Instance, new_min: int, new_max: int):
        """应用区间调整
        
        Args:
            instance: 实例对象
            new_min: 新的最小token数
            new_max: 新的最大token数
        """
        http_addr = instance.http_address
        old_interval = (instance.min_token, instance.max_token)
        
        # 更新实例的区间
        instance.min_token = new_min
        instance.max_token = new_max
        self.last_interval_change[http_addr] = time.time()
        
        # 记录调整历史
        self.interval_adjustment_history[http_addr].append({
            "timestamp": time.time(),
            "old_interval": old_interval,
            "new_interval": (new_min, new_max)
        })
        
        # 重置投票
        self.adjustment_votes[http_addr] = {"expand_left": 0, "expand_right": 0, "shrink": 0}
        
        print(f"Interval adjusted for {http_addr}: {old_interval} -> ({new_min}, {new_max})")
    
    def get_slo_aware_loaded_instance(self, instances: List[Instance], input_tokens: int = 0, 
                                     expected_output_tokens: int = 0) -> Optional[Instance]:
        """动态负载适应区间变化的SLO感知调度器（基于输入长度分类）
        
        实现动态实例区间自适应调整，包含滞回机制、速率限制和状态跃迁成本建模。
        
        改进点：
        1. 请求按 input_tokens 分类（output_tokens 仅用于估算处理时间）
        2. 阶段1可能找到多个负责实例
        3. 跃迁成本体现为对 TTFT/TPOT 的惩罚（惩罚因子 * 延迟）
        4. 区间连续性：相邻实例区间满足 current.max_token + 1 = next.min_token
        
        Args:
            instances: 候选实例列表
            input_tokens: 输入token数量（用于分类）
            expected_output_tokens: 预期输出token数量（用于估算处理时间）
            
        Returns:
            选中的实例，如果所有实例都不满足SLO则返回None（fail fast）
        """
        if not instances:
            return None
        
        with self.lock:
            # 使用 input_tokens 作为请求分类依据
            request_length = input_tokens
            
            # ============ 预计算：一次性计算所有实例的处理时间和惩罚系数 ============
            instance_cache = {}  # http_address -> (estimated_time, penalty)
            for instance in instances:
                self.initialize_instance(instance.http_address, instance.zmq_address)
                self._initialize_instance_interval(instance)
                
                # 预计算处理时间
                estimated_time = self.estimate_processing_time(instance, input_tokens, expected_output_tokens)
                
                # 预计算惩罚系数
                if instance.instance_type == "prefill":
                    penalty = self.prefill_transition_cost
                elif instance.instance_type == "decode":
                    penalty = self.decode_transition_cost
                else:
                    penalty = (self.prefill_transition_cost + self.decode_transition_cost) / 2
                
                instance_cache[instance.http_address] = (estimated_time, penalty)
            
            # ============ 阶段1：查找当前负责该请求长度区间的所有实例 ============
            responsible_instances = []
            for instance in instances:
                if instance.min_token <= request_length <= instance.max_token:
                    responsible_instances.append(instance)
            
            # ============ 阶段2：如果找到负责实例，检查是否有满足SLO的 ============
            if responsible_instances:
                # 检查所有负责实例，选择满足SLO且负载最低的
                best_responsible = None
                best_responsible_score = float('inf')
                
                for instance in responsible_instances:
                    cached_time, _ = instance_cache[instance.http_address]
                    satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                        instance, input_tokens, expected_output_tokens, 
                        transition_penalty=0.0, cached_time=cached_time
                    )
                    
                    if satisfies_slo:
                        # 综合评分：TTFT + TPOT（越小越好）
                        score = ttft + tpot
                        if score < best_responsible_score:
                            best_responsible_score = score
                            best_responsible = instance
                
                if best_responsible:
                    # 找到满足SLO的负责实例，直接返回
                    return best_responsible
            
            # ============ 阶段3：负责实例不满足SLO或不存在，搜索负责实例的相邻区间 ============
            # 相邻区间定义：基于负责实例的区间边界
            # - 左相邻：candidate.max_token + 1 == responsible.min_token
            # - 右相邻：candidate.min_token - 1 == responsible.max_token
            left_adjacent_candidates = []   # 左侧相邻实例
            right_adjacent_candidates = []  # 右侧相邻实例
            non_adjacent_candidates = []    # 非相邻实例
            
            # 如果有负责实例，找它们的相邻实例
            if responsible_instances:
                for instance in instances:
                    if instance in responsible_instances:
                        continue  # 跳过负责实例自身
                    
                    # 检查是否与任一负责实例相邻
                    is_left_adjacent = False
                    is_right_adjacent = False
                    
                    for resp_instance in responsible_instances:
                        # 判断是否为左侧相邻（candidate在左，responsible在右）
                        if instance.max_token + 1 == resp_instance.min_token:
                            is_left_adjacent = True
                            break
                        # 判断是否为右侧相邻（candidate在右，responsible在左）
                        elif instance.min_token - 1 == resp_instance.max_token:
                            is_right_adjacent = True
                            break
                    
                    if is_left_adjacent:
                        # 左侧相邻实例向右扩张（使用缓存的处理时间和惩罚系数）
                        cached_time, penalty = instance_cache[instance.http_address]
                        
                        satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                            instance, input_tokens, expected_output_tokens, 
                            transition_penalty=penalty, cached_time=cached_time
                        )
                        
                        if satisfies_slo:
                            left_adjacent_candidates.append((instance, ttft, tpot, penalty, "expand_right"))
                    
                    elif is_right_adjacent:
                        # 右侧相邻实例向左扩张（使用缓存的处理时间和惩罚系数）
                        cached_time, penalty = instance_cache[instance.http_address]
                        
                        satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                            instance, input_tokens, expected_output_tokens, 
                            transition_penalty=penalty, cached_time=cached_time
                        )
                        
                        if satisfies_slo:
                            right_adjacent_candidates.append((instance, ttft, tpot, penalty, "expand_left"))
                    
                    else:
                        # 非相邻实例（用于软分配，使用缓存的处理时间）
                        cached_time, _ = instance_cache[instance.http_address]
                        satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                            instance, input_tokens, expected_output_tokens, 
                            transition_penalty=self.soft_allocation_penalty, cached_time=cached_time
                        )
                        
                        if satisfies_slo:
                            non_adjacent_candidates.append((instance, ttft, tpot))
            
            else:
                # 没有负责实例的情况：请求落在区间空隙中
                # 只有最接近请求长度的两端实例才能扩张（左侧最近和右侧最近）
                # 中间的实例只能软分配
                
                # 找出左侧最接近的实例（max_token < request_length 中最大的）
                left_closest_instance = None
                left_closest_distance = float('inf')
                
                # 找出右侧最接近的实例（min_token > request_length 中最小的）
                right_closest_instance = None
                right_closest_distance = float('inf')
                
                for instance in instances:
                    if instance.max_token < request_length:
                        # 左侧实例
                        distance = request_length - instance.max_token
                        if distance < left_closest_distance:
                            left_closest_distance = distance
                            left_closest_instance = instance
                    elif instance.min_token > request_length:
                        # 右侧实例
                        distance = instance.min_token - request_length
                        if distance < right_closest_distance:
                            right_closest_distance = distance
                            right_closest_instance = instance
                
                # 对所有实例进行分类（使用缓存的处理时间和惩罚系数）
                for instance in instances:
                    cached_time, penalty = instance_cache[instance.http_address]
                    
                    # 判断是否为两端的最接近实例
                    is_edge_instance = (instance == left_closest_instance or instance == right_closest_instance)
                    
                    if is_edge_instance:
                        # 两端实例：可以扩张
                        satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                            instance, input_tokens, expected_output_tokens, 
                            transition_penalty=penalty, cached_time=cached_time
                        )
                        
                        if satisfies_slo:
                            if instance == left_closest_instance:
                                # 左侧最近实例向右扩张
                                left_adjacent_candidates.append((instance, ttft, tpot, penalty, "expand_right"))
                            else:  # instance == right_closest_instance
                                # 右侧最近实例向左扩张
                                right_adjacent_candidates.append((instance, ttft, tpot, penalty, "expand_left"))
                    else:
                        # 中间实例：只能软分配（应用软分配惩罚，使用缓存的处理时间）
                        satisfies_slo, ttft, tpot = self._check_slo_satisfaction(
                            instance, input_tokens, expected_output_tokens, 
                            transition_penalty=self.soft_allocation_penalty, cached_time=cached_time
                        )
                        
                        if satisfies_slo:
                            non_adjacent_candidates.append((instance, ttft, tpot))
            
            # 合并所有相邻候选
            adjacent_candidates = left_adjacent_candidates + right_adjacent_candidates
            
            # ============ 阶段4：从相邻候选中选择最优实例 ============
            if adjacent_candidates:
                best_candidate = None
                best_score = float('inf')
                best_candidate_without_adjustment = None
                best_score_without_adjustment = float('inf')
                
                for instance, ttft, tpot, penalty, direction in adjacent_candidates:
                    # 计算新区间
                    if direction == "expand_right":
                        proposed_min = instance.min_token
                        proposed_max = request_length
                    else:  # expand_left
                        proposed_min = request_length
                        proposed_max = instance.max_token
                    
                    # 综合评分：带惩罚的延迟（越小越好）
                    score = ttft + tpot
                    
                    # 检查是否满足调整约束
                    can_adjust = self._check_adjustment_constraints(instance, proposed_min, proposed_max)
                    
                    if score < best_score:
                        if can_adjust:
                            # 可以调整区间的最优候选
                            best_score = score
                            best_candidate = (instance, proposed_min, proposed_max, direction)
                        elif score < best_score_without_adjustment:
                            # 不能调整区间但性能更优的候选（作为备选）
                            best_score_without_adjustment = score
                            best_candidate_without_adjustment = instance
                
                # 决策逻辑：优先选择可以调整区间的候选
                if best_candidate:
                    instance, new_min, new_max, direction = best_candidate
                    
                    # 实施滞回机制：需要连续投票
                    if self._vote_for_adjustment(instance, direction):
                        # 达到滞回阈值，应用区间调整
                        self._apply_interval_adjustment(instance, new_min, new_max)
                        
                        # 如果有负责实例，收缩其区间
                        if responsible_instances:
                            for resp_instance in responsible_instances:
                                if direction == "expand_right":
                                    # 左侧实例向右扩张，负责实例向右收缩
                                    new_resp_min = request_length + 1
                                    new_resp_max = resp_instance.max_token
                                else:  # expand_left
                                    # 右侧实例向左扩张，负责实例向左收缩
                                    new_resp_min = resp_instance.min_token
                                    new_resp_max = request_length - 1
                                
                                # 检查收缩后区间是否合法
                                if new_resp_max >= new_resp_min and (new_resp_max - new_resp_min) >= self.min_stable_interval:
                                    self._apply_interval_adjustment(resp_instance, new_resp_min, new_resp_max)
                    
                    return instance
                
                elif best_candidate_without_adjustment:
                    # 没有可以调整区间的候选，但有不需要调整就满足条件的候选
                    print(f"Selected adjacent instance without interval adjustment (constraints not met): {best_candidate_without_adjustment.http_address}")
                    return best_candidate_without_adjustment
            
            # ============ 阶段5：相邻区间无法满足，考虑非相邻区间（软分配） ============
            if non_adjacent_candidates and self.soft_allocation_enabled:
                # 软分配：临时调度但不改变区间
                # 选择性能最优的实例
                best_instance = min(non_adjacent_candidates, key=lambda x: x[1] + x[2])
                print(f"Soft allocation: input_tokens={input_tokens} assigned to non-adjacent instance {best_instance[0].http_address}")
                return best_instance[0]
            
            # ============ 阶段6：所有策略均失败，fail fast ============
            print(f"Fail fast: No instance can satisfy SLO for request (input={input_tokens}, output={expected_output_tokens})")
            return None