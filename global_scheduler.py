# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from asyncio.log import logger
import json
import os
import random
import socket
import sys
import threading
import traceback
import uuid
from typing import Dict, List, Optional
from utils import generate_split_request_id,forward_request
import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request, Response, jsonify
from instance import Instance
from async_load_tracker import AsyncLoadTracker


class GlobalScheduler:
    """全局调度器"""
    
    def __init__(self, profiler_file_path: str = None, model_name: str = None):
        self.instances: Dict[str, Instance] = {}  # http_address -> Instance
        self.prefill_instances: Dict[str, Instance] = {}  # prefill实例池
        self.decode_instances: Dict[str, Instance] = {}  # decode实例池
        
        self.instances_lock = threading.Lock()
        # 创建AsyncLoadTracker时传入性能配置文件路径
        self.load_tracker = AsyncLoadTracker(profiler_file_path=profiler_file_path)
        
        # 保存模型名称，用于性能估算
        self.model_name = model_name
        
        # 轮询索引
        self.round_robin_index = 0
        self.round_robin_lock = threading.Lock()
        
        # ZMQ相关
        self.context = zmq.Context()
        self.router_socket = None
        
    def add_instance(self, instance: Instance):
        """添加实例,根据实例类型将实例添加到指定池中（仅支持prefill和decode）"""
        with self.instances_lock:
            self.instances[instance.http_address] = instance
            
            # 根据类型分类存储（PD分离场景）
            if instance.instance_type == "prefill":
                self.prefill_instances[instance.http_address] = instance
            elif instance.instance_type == "decode":
                self.decode_instances[instance.http_address] = instance
            else:
                # PD分离场景不支持general类型
                logger.warning(f"Unsupported instance type '{instance.instance_type}' for PD separation scenario. Instance not added to type-specific pool.")
                return

            # 初始化负载追踪（传递zmq地址以建立映射）
            self.load_tracker.initialize_instance(instance.http_address, instance.zmq_address)
            logger.info(f"Added instance: {instance} to {instance.instance_type} pool")

    def remove_instance(self, http_address: str):
        """移除实例,根据http地址将实例从各个池中移除"""
        with self.instances_lock:
            instance = self.instances.pop(http_address, None)
            if instance:
                # 从对应的类型池中移除
                self.prefill_instances.pop(http_address, None)
                self.decode_instances.pop(http_address, None)
                self.load_tracker.remove_instance(http_address)
                logger.info(f"Removed instance: {http_address}")
    
    def get_suitable_instances(self, token_length: int, instance_type: str = None) -> List[Instance]:
        """根据token长度和类型(或者单类型)获取合适的实例队列"""
        with self.instances_lock:
            suitable_instances = []
            
            # 选择实例池
            if instance_type == "prefill":
                pool = self.prefill_instances
            elif instance_type == "decode":
                pool = self.decode_instances
            else:
                pool = self.instances
            
            for instance in pool.values():
                # 此处用于提前选择合适的实例(round_bin,random)
                #if instance.min_token <= token_length <= instance.max_token:
                    # suitable_instances.append(instance)
                suitable_instances.append(instance)
            
            return suitable_instances
    
    def select_instance(self, token_length: int, max_tokens: int = 1, strategy: str = "round_robin", 
                       instance_type: str = None) -> Optional[Instance]:
        """选择实例的策略"""
        # 获取实例
        suitable_instances = self.get_suitable_instances(token_length, instance_type)
        
        if not suitable_instances:
            # 如果没有合适的实例，从全部实例中选择
            with self.instances_lock:
                suitable_instances = list(self.instances.values())
        
        if not suitable_instances:
            return None
        # 轮询
        if strategy == "round_robin":
            with self.round_robin_lock:
                index = self.round_robin_index % len(suitable_instances)
                self.round_robin_index += 1
                return suitable_instances[index]
        # 随机
        elif strategy == "random":
            return random.choice(suitable_instances)
        # 最小负载（基于预估处理时间）
        elif strategy == "least_loaded":
            return self.load_tracker.get_least_loaded_instance(suitable_instances, token_length, max_tokens)
        # 基于SLO敏感的实例选择策略
        elif strategy == "slo_aware":
            return self.load_tracker.get_slo_aware_loaded_instance(suitable_instances, token_length, max_tokens)
        else:
            return suitable_instances[0]
    
    def _listen_for_register(self, poller, router_socket):
        """监听实例注册和ping消息"""
        while True:
            socks = dict(poller.poll())
            if router_socket in socks:
                try:
                    remote_address, message = router_socket.recv_multipart()
                    data = msgpack.loads(message)
                    
                    message_type = data.get("type")
                    
                    if message_type == "Ping":
                        # 处理ping消息，用于接收prefill实例发来的负载
                        self._handle_ping_message(data)
                    
                    elif message_type in ["P", "D"]:
                        # 处理prefill/decode实例注册（来自multi_local_scheduler.py的逻辑）
                        self._handle_instance_registration(data)
                    
                    elif message_type == "PREFILL_COMPLETE":
                        # 处理prefill完成通知，用于降低prefill实例的负载
                        self._handle_prefill_complete(data)
                    
                    else:
                        logger.warning(f"Unknown message type: {message_type}, data: {data}")
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
    
    def _handle_ping_message(self, data):
        """处理ping消息（PD分离场景暂未使用）"""
        # PD分离场景下，实例通过注册消息（type="P"或"D"）加入
        # Ping机制暂时保留但不实现
        pass
    
    def _handle_prefill_complete(self, data):
        """处理prefill完成通知，降低prefill实例的负载
        
        当prefill实例完成计算并即将发送KV cache到decode实例时，
        会发送此通知以便调度器及时更新负载状态。
        
        Args:
            data: 包含以下字段的字典:
                - http_address: prefill实例的HTTP地址
                - zmq_address: prefill实例的ZMQ地址
                - request_id: 请求ID
                - input_tokens: 输入token数
                - timestamp: 通知发送时间戳
        """
        try:
            http_address = data.get("http_address")
            request_id = data.get("request_id")
            input_tokens = data.get("input_tokens", 0)
            
            if not http_address or not request_id:
                logger.warning("Invalid PREFILL_COMPLETE message: missing http_address or request_id")
                return
            
            # 调用load_tracker完成prefill请求的负载更新
            success = self.load_tracker.complete_prefill_request(
                http_address, request_id, input_tokens
            )
            
            if success:
                logger.debug(f"Prefill complete notification processed: "
                           f"instance={http_address}, request_id={request_id}, "
                           f"input_tokens={input_tokens}")
            else:
                logger.warning(f"Failed to process prefill complete: "
                             f"instance={http_address}, request_id={request_id}")
                
        except Exception as e:
            logger.error(f"Error handling PREFILL_COMPLETE message: {e}")
    
    def _handle_instance_registration(self, data):
        """处理实例注册消息"""
        http_address = data.get("http_address")
        zmq_address = data.get("zmq_address")
        instance_type = "prefill" if data.get("type") == "P" else "decode"
        
        if not all([http_address, zmq_address]):
            return
        
        # 如果实例已存在，跳过注册（避免覆盖动态调整后的区间）
        with self.instances_lock:
            if http_address in self.instances:
                # 实例已存在，仅更新最后ping时间
                self.instances[http_address].last_ping_time = __import__('time').time()
                return
        
        # 安全转换 tp
        try:
            tp = int(data.get("tp", 1) or 1)
        except (ValueError, TypeError):
            tp = 1
        
        # 安全转换 min_token
        try:
            min_token = int(data.get("min_token", 0) or 0)
        except (ValueError, TypeError):
            min_token = 0
        
        # 安全转换 max_token
        max_token_raw = data.get("max_token", float('inf'))
        if not max_token_raw or max_token_raw == 'inf' or max_token_raw == float('inf'):
            max_token = float('inf')
        else:
            try:
                max_token = int(max_token_raw)
            except (ValueError, TypeError):
                max_token = float('inf')
        
        instance = Instance(
            http_address=http_address,
            zmq_address=zmq_address,
            tp=tp,
            instance_type=instance_type,
            min_token=min_token,
            max_token=max_token
        )
        
        self.add_instance(instance)
        logger.info(f"New instance registered: {http_address}, type={instance_type}, interval=[{min_token}, {max_token}]")
    
    async def handle_slo_update(self):
        """处理 SLO 的统一 HTTP 接口(支持GET查询和PUT/POST更新)"""
        try:
            # GET 请求：查询当前 SLO
            if request.method == 'GET':
                slo = self.load_tracker.get_slo()
                return jsonify({
                    "slo": {
                        "ttft_ms": slo.TTFT,
                        "tpot_ms": slo.TPOT
                    }
                }), 200
            
            # PUT/POST 请求：更新 SLO
            request_content = await request.get_data()
            try:
                data = json.loads(request_content)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format"}), 400
            
            ttft_ms = data.get("ttft_ms")
            tpot_ms = data.get("tpot_ms")
            
            # 也支持字符串格式 "ttft,tpot"
            if ttft_ms is None and tpot_ms is None:
                slo_str = data.get("slo")
                if slo_str:
                    try:
                        parts = slo_str.split(',')
                        if len(parts) == 2:
                            ttft_ms = float(parts[0])
                            tpot_ms = float(parts[1])
                    except (ValueError, IndexError):
                        pass
            
            if ttft_ms is None or tpot_ms is None:
                return jsonify({
                    "error": "Missing SLO parameters. Provide 'ttft_ms' and 'tpot_ms', or 'slo' in format 'TTFT,TPOT'"
                }), 400
            
            try:
                ttft_ms = float(ttft_ms)
                tpot_ms = float(tpot_ms)
                
                if ttft_ms <= 0 or tpot_ms <= 0:
                    return jsonify({
                        "error": "SLO values must be positive"
                    }), 400
                
            except ValueError:
                return jsonify({
                    "error": "Invalid SLO values. Must be numbers"
                }), 400
            
            # 更新 SLO
            old_slo = self.load_tracker.get_slo()
            self.load_tracker.set_slo(ttft_ms, tpot_ms)
            
            return jsonify({
                "message": "SLO updated successfully",
                "old_slo": {
                    "ttft_ms": old_slo.TTFT,
                    "tpot_ms": old_slo.TPOT
                },
                "new_slo": {
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error handling SLO request: {e}")
            return jsonify({
                "error": f"Internal server error: {str(e)}"
            }), 500
    
    async def handle_penalty_config(self):
        """处理惩罚系数配置的HTTP接口（支持GET查询和PUT/POST更新）"""
        try:
            # GET: 查询当前配置
            if request.method == 'GET':
                return jsonify(self.load_tracker.get_penalty_config()), 200
            
            # PUT/POST: 更新配置
            data = json.loads(await request.get_data())
            old_config = self.load_tracker.get_penalty_config()
            
            new_config = self.load_tracker.set_penalty_config(
                prefill=data.get("prefill_transition_cost"),
                decode=data.get("decode_transition_cost"),
                soft=data.get("soft_allocation_penalty")
            )
            
            return jsonify({
                "message": "Penalty config updated",
                "old": old_config,
                "new": new_config
            }), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def start_service_discovery(self, hostname: str, port: int):
        """启动服务发现"""
        if not hostname:
            hostname = socket.gethostname()
        if port == 0:
            raise ValueError("Port cannot be 0")
        
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{hostname}:{port}")
        
        poller = zmq.Poller()
        poller.register(self.router_socket, zmq.POLLIN)
        
        listener_thread = threading.Thread(
            target=self._listen_for_register,
            args=[poller, self.router_socket],
            daemon=True
        )
        listener_thread.start()
        return listener_thread
    
    async def handle_request(self):
        """处理HTTP请求"""
        try:
            headers = dict(request.headers)
            headers.pop("Host", None)
            headers.pop("Content-Length", None)
            
            request_content = await request.get_data()
            try:
                request_data = json.loads(request_content)
            except json.JSONDecodeError:
                return Response("Invalid JSON", status=400)
            
            original_request_data = request_data.copy()
            # 获取prompt长度和最大输出长度
            prompt_len = request_data.get("prompt_len", 0)
            max_tokens = request_data.get("max_tokens", 1) 
            
            # 检查是否有分离的prefill和decode实例，并获取处于其区间的实例
            prefill_instances = self.get_suitable_instances(prompt_len, "prefill")
            decode_instances = self.get_suitable_instances(prompt_len, "decode")
            
            # 检查是否可以使用分离式处理
            if not prefill_instances:
                return Response(
                    "No prefill instances available for split processing",
                    status=503
                )
            
            if not decode_instances:
                return Response(
                    "No decode instances available for split processing", 
                    status=503
                )
            
            # 选择prefill和decode实例
            # round_robin least_loaded random slo_aware
            # 此处可修改策略
            prefill_instance = self.select_instance(
                prompt_len, max_tokens, strategy="slo_aware", instance_type="prefill"
            )
            decode_instance = self.select_instance(
                prompt_len, max_tokens, strategy="slo_aware", instance_type="decode"
            )
            
            if not prefill_instance:
                return Response(
                    "Unable to select suitable prefill instance",
                    status=503
                )
            
            if not decode_instance:
                return Response(
                    "Unable to select suitable decode instance",
                    status=503
                )
            
            print(f"Using split processing: prefill={prefill_instance.http_address}, decode={decode_instance.http_address}")
            
            # 生成统一的请求ID
            shared_request_id = generate_split_request_id(prefill_instance, decode_instance)
            print(f"Generated shared request ID: {shared_request_id}")
            
            # 记录负载
            prefill_request_id = self.load_tracker.add_request_split(
                prefill_instance, decode_instance, prompt_len, 1, "P", shared_request_id
            )
            decode_request_id = self.load_tracker.add_request_split(
                prefill_instance, decode_instance, prompt_len, max_tokens - 1, "D", shared_request_id
            )
            #print(request_data)
            # 创建prefill请求（max_tokens = 1）
            prefill_request = request_data.copy()
            prefill_request['max_tokens'] = 1
            
            # 执行prefill阶段
            async for _ in forward_request(f'http://{prefill_instance.http_address}/v1/completions',
                                       prefill_request, shared_request_id):
                continue

            # 包装decode生成器，在完成时清除负载
            async def decode_with_cleanup():
                try:
                    async for chunk in forward_request(f'http://{decode_instance.http_address}/v1/completions',
                                                original_request_data, shared_request_id):
                        yield chunk
                    # decode完成，清除负载
                    self.load_tracker.complete_request_by_id(shared_request_id)
                except Exception:
                    # 失败时也清除负载
                    self.load_tracker.fail_request_by_id(shared_request_id)
                    raise

            # 返回响应
            response = await make_response(decode_with_cleanup())
            response.timeout = None
            return response

        except asyncio.TimeoutError:
            # 超时时清除负载
            if 'shared_request_id' in locals():
                self.load_tracker.fail_request_by_id(shared_request_id)
            return Response("Upstream service timeout", status=504)
        except Exception as e:
            # 异常时清除负载
            if 'shared_request_id' in locals():
                self.load_tracker.fail_request_by_id(shared_request_id)
            import traceback
            traceback.print_exc()
            return Response(
                f"Proxy Error: {str(e)}",
                status=500
            )
    
    async def handle_remove_instance(self, http_address: str = None):
        """手动移除实例的HTTP接口"""
        try:
            # 如果URL中没有http_address参数，尝试从请求体获取
            if not http_address:
                request_content = await request.get_data()
                if request_content:
                    try:
                        data = json.loads(request_content)
                        http_address = data.get("http_address")
                    except json.JSONDecodeError:
                        return jsonify(
                            {"error": "Invalid JSON format"}
                        ), 400
            
            if not http_address:
                return jsonify(
                    {"error": "Missing http_address. Provide it in URL path or request body"}
                ), 400
            
            # URL解码（处理特殊字符如冒号）
            from urllib.parse import unquote
            http_address = unquote(http_address)
            
            # 检查实例是否存在
            with self.instances_lock:
                if http_address not in self.instances:
                    return jsonify(
                        {"error": f"Instance with http_address {http_address} not found"}
                    ), 404
                
                # 获取实例信息用于响应
                instance = self.instances[http_address]
                instance_info = {
                    "http_address": instance.http_address,
                    "zmq_address": instance.zmq_address,
                    "tp": instance.tp,
                    "min_token": instance.min_token,
                    "max_token": instance.max_token,
                    "type": instance.instance_type
                }
            
            # 移除实例
            self.remove_instance(http_address)
            
            return jsonify(
                {
                    "message": "Instance removed successfully",
                    "removed_instance": instance_info
                }
            ), 200
            
        except Exception as e:
            logger.error(f"Error removing instance: {e}")
            return jsonify(
                {"error": f"Internal server error: {str(e)}"}
            ), 500
    
    async def handle_get_instance(self, http_address: str):
        """获取单个实例详情的HTTP接口"""
        try:
            # URL解码
            from urllib.parse import unquote
            http_address = unquote(http_address)
            
            with self.instances_lock:
                instance = self.instances.get(http_address)
                if not instance:
                    return jsonify(
                        {"error": f"Instance with http_address {http_address} not found"}
                    ), 404
                
                # 获取指定实例的负载信息
                load_info = self.load_tracker.get_load(http_address) or {}
                # 将实例信息以json格式返回给用户
                return jsonify({
                    "instance": {
                        "http_address": instance.http_address,
                        "zmq_address": instance.zmq_address,
                        "tp": instance.tp,
                        "min_token": instance.min_token,
                        "max_token": instance.max_token,
                        "type": instance.instance_type,
                        "last_ping_time": instance.last_ping_time
                    },
                    "load_info": load_info
                })
                
        except Exception as e:
            logger.error(f"Error getting instance: {e}")
            return jsonify(
                {"error": f"Internal server error: {str(e)}"}
            ), 500
    
    async def handle_get_instances_by_type(self):
        """获取按类型分类的实例信息"""
        try:
            with self.instances_lock:
                return jsonify({
                    "summary": {
                        "total_instances": len(self.instances),
                        "prefill_instances": len(self.prefill_instances),
                        "decode_instances": len(self.decode_instances)
                    },
                    "prefill_instances": [
                        {
                            "http_address": inst.http_address,
                            "zmq_address": inst.zmq_address,
                            "tp": inst.tp,
                            "min_token": inst.min_token,
                            "max_token": inst.max_token,
                            "type": inst.instance_type
                        }
                        for inst in self.prefill_instances.values()
                    ],
                    "decode_instances": [
                        {
                            "http_address": inst.http_address,
                            "zmq_address": inst.zmq_address,
                            "tp": inst.tp,
                            "min_token": inst.min_token,
                            "max_token": inst.max_token,
                            "type": inst.instance_type
                        }
                        for inst in self.decode_instances.values()
                    ]
                })
        except Exception as e:
            logger.error(f"Error getting instances by type: {e}")
            return jsonify(
                {"error": f"Internal server error: {str(e)}"}
            ), 500
    
    async def handle_get_interval_history(self):
        """获取所有实例的区间变化历史记录（测试接口）
        
        返回格式：实例:旧区间->新区间
        """
        try:
            history = self.load_tracker.get_interval_change_history()
            
            # 添加当前各实例的区间状态
            with self.instances_lock:
                current_intervals = {}
                for http_addr, inst in self.instances.items():
                    current_intervals[http_addr] = {
                        "current_interval": f"[{inst.min_token}, {inst.max_token}]",
                        "type": inst.instance_type
                    }
                history["current_intervals"] = current_intervals
            
            return jsonify(history), 200
            
        except Exception as e:
            logger.error(f"Error getting interval history: {e}")
            return jsonify(
                {"error": f"Internal server error: {str(e)}"}
            ), 500

def main():
    parser = argparse.ArgumentParser(description="Global Scheduler")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--discovery-port", type=int, default=30001, help="Service discovery port")
    parser.add_argument("--slo", type=str, default="1000,100", 
                       help="SLO targets in format 'TTFT,TPOT' in milliseconds (e.g., '1000,100')")
    parser.add_argument("--profiler", type=str, default=None,
                       help="Path to profiler JSON file for performance estimation")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name for performance estimation (e.g., 'meta-llama/Llama-3.2-3B-Instruct')")
    args = parser.parse_args()
    
    # 解析 SLO 参数（单位：毫秒）
    try:
        slo_parts = args.slo.split(',')
        if len(slo_parts) != 2:
            raise ValueError("SLO format must be 'TTFT,TPOT'")
        slo_ttft_ms = float(slo_parts[0])
        slo_tpot_ms = float(slo_parts[1])
        print(f"SLO configured: TTFT={slo_ttft_ms}ms, TPOT={slo_tpot_ms}ms")
    except (ValueError, IndexError) as e:
        print(f"Error parsing SLO parameter: {e}")
        print("Using default SLO: TTFT=1000ms, TPOT=200ms")
        slo_ttft_ms = 1000.0
        slo_tpot_ms = 200.0
    
    # 创建全局调度器，传入性能配置文件路径和模型名称
    scheduler = GlobalScheduler(
        profiler_file_path=args.profiler
    )
    
    # 打印配置信息
    if args.profiler:
        print(f"Performance profiler: {args.profiler}")
    if args.model:
        print(f"Model name: {args.model}")
        # 设置模型名称到load_tracker
        # 如meta-llama/Llama-3.2-3B-Instruct(需要与配置文件中保持一致)
        scheduler.load_tracker.set_model_name(args.model)
    
    # 设置 SLO
    scheduler.load_tracker.set_slo(slo_ttft_ms, slo_tpot_ms)
    
    # 启动服务发现
    discovery_thread = scheduler.start_service_discovery(args.host, args.discovery_port)
    
    # 创建Quart应用
    app = Quart(__name__)
    
    # 注册路由
    app.route('/v1/completions', methods=['POST'])(scheduler.handle_request)
    
    @app.route('/health', methods=['GET'])
    async def health():
        return Response("OK", status=200)
    
    @app.route('/instances', methods=['GET'])
    async def get_instances():
        return jsonify([
            {
                "http_address": inst.http_address,
                "zmq_address": inst.zmq_address,
                "tp": inst.tp,
                "min_token": inst.min_token,
                "max_token": inst.max_token,
                "type": inst.instance_type
            }
            for inst in scheduler.instances.values()
        ])
    
    # 添加实例管理接口
    app.route('/admin/instances/<path:http_address>', methods=['GET'])(scheduler.handle_get_instance)
    app.route('/admin/instances/<path:http_address>', methods=['DELETE'])(scheduler.handle_remove_instance)
    app.route('/admin/instances', methods=['DELETE'])(scheduler.handle_remove_instance)
    app.route('/admin/instances-by-type', methods=['GET'])(scheduler.handle_get_instances_by_type)
    
    # 添加 SLO 管理接口（统一接口支持GET/PUT/POST）
    app.route('/admin/slo', methods=['GET', 'PUT', 'POST'])(scheduler.handle_slo_update)
    
    # 添加惩罚系数配置接口
    app.route('/admin/penalty', methods=['GET', 'PUT', 'POST'])(scheduler.handle_penalty_config)
    
    # 添加区间变化历史记录接口（测试用）
    app.route('/admin/interval-history', methods=['GET'])(scheduler.handle_get_interval_history)

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
