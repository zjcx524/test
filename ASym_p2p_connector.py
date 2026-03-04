# SPDX-License-Identifier: Apache-2.0

import re
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import (
    model_aware_kv_ops_helper as kv_helper)
from vllm.distributed.kv_transfer.kv_pipe.p2p_nccl_pipe import P2pNcclPipe
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

#鏂板姞
from vllm.distributed.parallel_state import get_tp_group

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class ASymP2pConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        logger.info("starting ASymP2pConnector")
        self.rank = rank
        self.vllm_config = config
        self.config = config.kv_transfer_config
        self.kv_helper = kv_helper(config)
        self.src_tp=0
        self.dst_tp=0

        assert self.config.kv_connector == "ASymP2pConnector"

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.p2p_nccl_pipe = P2pNcclPipe(
            local_rank=local_rank,
            config=self.vllm_config,
            hostname="",
            port_offset=rank,
        )
        
        # 负载通知功能配置（用于在prefill完成时通知全局调度器降低负载）
        self.enable_load_notify = self.config.get_from_extra_config(
            "enable_load_notify", False)
        
        # 获取proxy地址用于发送负载通知
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip and proxy_port:
            self.proxy_address = f"{proxy_ip}:{proxy_port}"
        else:
            self.proxy_address = ""
        
        # 初始化异步通知相关组件（仅rank 0负责发送通知）
        self._load_notify_queue: deque = deque()
        self._load_notify_cv = threading.Condition()
        self._load_notify_thread = None
        
        if self.enable_load_notify and rank == 0 and self.proxy_address:
            self._start_load_notify_worker()
            logger.info("Load notify enabled, proxy_address: %s", self.proxy_address)
        elif self.enable_load_notify:
            logger.warning("Load notify enabled but proxy_address not configured or rank != 0")
    
    def _start_load_notify_worker(self):
        """启动异步负载通知工作线程"""
        import zmq
        import msgpack
        
        def worker():
            context = zmq.Context()
            sock = context.socket(zmq.DEALER)
            # 使用独特的identity避免与其他socket冲突
            sock.setsockopt_string(zmq.IDENTITY, 
                f"{self.p2p_nccl_pipe.zmq_address}_load_notify")
            sock.connect(f"tcp://{self.proxy_address}")
            logger.info("Load notify worker connected to %s", self.proxy_address)
            
            while True:
                with self._load_notify_cv:
                    while not self._load_notify_queue:
                        self._load_notify_cv.wait()
                    notify_data = self._load_notify_queue.popleft()
                
                try:
                    sock.send(msgpack.dumps(notify_data))
                    logger.debug("Load notify sent: request_id=%s, input_tokens=%d", 
                                notify_data.get("request_id", ""), 
                                notify_data.get("input_tokens", 0))
                except Exception as e:
                    logger.warning("Failed to send load notify: %s", e)
        
        self._load_notify_thread = threading.Thread(target=worker, daemon=True)
        self._load_notify_thread.start()
    
    def _notify_prefill_complete(self, request_id: str, input_tokens: int):
        """异步通知全局调度器prefill完成
        
        Args:
            request_id: 请求ID
            input_tokens: 输入token数
        
        Note:
            此方法是非阻塞的，将通知放入队列后立即返回
        """
        if not self.enable_load_notify or self._load_notify_thread is None:
            return
        
        notify_data = {
            "type": "PREFILL_COMPLETE",
            "http_address": self.p2p_nccl_pipe.http_address,
            "zmq_address": self.p2p_nccl_pipe.zmq_address,
            "request_id": request_id,
            "input_tokens": input_tokens,
            "timestamp": time.time()
        }
        
        with self._load_notify_cv:
            self._load_notify_queue.append(notify_data)
            self._load_notify_cv.notify()

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        # input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        # get tp size
        tp_group = get_tp_group()
        world_size = tp_group.world_size

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            # current_tokens = input_tokens_tensor[start_pos:end_pos]
            keys, values = [], []
            all_keys, all_values = [], []  # 初始化变量

            request_id = request_ids[idx]
            prefill_ip, prefill_port, prefill_tp, decode_ip, decode_port, decode_tp = self.parse_request_id(request_id)
            #logger.info("prefill_tp:%d,decode_tp:%d",prefill_tp,decode_tp)

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                # key_cache:seq_slen, num_heads, head_size
                key_cache, value_cache = self.kv_helper.get_kv_from_cache(
                    kv_cache, num_heads, head_size)
                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                local_key = key_cache[current_slot_mapping]
                local_value = value_cache[current_slot_mapping]

                #logger.info("self.rank: %d, local_key.shape: %s", self.rank, str(local_key.shape))

                # 收集所有层的key和value
                all_keys.append(local_key.unsqueeze(0))
                all_values.append(local_value.unsqueeze(0))

            # 在层循环结束后合并所有层的key和value
            if all_keys:  # 确保列表不为空
                #logger.info("begin cat kvcache")
                all_keys = torch.cat(all_keys, dim=0)
                all_values = torch.cat(all_values, dim=0)

                # if world_size > 1:
                #     local_new_key = tp_group.all_gather(all_keys, dim=2)
                #     local_new_value = tp_group.all_gather(all_values, dim=2)
                # else:
                #     local_new_key = all_keys
                #     local_new_value = all_values

                # if self.rank < decode_tp:
                #     # cut num_heads
                #     dim_size = local_new_key.shape[2]

                #     start = self.rank * (dim_size // decode_tp)
                #     end = (self.rank + 1) * (dim_size // decode_tp)

                #     # cut kv for decode rank
                #     keys = local_new_key[:, :, start:end, :]  # 修正切片维度
                #     values = local_new_value[:, :, start:end, :]  # 修正切片维度


                if prefill_tp > decode_tp:
                    #logger.info("asymmstric")
                    local_new_key = tp_group.all_gather(all_keys, dim=2)
                    local_new_value = tp_group.all_gather(all_values, dim=2)

                    dim_size = local_new_key.shape[2]

                    start = self.rank * (dim_size // decode_tp)
                    end = (self.rank + 1) * (dim_size // decode_tp)

                    # cut kv for decode rank
                    keys = local_new_key[:, :, start:end, :]  # 修正切片维度
                    values = local_new_value[:, :, start:end, :]  # 修正切片维度

                else:
                    logger.info("symmstric")
                    keys = all_keys
                    values  = all_values

                # 发送KV缓存和隐藏状态
                if self.rank < min(prefill_tp, decode_tp): 
                    # rank 0--(decode_tp-1) get hidden:[5, 2048]
                    logger.info("self.rank: %d, begining send", self.rank)
                    local_hidden = hidden_or_intermediate_states[start_pos:end_pos]
                    
                    # 五维张量
                    kvcache = torch.stack((keys, values), dim=0)

                    remote_address = decode_ip + ":" + str(decode_port + self.rank)

                    # 在发送KV cache前通知全局调度器降低prefill实例负载（仅rank 0发送）
                    if self.rank == 0:
                        self._notify_prefill_complete(request_id, slen)

                    # logger.info("self.rank: %d, kvcache.shape: %s", self.rank, str(kvcache.shape))
                    # logger.info(kvcache.numel()==0)

                    self.p2p_nccl_pipe.send_tensor(request_id + "kv", kvcache, remote_address)
                    self.p2p_nccl_pipe.send_tensor(request_id + "hidden", local_hidden, remote_address)

        # logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())
        # logger.info("finished send")

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        bypass_model_exec = True
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []

        # get tp size
        tp_group = get_tp_group()
        world_size = tp_group.world_size

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            # original remote_address
            # request_id = request_ids[idx]
            # ip, port = self.parse_request_id(request_id, False)
            # remote_address = ip + ":" + str(port + self.rank)

            request_id = request_ids[idx]
            prefill_ip, prefill_port, prefill_tp, decode_ip, decode_port, decode_tp = self.parse_request_id(request_id)
            remote_address = prefill_ip + ":" + str(prefill_port + self.rank)

            # decode rank recv
            if self.rank < min(prefill_tp, decode_tp):

                # logger.info("self.rank: %d, before recv", self.rank)

                kvcache = self.p2p_nccl_pipe.recv_tensor(request_id + "kv",
                                                            remote_address)
                hidden = self.p2p_nccl_pipe.recv_tensor(request_id + "hidden",
                                                            remote_address)
                
                # logger.info("self.rank: %d, have recv", self.rank)
            
                if kvcache is None or hidden is None:
                    # didn't find any match.
                    bypass_model_exec = False
                    continue

                # logger.info("self.rank: %d, kvcache.shape: %s", self.rank, str(kvcache.shape))


            if decode_tp > prefill_tp:
                # if have asym parallelism
                # broadcast shape
                kv_shape = tp_group.broadcast_kv_shape(kvcache, 0, False)
                hidden_shape = tp_group.broadcast_hidden_shape(hidden, 0, False)
                kv_shape = [int(x) for x in kv_shape]
                hidden_shape = [int(x) for x in hidden_shape]
                # kv_shape = [2, 16, 5, 8, 64]
                # hidden_shape = [5, 2048]
                if self.rank >= prefill_tp:
                    kvcache = torch.empty(tuple(kv_shape), dtype=torch.float16, device="cuda")
                    hidden = torch.empty(tuple(hidden_shape), dtype=torch.float16, device="cuda")

                # broadcast kv
                kvcache = tp_group.all_gather(kvcache, dim=3)

                # cut kv for decode rank
                dim_size = kvcache.shape[3]
                # 由于decode tp > prefill tp时，rank >= prefill_tp的decode rank没有接收到任何数据
                # 所以allgather的维度被扩大了decode_tp / prefill_tp倍
                # 需要对维度进行修正
                base = dim_size // ( decode_tp // prefill_tp ) 

                start = self.rank * (base // decode_tp)
                end = (self.rank + 1) * (base // decode_tp)

                # broadcast hidden
                hidden = tp_group.broadcast(hidden, src=0)
            else:
                pass


            num_computed_tokens = current_tokens.shape[0]

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # call self.kv_store to get kv layer by layer
            for layer_id in range(start_layer, end_layer):
                layer = model_executable.model.layers[layer_id]
                # get kvcache object
                kv_cache = kv_caches[layer_id - start_layer]

                # get remote kvcache
                remote_k, remote_v = kvcache[0][layer_id], kvcache[1][layer_id]

                # remote_k:[5, 4, 64]
                self.kv_helper.put_kv_to_cache(model_executable, remote_k,
                                               remote_v, layer, kv_cache,
                                               slot_mapping, start_pos,
                                               end_pos)

            hidden_or_intermediate_states_for_one_req.append(hidden)

        # logger.info("finished recv")

        if not bypass_model_exec:
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    # @staticmethod
    # def parse_request_id(request_id: str, is_prefill=True) -> Tuple[str, int]:
    #     logger.debug("parse_request_id, request_id: %s, is_prefill: %s",
    #                  request_id, is_prefill)
    #     # Regular expression to match the string hostname and integer port
    #     if is_prefill:
    #         pattern = r"___decode_addr_(.*):(\d+)"
    #     else:
    #         pattern = r"___prefill_addr_(.*):(\d+)___"

    #     # Use re.search to find the pattern in the request_id
    #     match = re.search(pattern, request_id)
    #     if match:
    #         # Extract the ranks
    #         ip = match.group(1)
    #         port = int(match.group(2))

    #         logger.debug("parse_request_id, request_id: %s, ip: %s, port: %s",
    #                      request_id, ip, str(port))
    #         return ip, port
    #     raise ValueError(
    #         f"Request id {request_id} does not contain hostname and port")

        
    @staticmethod
    def parse_request_id(request_id: str) -> Tuple[str, int, int, str, int, int]:
        """
        Parse the request_id to extract prefill and decode addresses and ports.

        Args:
            request_id (str): The request ID string.

        Returns:
            Tuple[str, int, str, int]: prefill_ip, prefill_port, decode_ip, decode_port
        """
        logger.debug("parse_request_id, request_id: %s", request_id)

        # logger.info("now parse_request_id, request_id: %s", request_id)
        # logger.info("now parse_request_id, request_id: %s", request_id)

        # Regular expression to match prefill and decode addresses and ports
        # pattern = r"___prefill_addr_(.*):(\d+)_tp_\d+___decode_addr_(.*):(\d+)_tp_\d+"
        # ___prefill_addr_10.0.0.102:14001_tp_2___decode_addr_10.0.0.102:15001_tp_1
        # pattern = r"___prefill_addr_(\d{1,3}\.){3}\d{1,3}:\d+_tp_\d+__decode_addr_(\d{1,3}\.){3}\d{1,3}:\d+_tp_\d+"
        pattern = (
            r"cmpl-___prefill_addr_"
            r"((?:\d{1,3}\.){3}\d{1,3})"  # prefill IP
            r":(\d+)"                      # prefill port
            r"_tp_(\d+)"                   # prefill tp
            r"___decode_addr_"
            r"((?:\d{1,3}\.){3}\d{1,3})"  # decode IP
            r":(\d+)"                      # decode port
            r"_tp_(\d+)"                   # decode tp
            r"_.*"                         # other
        )
        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the IPs and ports
            prefill_ip = match.group(1)
            prefill_port = int(match.group(2))
            prefill_tp = int(match.group(3))
            decode_ip = match.group(4)
            decode_port = int(match.group(5))
            decode_tp = int(match.group(6))

            logger.debug(
                "parse_request_id, prefill_ip: %s, prefill_port: %d, prefill_tp: %d, decode_ip: %s, decode_port: %d, decode_tp: %d",
                prefill_ip, prefill_port, prefill_tp, decode_ip, decode_port, decode_tp
            )
            return prefill_ip, prefill_port, prefill_tp, decode_ip, decode_port, decode_tp

        # Raise an error if the pattern does not match
        raise ValueError(f"Request ID {request_id} does not contain valid prefill and decode addresses, ports, and tp values")


    def close(self) -> None:
        self.p2p_nccl_pipe.close()