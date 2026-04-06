def _calculate_effective_prompt_length(self, instance: Instance, input_tokens: int) -> int:
        """计算prefill阶段的有效prompt长度（排除已有KV cache）
        
        对于prefill实例，有效prompt长度 = 全部输入长度 - 已缓存的KV tokens
        
        当前实现：如果instance有kv_cache_tokens属性，则计算有效长度；
        否则返回全部输入长度（假设无KV缓存）。
        
        未来扩展：
        - 支持instance跟踪KV cache状态（按会话或按用户）
        - 支持动态KV cache管理（reuse、evict等）
        
        Args:
            instance: 实例对象
            input_tokens: 全部输入token数量
            
        Returns:
            有效prompt长度（待处理的新token数量）
        """
        # 检查实例是否有KV cache信息
        # TODO:根据每个实例所有处理请求的共有前缀为复用长度，当前简化为直接设置属性，可以与KVcache感知的请求调度结合
        if hasattr(instance, 'kv_cache_tokens') and instance.kv_cache_tokens > 0:
            # 有效长度 = 全部输入 - 已缓存KV对应的前缀
            effective_length = max(0, input_tokens - instance.kv_cache_tokens)
            return effective_length
        
        # 默认：无KV缓存，全部输入都是新的prompt
        return input_tokens