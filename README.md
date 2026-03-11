# Global Scheduler v2

这是一个重构后的全局调度器，将代码拆分为多个模块以提高可维护性。

## 文件结构

```
v2/
├── __init__.py             # 包初始化文件
├── instance.py             # Instance 类定义
├── load_tracker.py         # 传统负载追踪器（已废弃）
├── async_load_tracker.py   # 异步负载追踪器（新版本）
├── global_scheduler.py     # GlobalScheduler 类和主程序入口
├── interactive_menu.py     # InteractiveMenu 交互式菜单类
├── utils.py               # 工具函数
├── script/                # 脚本目录
│   └── start_genserve.sh   # 启动脚本
├── test/                  # 测试目录
└── README.md              # 本文件
```

## 模块说明

### instance.py
- **Instance类**: 管理单个实例的配置信息
- 包含HTTP地址、ZMQ地址、并行度配置、token长度区间等
- 实现了__eq__、__hash__、__repr__魔法方法

### async_load_tracker.py
- **AsyncLoadTracker类**: 异步负载追踪器（新版本）
- 基于请求ID的ZMQ地址解析，支持乱序请求处理
- 自动识别实例并更新负载信息
- 支持分离式prefill+decode请求的负载管理
- 提供请求完成和失败的异步处理能力

### load_tracker.py
- **LoadTracker类**: 传统负载追踪器（已废弃）
- 保留用于兼容性，建议使用AsyncLoadTracker

### interactive_menu.py
- **InteractiveMenu类**: 交互式控制台菜单
- 提供用户友好的命令行界面
- 支持两种退出模式：退出菜单（保持服务）和关闭服务
- 支持实时查看服务状态、实例信息、API文档等
- 包含完整的使用示例和帮助系统

### global_scheduler.py
- **GlobalScheduler类**: 核心调度器类
- 集成AsyncLoadTracker进行智能负载管理
- 管理所有实例的注册、移除和调度
- 提供完整的HTTP API接口
- 处理分离式prefill+decode请求
- 支持基于ZMQ地址的请求ID解析
- 包含主程序入口点

### utils.py
- **generate_split_request_id**: 生成包含ZMQ地址信息的请求ID
- **forward_request_with_split**: 处理分离式请求的工具函数
- **random_uuid**: UUID生成工具函数
- 其他通用工具函数

## 使用方法

### 启动调度器

#### 交互式模式（默认）:
```bash
python global_scheduler.py --host 0.0.0.0 --port 8000 --discovery-port 30001
```

#### 非交互式模式:
```bash
python global_scheduler.py --host 0.0.0.0 --port 8000 --discovery-port 30001 --no-interactive
```

### 交互式菜单命令
- `quit` 或 `q` - 退出菜单但保持服务运行
- `shutdown` 或 `stop` - 完全关闭服务
- `help` - 查看帮助信息
- `status` - 查看服务状态
- `instances` - 查看实例管理信息

## API接口测试

### 1. 健康检查
```bash
# 检查服务是否正常运行
curl -X GET "http://localhost:8000/health"
```

### 2. 查看所有实例
```bash
# 获取所有实例的简单列表
curl -X GET "http://localhost:8000/instances"
```

### 3. 按类型查看实例
```bash
# 获取按类型分组的实例详细信息（prefill/decode）
curl -X GET "http://localhost:8000/admin/instances-by-type"
```

### 4. 获取单个实例详情
```bash
# 注意：URL中的冒号需要编码为%3A
curl -X GET "http://localhost:8000/admin/instances/192.168.1.100%3A8080"
```

### 5. 删除实例

#### 通过URL路径删除
```bash
curl -X DELETE "http://localhost:8000/admin/instances/192.168.1.100%3A8080"
```

#### 通过请求体删除（兼容性）
```bash
curl -X DELETE "http://localhost:8000/admin/instances" \
  -H "Content-Type: application/json" \
  -d '{
    "http_address": "192.168.1.100:8080"
  }'
```

### 6. SLO 管理接口

#### 查询当前 SLO
```bash
curl -X GET "http://localhost:8000/admin/slo"
```

#### 更新 SLO（方式一：JSON格式）
```bash
curl -X PUT "http://localhost:8000/admin/slo" \
  -H "Content-Type: application/json" \
  -d '{
    "ttft_ms": 500,
    "tpot_ms": 50
  }'
```

#### 更新 SLO（方式二：字符串格式）
```bash
curl -X POST "http://localhost:8000/admin/slo" \
  -H "Content-Type: application/json" \
  -d '{
    "slo": "600,60"
  }'
  curl -X POST "http://localhost:8000/admin/slo" \
  -H "Content-Type: application/json" \
  -d '{
    "slo": "200,20"
  }'
  curl -X POST "http://localhost:8000/admin/slo" \
  -H "Content-Type: application/json" \
  -d '{
    "slo": "300,40"
  }'
```

### 7. 推理请求
```bash
# 发送推理请求（需要有prefill和decode实例已注册）
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "prompt_len": 5,
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 8. 查看区间变化历史记录
```bash
# 查看所有实例的区间动态调整历史
curl -X GET "http://localhost:8000/admin/interval-history"
```

返回示例：
```json
{
  "total_adjustments": 2,
  "instances": {
    "10.0.0.103:20001": {
      "changes": [...],
      "formatted": [
        "[0, inf] -> [0, 5000]",
        "[0, 5000] -> [0, 3000]"
      ]
    }
  },
  "current_intervals": {
    "10.0.0.103:20001": {
      "current_interval": "[2501, 7000]",
      "type": "prefill"
    },
    "10.0.0.103:20002": {
      "current_interval": "[1, 2500]",
      "type": "decode"
    }
  }
}
```

**字段说明：**
- `total_adjustments`: 总的区间调整次数
- `instances`: 每个实例的区间变化历史
  - `changes`: 原始变化记录
  - `formatted`: 格式化的变化记录（旧区间 -> 新区间）
- `current_intervals`: 当前各实例的区间状态和类型

### 9. 惩罚系数配置接口

运行时动态调整调度惩罚系数，无需重启服务。

#### 查询当前配置
```bash
curl -X GET "http://localhost:8000/admin/penalty"
```

#### 更新配置（可单独更新任意参数）
```bash
curl -X PUT "http://localhost:8000/admin/penalty" \
  -H "Content-Type: application/json" \
  -d '{
    "prefill_transition_cost": 2.0,
    "decode_transition_cost": 2.0,
    "soft_allocation_penalty": 3.0
  }'
```

**参数说明：**
| 参数 | 说明 | NVLink环境建议值 | 跨节点环境建议值 |
|------|------|-----------------|-----------------|
| `prefill_transition_cost` | prefill实例跃迁惩罚 | 2.0+ | 0.3-0.5 |
| `decode_transition_cost` | decode实例跃迁惩罚 | 2.0+ | 0.3-0.5 |
| `soft_allocation_penalty` | 软分配惩罚 | 3.0+ | 0.5-1.0 |

### 10. 批量测试脚本

#### 快速健康检查
```bash
# 测试所有基本接口
echo "=== 健康检查 ==="
curl -s "http://localhost:8000/health"

echo -e "\n=== 实例列表 ==="
curl -s "http://localhost:8000/instances" | python3 -m json.tool

echo -e "\n=== 按类型查看实例 ==="
curl -s "http://localhost:8000/admin/instances-by-type" | python3 -m json.tool

echo -e "\n=== 当前SLO配置 ==="
curl -s "http://localhost:8000/admin/slo" | python3 -m json.tool

echo -e "\n=== 区间变化历史 ==="
curl -s "http://localhost:8000/admin/interval-history" | python3 -m json.tool
```

#### 完整功能测试
```bash
#!/bin/bash
# 完整的API功能测试脚本

BASE_URL="http://localhost:8000"

echo "=== 全局调度器API测试 ==="

# 1. 健康检查
echo "1. 健康检查..."
curl -s "http://localhost:8000/health"
echo

# 2. 查看初始实例
echo "2. 查看初始实例..."
curl -s "http://localhost:8000/instances" | python3 -m json.tool
echo

# 3. 查看按类型分组的实例
echo "3. 查看按类型分组的实例..."
curl -s "http://localhost:8000/admin/instances-by-type" | python3 -m json.tool
echo

# 4. 查询当前SLO
echo "4. 查询当前SLO..."
curl -s "http://localhost:8000/admin/slo" | python3 -m json.tool
echo

# 5. 更新SLO
echo "5. 更新SLO..."
curl -s -X PUT "http://localhost:8000/admin/slo" \
  -H "Content-Type: application/json" \
  -d '{
    "ttft_ms": 800,
    "tpot_ms": 80
  }' | python3 -m json.tool
echo

# 6. 验证SLO已更新
echo "6. 验证SLO已更新..."
curl -s "$BASE_URL/admin/slo" | python3 -m json.tool
echo

# 7. 查看区间变化历史
echo "7. 查看区间变化历史..."
curl -s "http://localhost:8000/admin/interval-history" | python3 -m json.tool
echo

echo "=== 测试完成 ==="
```

## 命令行参数说明

```bash
python global_scheduler.py [OPTIONS]

选项:
  --host TEXT              监听地址 (默认: 0.0.0.0)
  --port INTEGER           HTTP服务端口 (默认: 8000)
  --discovery-port INTEGER ZMQ服务发现端口 (默认: 30001)
  --slo TEXT               SLO目标，格式为 'TTFT,TPOT'，单位毫秒 (默认: "1000,100")
  --profiler TEXT          性能分析JSON文件路径
  --model TEXT             模型名称，用于性能估算 (如: 'meta-llama/Llama-3.2-3B-Instruct')
```

### 启动示例

#### 基本启动
```bash
python global_scheduler.py --host 0.0.0.0 --port 8000 --discovery-port 30001
```

#### 带SLO配置启动
```bash
python global_scheduler.py \
  --host 0.0.0.0 \
  --port 8000 \
  --discovery-port 30001 \
  --slo "500,50"
```

#### 带性能分析器启动
```bash
python global_scheduler.py \
  --host 0.0.0.0 \
  --port 8000 \
  --discovery-port 30001 \
  --slo "500,50" \
  --profiler ./profiler.json \
  --model "meta-llama/Llama-3.2-3B-Instruct"
```

## API接口说明

### 主要接口
| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/completions` | POST | 推理请求（分离式prefill+decode） |
| `/health` | GET | 健康检查 |
| `/instances` | GET | 查看所有已注册实例 |

### 实例管理接口
| 接口 | 方法 | 说明 |
|------|------|------|
| `/admin/instances/<address>` | GET | 获取单个实例详情及负载信息 |
| `/admin/instances/<address>` | DELETE | 删除指定实例 |
| `/admin/instances` | DELETE | 通过请求体删除实例 |
| `/admin/instances-by-type` | GET | 按类型（prefill/decode）查看实例 |

### SLO管理接口
| 接口 | 方法 | 说明 |
|------|------|------|
| `/admin/slo` | GET | 查询当前SLO配置 |
| `/admin/slo` | PUT/POST | 更新SLO配置 |

### 调试监控接口
| 接口 | 方法 | 说明 |
|------|------|------|
| `/admin/interval-history` | GET | 查看实例区间变化历史记录 |

> **注意**: 实例注册是通过 ZMQ 服务发现自动完成的，实例启动时会向调度器的 `discovery-port` 发送注册消息（type="P" 或 "D"）。

## 特性说明

### 异步负载追踪
- 基于请求ID的ZMQ地址解析，支持乱序返回的请求处理
- 请求ID格式：`___prefill_addr_{zmq}_tp_{tp}___decode_addr_{zmq}_tp_{tp}_{uuid}`
- 自动识别实例并更新相应的负载信息
- 支持通过 `complete_request_by_id()` 和 `fail_request_by_id()` 进行异步完成

### 交互式菜单增强
- `quit/q` - 退出菜单但保持服务运行（新增）
- `shutdown/stop` - 完全关闭服务
- 支持Ctrl+C优雅处理，不会意外关闭服务

### 分离式处理
- 支持prefill和decode实例的分离式处理
- 智能实例选择策略（轮询、最小负载、随机）
- 统一的请求ID管理和负载追踪

## 故障排除

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查端口是否被占用
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :30001
   ```

2. **实例添加失败**
   - 检查JSON格式是否正确
   - 确认http_address和zmq_address格式为 `ip:port`
   - 验证instance_type是否为 `prefill`、`decode` 或 `general`

3. **推理请求失败**
   - 确保至少有一个prefill实例和一个decode实例
   - 检查实例的token长度范围配置
   - 查看服务日志获取详细错误信息

### 日志查看
- 服务启动后会显示详细的API地址和端口信息
- 使用交互式菜单的 `status` 命令查看服务状态
- 使用 `instances` 命令查看实例管理信息

## 开发说明

### 扩展新功能
1. 实例类型：在 `instance.py` 中添加新的实例类型支持
2. 负载策略：在 `async_load_tracker.py` 中扩展负载选择算法
3. API接口：在 `global_scheduler.py` 中添加新的HTTP接口

### 测试建议
1. 使用提供的curl命令进行基本功能测试
2. 运行完整功能测试脚本验证所有接口
3. 测试交互式菜单的各种命令
4. 验证异步负载追踪的准确性
