# LingBot-VA 昇腾 NPU 适配总结

## 适配目标

将 `libero_train` 训练流程从 CUDA 迁移到昇腾 NPU，8 卡运行：

```bash
NGPU=8 CONFIG_NAME='libero_train' bash script/run_va_posttrain.sh
```

核心原则：**代码保持** **`cuda`/`nccl`** **表面 API 不变**，由 `npu_adapter.adapt_to_accelerator_device()` 运行时自动映射到 NPU/HCCL。

***

## 快速开始

1. 基础镜像选择：ascend-a2-ubuntu:v4.0
2. 安装依赖： pip install lerobot==0.3.3
3. 配置模型权重以及数据集路径：wan\_va/configs
4. 训练运行拉起：

```bash
NGPU=8 CONFIG_NAME='libero_train' bash script/run_va_posttrain.sh
```

## 关键改动

### 1. 启动脚本 `script/run_va_posttrain.sh`

- 新增 `setup_ascend_env()`：source CANN toolkit / ATB 环境变量、驱动 LD\_LIBRARY\_PATH
- Python 解释器切到 `/opt/mamba/envs/ascend-torch/bin/python`（`PYTHON_BIN` 可覆盖）
- `HF_HOME` 等缓存目录自动 fallback 到仓库内 `.cache/`（解决只读文件系统）
- HCCL 端口设 `auto` 避免 16666 冲突
- 默认 `WAN_VA_DISABLE_FLEX_COMPILE=1`
- 默认 `CONFIG_NAME` 改为 `libero_train`

### 2. NPU 适配入口 `wan_va/train.py`

- 顶部加入 `from npu_adapter import adapt_to_accelerator_device; adapt_to_accelerator_device()`
- 所有 `dist.barrier()` 替换为 `dist_barrier()`

### 3. 分布式同步 `wan_va/distributed/util.py`

- 新增 `dist_barrier()`：自动带 `device_ids` 的安全 barrier
- `init_process_group` 增加 `device_id=torch.device("cuda", local_rank)`
- 移除 `_configure_model()` 中无用的提前 barrier

### 4. FlexAttention 编译开关 `wan_va/modules/model.py`

- 新增 `_create_block_mask()` 方法：`WAN_VA_DISABLE_FLEX_COMPILE=1` 时走 `_compile=False`，否则保持编译路径
- 注释 flash\_attn import（昇腾环境不可用）
- `attn_mode='flex'` 暂退到 `custom_sdpa`

### 5. 数据集加载 `wan_va/dataset/lerobot_latent_dataset.py`

- `num_init_worker` 从配置读取（默认 4），避免 128 worker × 8 rank 进程爆炸
- `empty_emb.pt` 缺失时容错，CFG dropout 用 `zeros_like` 兜底

### 6. 配置 `wan_va/configs/`

- 数据集路径、预训练模型路径 → 实际路径
- 降低 worker 数

***

## 踩坑记录

| 问题                                  | 根因                                         | 解决                                         |
| ----------------------------------- | ------------------------------------------ | ------------------------------------------ |
| **`No module name`**`d torch`       | 系统 Python 无训练环境                            | 脚本指定 ascend-torch Python                   |
| `dcmi module initialize failed`     | 未加载 Ascend 驱动环境                            | `setup_ascend_env()` source 环境变量           |
| HCCL 端口 16666 冲突                    | 默认端口被占用                                    | `HCCL_*_SOCKET_PORT_RANGE=auto`            |
| TorchInductor `get_gpu_type()` 断言失败 | `torch_npu` 和 CUDA 兼容层同时暴露，Inductor 看到两个后端 | `WAN_VA_DISABLE_FLEX_COMPILE=1` 禁用 mask 编译 |
| FSDP barrier 通信失败                   | barrier 无 `device_ids`，HCCL 需要             | `dist_barrier()` 封装                        |
| 数据集初始化进程爆炸                          | 硬编码 128 worker                             | 配置化 `dataset_init_worker=4`                |

***

## 涉及文件

`script/run_va_posttrain.sh` · `wan_va/train.py` · `wan_va/distributed/util.py` · `wan_va/modules/model.py` · `wan_va/dataset/lerobot_latent_dataset.py` · `wan_va/configs/va_libero_cfg.py` · `wan_va/configs/va_libero_train_cfg.py` · `tests/test_flex_mask_compile_switch.py`

***

