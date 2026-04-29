# TrialCEBRA

[![PyPI](https://img.shields.io/pypi/v/TrialCEBRA?color=blue)](https://pypi.org/project/TrialCEBRA/)
[![Tests](https://github.com/colehank/TrialCEBRA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/TrialCEBRA/actions)  
[[English](README.md) | 中文]

为 [CEBRA](https://cebra.ai) 提供 trial 感知对比学习，支持传入 3D epoch 格式数据 `(ntrial, ntime, nneuro)`，trial 边界自动处理。

---

## 安装

**第一步** — 前往 [pytorch.org](https://pytorch.org/get-started/locally/) 安装 PyTorch。

**第二步**

```bash
pip install TrialCEBRA
```

---

## 快速开始

```python
import numpy as np
from trial_cebra import TrialCEBRA

X = np.random.randn(40, 50, 64).astype(np.float32)   # (ntrial, ntime, nneuro)
y = np.random.randn(40, 16).astype(np.float32)        # (ntrial, stim_dim)

model = TrialCEBRA(
    model_architecture = "offset10-model",
    conditional        = "delta",
    time_offsets       = 5,
    delta              = 0.3,
    output_dimension   = 3,
    max_iterations     = 1000,
    batch_size         = 512,
)

model.fit(X, y)
emb = model.transform(X)        # (ntrial, ntime, 3)  — 形状保持一致
```

传入 2D 扁平数据 `(N, nneuro)` 时回退为原生 CEBRA 行为，不受影响。

---

## Conditional 体系

| `conditional` | Trial 选择 | Trial 内采样 | y 形状 |
|---|---|---|---|
| `"time"` | 均匀随机 | ±`time_offsets` 时间窗 | 不需要 |
| `"delta"` | 基于 y 的 Gaussian 相似度 | 均匀（自由） | `(ntrial, nd)` 或 `(ntrial, ntime, nd)` |
| `"time_delta"` | 跨 trial 联合 argmin | ±`time_offsets` 时间窗 | `(ntrial, ntime, nd)` |

同时传入离散整型标签（`int64` dtype）和连续标签，可为 `"delta"` 启用类条件 trial 选择。

---

## 关键参数

```python
TrialCEBRA(
    conditional            = "delta",    # "time" | "delta" | "time_delta"
    time_offsets           = 10,         # trial 内时间窗半宽
    delta                  = 0.1,        # trial 相似度 Gaussian 噪声标准差
    sample_fix_trial       = False,      # True：在 init 时固定 trial 配对
    sample_exclude_intrial = True,       # True：正样本始终来自不同 trial
    sample_prior           = "balanced", # "balanced"（默认）或 "uniform"
    output_dimension       = 3,
    # ... 接受所有 cebra.CEBRA 参数
)
```

`fit` 后可通过 `model.distribution_` 访问采样分布实例。

---

## Transform

`transform()` 保持输入维度数不变：

```python
emb = model.transform(X)          # (ntrial, ntime, 3)  若 X 为 (ntrial, ntime, nneuro)
emb = model.transform(X_flat)     # (N, 3)              若 X_flat 为 (N, nneuro)
emb = model.transform_epochs(X)   # 严格 3D 变体，X.ndim != 3 时抛出异常
```

---

## Metrics

所有 Metrics 方法均直接接受 epoch 格式 `(ntrial, ntime, nneuro)` 数据：

```python
loss = model.infonce_loss(X, y)
gof  = model.goodness_of_fit_score(X, y)
hist = model.goodness_of_fit_history()       # 训练曲线，无需传 X

# Consistency score：接受 3D embedding 列表
emb1 = model.transform(X1)   # (ntrial, ntime, 3)
emb2 = model.transform(X2)
scores, pairs, ids = TrialCEBRA.consistency_score(
    [emb1, emb2], between="runs"
)
# between-datasets：labels 支持 (ntrial, ntime) 或 (ntrial*ntime,)
scores, pairs, ids = TrialCEBRA.consistency_score(
    [emb1, emb2],
    between="datasets",
    labels=[y1, y2],
    dataset_ids=["mouse1", "mouse2"],
)
```

---

## Decoder

CEBRA 的 decoder（`KNNDecoder`、`L1LinearRegressor`）是独立的 sklearn estimator，期望 2D 输入。使用时先将 embedding 展平：

```python
import cebra

emb      = model.transform(X)                  # (ntrial, ntime, 3)
emb_flat = emb.reshape(-1, emb.shape[-1])       # (ntrial*ntime, 3)
y_flat   = y.reshape(-1)                        # (ntrial*ntime,)

decoder = cebra.KNNDecoder()
decoder.fit(emb_flat, y_flat)
score = decoder.score(emb_flat, y_flat)
```

---

## 多会话训练

将 `X` 传为 epoch 数组的列表（每 session 一个）：

```python
X = [
    np.random.randn(30, 100, 64).astype(np.float32),   # session 0
    np.random.randn(25,  80, 48).astype(np.float32),   # session 1
]
y_cont = [np.random.randn(30, 100, 16).astype(np.float32),
          np.random.randn(25,  80, 16).astype(np.float32)]
y_disc = [np.zeros((30, 100), dtype=np.int64),
          np.zeros((25,  80), dtype=np.int64)]

model = TrialCEBRA(conditional="delta", output_dimension=3, max_iterations=1000)
model.fit(X, y_disc, y_cont)
```

多会话支持 `"delta"` 和 `"time_delta"`；`"time"` 会抛出 `NotImplementedError`。

---

## 参与贡献

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest tests/ -v
```

发布新版本：`git tag vX.X.X && git push origin vX.X.X`。
