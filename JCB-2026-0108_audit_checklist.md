# JCB-2026-0108 审稿意见 → 代码库逐项核查清单

**核查环境**：`D:\anaconda3\envs\py38`（Python 3.8.20 / numpy 1.24.4 / scipy 1.10.1 / torch 2.4.1+cpu / lifelines 0.27.8）
**核查日期**：2026-09-27
**对照版本**：已发布版 `federated-survival 0.5.0`（审稿人所测） ⟷ 仓库源码 `0.7.0`

---

## 0. 一个关键前提

py38 环境的 `site-packages` 里**另装了一份已发布包**：

| 项 | 值 |
|---|---|
| 安装路径 | `D:\anaconda3\envs\py38\lib\site-packages\federated_survival\` |
| `dist-info` 版本 | **0.5.0** |
| `__init__.py` 中的 `__version__` | **0.1.0** |
| 子包 | 仅 `core / data / examples / utils`（无 `protocols`、`models`、`api.py`） |
| 随包测试 | **无** |

→ 这个环境就是**审稿人测试环境的忠实复现**。因此本次核查采用"前后对照"：
**前半 = 在 py38 中对已安装 0.5.0 复现缺陷；后半 = 在 py38 中对仓库 0.7.0 验证修复。**

> 复现脚本：`C:\Users\skyee\federated_survival\audit\repro_050.py`（从非仓库目录运行，使 site-packages 的 0.5.0 生效）

---

## 1. 缺陷复现（已安装 0.5.0，py38，before）

| # | 审稿人指控 | 实测结果 | 判定 |
|---|---|---|---|
| 1 | `__version__` 是 0.1.0，PyPI 是 0.5.0 | 模块 `0.1.0` / dist-info `0.5.0` | ✅ 复现 |
| 2 | `calculate_ibs()` 是空桩，返回 `None` | 源码为 `# TODO: 实现IBS计算` + `pass`，调用返回 `None` | ✅ 复现 |
| 3 | `split_type='Dirichlet'` 静默执行 time-non-iid | 见下方代码证据；两种分区的客户端大小与均值时间**完全相同** | ✅ 复现 |
| 4 | `batch_size` 完全不起作用 | 8 vs 64 → 测试 C-index **[0.5655, 0.5693] 逐位相同** | ✅ 复现 |
| 5 | `dp_clip_norm` / `dp_noise_multiplier` 完全不起作用 | 0.1 vs 5.0/10.0 → C-index **[0.5873, 0.4924] 逐位相同** | ✅ 复现 |
| 6 | 未发布任何测试 | site-packages 无 `tests/` | ✅ 复现 |

### 代码级根因（0.5.0）

**① Dirichlet 死分支**（`data/splitter.py`）——`__init__` 已把值小写化，分派处却比大写：

```python
# 第 37 行
self.split_type = split_type.lower()
...
# 第 75 行（永远不成立 → 落入 else = time-non-iid）
elif self.split_type == 'Dirichlet':
    ...
else:  # time-non-iid
    client_data = self._split_time_non_iid(train_data)
```

**② `batch_size` 从未传入训练**（`core/client.py` 第 101/109 行写死 `batch_size=self.N`；`runner.py`/`server.py` 硬编码 `2048`）。

**③ DP 参数未参与噪声计算**（`core/differential_privacy.py` 第 44 行）：

```python
sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * sensitivity / self.epsilon
# self.noise_multiplier / self.clip_norm 从未出现在该公式中
```

**④ `scipy.integrate.simps` 崩溃**：0.5.0 的 IBS 由 pycox `EvalSurv.integrated_brier_score()` 计算，内部调用 `scipy.integrate.simps`（scipy ≥ 1.14 已移除，却在包自身依赖范围内）。

---

## 2. 修复验证（仓库 0.7.0，py38，after）

### 2.1 测试与工程化

| 项 | 0.5.0 | 0.7.0（py38 实测） |
|---|---|---|
| 随包测试 | 无 | **24 个测试文件** |
| 测试执行 | — | **206 个用例全部通过**（0 failed / 0 error，exit 0） |
| CI | 无 | `.github/workflows/ci.yml` + `docs.yml` |
| 文档站 | 仅 README | `mkdocs.yml` + `docs/` **11 篇** + docs CI |
| 版本一致性 | `0.1.0` vs `0.5.0` ❌ | `pyproject.toml` = `__init__.py` = **0.7.0** ✅ |
| 依赖范围 | — | `requires-python >= 3.8`，py38 兼容 ✅ |

### 2.2 功能缺陷

| # | 指控 | 0.7.0 实测 | 判定 |
|---|---|---|---|
| 2 | `calculate_ibs()` 空桩 | 已实现：拟合较优曲线 **0.2195** < 较差曲线 **0.3390**（越低越好） | ✅ 已修 |
| 3 | `Dirichlet` 死分支 | `splitter.py:90` 改为 `== 'dirichlet'`；实测客户端大小 **`[81, 93, 66]`**、均值时间 **`[0.982, 0.502, 0.845]`**（与 time-non-iid 明确不同） | ✅ 已修 |
| 4 | `batch_size` 无效 | 8 → C-index **0.5863**；64 → **0.6129**；预测逐点差异 `[0.00088, 0.00044, 0.00164, 0.00238, 0.00114]` | ✅ 已修 |
| 5 | DP 参数无效 | `clip_norm` 0.1/1.0/10 → C-index **0.5825 / 0.1565 / 0.0**；`noise_mult` 0.1/5 → **0.5825 / 0.0** | ✅ 已修 |
| 6 | `simps` 崩溃 | 新增 `_compat.py`：`scipy` 缺 `simps` 时以 `simpson` 补齐（含位置参数兼容） | ✅ 已修 |
| 7 | 示例需手设 `n_features` | `api.py:135` `self.config.n_features = int(first_x.shape[1])` 自动推断 | ✅ 已修 |
| 9 | Figure 4 配置 100% 种子返回不可用预测 | 5/5 种子 `finite=True`、**曲线互异**（20/20）、**单调非增** | ✅ 已修 |

**Figure-4 配置 5 种子实测**（SDGM1, n=100, 3 clients, ε=1）：

| seed | 0 | 1 | 2 | 3 | 4 | 均值 | 标准差 |
|---|---|---|---|---|---|---|---|
| Federated C-index | 0.5248 | 0.6209 | 0.5267 | **0.4507** | 0.6985 | **0.5643** | **0.0963** |

---

## 3. 仍未完全闭合的项 ⚠️ / ❌

| # | 审稿人指控 | 0.7.0 现状 | 结论 |
|---|---|---|---|
| 10 | 仅单次运行，无重复 / 置信区间 / 显著性检验 | 已有 5 随机种子结果（`results/gbsg_5seeds_repro`、`results/gbsg_non_iid_fedprox`），绘图使用 `mean_ci95` 误差带；**但无 wilcoxon/ttest/bootstrap 假设检验** | ⚠️ 部分 |
| 11 | 未使用真实数据集 | 已加入 **GBSG**（`n_train=548`）× 5 种子 × FedAvg/FedProx | ⚠️ 部分（建议补 colon 等第二个数据集） |
| 12 | 无外部方法对比（WebDISCO、Verticox） | 已新增协议 `protocols/webdisco.py`；**Verticox 属纵向联邦（vertical FL），与本包横向联邦（horizontal FL）的威胁模型和数据布局不兼容，不具可比性，不在范围内** | ✅ 已处理 |
| 13 | 数据为模拟、真值已知，却从不与真实基准对比 | 全库 `oracle / ground_truth / true_survival` **零命中** | ❌ 未处理 |
| 14 | 无显著性检验 | 仅 `mean_ci95` 误差带；无任何假设检验函数 | ❌ 未处理 |
| 15 | 弱信号下"联邦显著优于本地"无法与噪声区分 | 5 种子均值 **0.5643**、标准差 **0.0963**；seed 3 仅 **0.4507 < 0.5**。GBSG 上 Federated ≈ Center（0.5756 vs 0.5743）、与 Weighted Local 差距很小 | ⚠️ 主张需收敛/改写 |

### DP 数值溢出：根因与修复（已解决）

**症状**：`dp_clip_norm=1.0` 时生存曲线变为非有限值（`finite=False, within_probability_bounds=False, monotone_non_increasing=False`），C-index 掉到 **0.1565**；`noise_mult=5` 时 C-index = **0.0**；默认 `raise` 模式下直接抛 `PredictionValidationError`。

**根因（两层）**

1. **预测端数值不稳定**。pycox 计算累计风险为 `H0(t) · exp(logh)`；当 `logh` 很大时 `exp(logh) = inf`，再乘首个事件时间之前的 `H0 = 0`，得到 `inf × 0 = NaN`。整条曲线因此变成 NaN（`pycox/models/cox.py` 另报 `overflow encountered in exp`）。这不是 DP 独有——模型发散时同样触发。
2. **噪声标定未计入客户端平均**。`privatize_model_update` 原本令 `sigma = noise_multiplier × dp_sensitivity × clip_norm`，默认全为 1.0 → 每个权重元素注入标准差 **1.0** 的噪声，而神经网络权重典型量级仅 0.01–0.3，信噪比被淹没。同时 `get_noise_scale()`（含 `1/√K` 衰减）是**死代码**，与实际加噪不一致。

**修复（已实施并验证）**

| 文件 | 改动 |
|---|---|
| `models/adapters.py` | 新增 `stable_cox_survival()`：以对数空间计算 `S(t\|x)=exp(-exp(log H0(t) + logh))`，并对数截断到 700。数学上等价，但 `H0=0 → log H0=-inf → S=1`，**永不产生 NaN**；输出恒有限、⊂[0,1]、随时间非增。`_CoxPHAdapter`（含 DeepSurv）与 `CoxCCAdapter` 已接入 |
| `core/differential_privacy.py` | `privatize_model_update()` 计入 `1/√K` 客户端平均（与 `get_noise_scale` 对齐，消除死代码不一致）；新增 `last_noise_to_signal` 等诊断 |
| `core/client.py` | 调用处传入 `num_clients=self.config.num_clients` |
| `core/runner.py` | `privacy_info` 增加 `effective_noise_sigma`，用户可直接看出实际注入的噪声强度 |
| `tests/test_stable_survival.py` | 新增 8 个回归测试（含"朴素路径确实产生 NaN"的对照断言） |

**修复前后对比（py38 实测，同一配置同一随机种子）**

| DP 设置 | 修复前 | 修复后 |
|---|---|---|
| `clip_norm=0.1` | 0.5825 | **0.6091** |
| `clip_norm=1.0` | **0.1565，非有限** | **0.5930，有限且曲线互异** |
| `clip_norm=10` | 0.0，非有限 | **0.5，有限**（并被诊断标记为退化） |
| `noise_mult=0.1` | 0.5825 | **0.6091** |
| `noise_mult=5` | 0.0，非有限 | **0.5，有限**（并被诊断标记为退化） |

`clip_norm=1.0` 这一档从"崩溃"回到 **0.5930**，已与关闭 DP 时的水平相当。极端设置（`clip_norm=10`、`noise_mult=5`）现在返回**合法但无信息量**的曲线（C-index = 0.5 且 `identical_across_samples=True`）——这是正确行为：算法不再崩溃，而是如实报告"该隐私预算已淹没信号"。

**回归**：全量测试 **214 个用例全部通过**（原 206 + 新增 8），0 失败。

**仍需在论文/文档中说明**：给出**经过验证的 DP 超参组合**（建议 `clip_norm ∈ [0.01, 0.1]`、`noise_multiplier ∈ [0.1, 1.0]`），并在文中明确该机制是"实验性客户端更新扰动"，不等同于记录级 DP-SGD。

---

## 4. 修改清单（可直接落到回复信 / 修订稿）

### 4.1 已在 0.7.0 闭合（回复信可直接引用实测数据）

- [x] 随包测试 + CI：206 用例通过，`ci.yml` / `docs.yml`
- [x] `calculate_ibs()` 补全实现（去掉 `pass` 桩）
- [x] Dirichlet 分派大小写 bug 修复
- [x] `batch_size` / `dp_clip_norm` / `dp_noise_multiplier` 真正接入训练与噪声计算
- [x] `_compat.py` 兼容 scipy ≥ 1.14 的 `simps`
- [x] `n_features` 自动推断
- [x] 版本号统一为 0.7.0
- [x] MkDocs API 文档站
- [x] 多随机种子（5 seeds）实验脚手架与 CI95 误差带
- [x] 真实数据集 GBSG 实验
- [x] WebDISCO 协议实现
- [x] **Cox 生存曲线数值稳定化**（对数空间计算，彻底消除 `inf × 0 = NaN`）+ 8 个回归测试
- [x] **DP 噪声标定计入 `1/√K` 客户端平均**，并暴露 `last_noise_to_signal` / `effective_noise_sigma` 诊断

### 4.2 待办（本轮必须补，否则审稿人仍会退回）

1. ~~加入 Verticox~~ — **不适用**：Verticox 是纵向联邦，与本包横向联邦设定不可比。回复信中应**明确说明这一点**，并只承诺横向联邦基线（WebDISCO 已实现）
2. **实现"真实基准（真值生存时间/风险）对照"评测**——模拟数据下真值已知，应报告 oracle C-index 作为上界
3. **加入显著性检验**（配对 bootstrap / Wilcoxon），把"substantially outperforms"改成有统计支持的表述
4. **修正核心主张的措辞**：现有数据显示 Federated ≈ Center、略优于 Weighted Local；"显著优于本地模型"在弱信号 regime 下不成立
5. **文档化 DP 超参与范围**：代码侧溢出已修复；仍需给出经验证的 `clip_norm`/`noise_multiplier` 推荐区间，并声明该机制为实验性客户端更新扰动
6. 在正文中**明确本稿相对 *Computing* 108:32 (2026) 前作的新增贡献**（编辑部明确要求）

---

## 附：本次核查所用命令

```bash
# 在 py38 中对已安装 0.5.0 复现缺陷（site-packages 生效）
cd C:/Users/skyee/federated_survival/audit
D:/anaconda3/envs/py38/python.exe repro_050.py

# 在 py38 中对仓库 0.7.0 验证修复
cd C:/Users/skyee/federated_survival/federated-survival-v0.7.0-software-paper/federated-survival-v0.7.0-software-paper
D:/anaconda3/envs/py38/python.exe -m pytest -q
D:/anaconda3/envs/py38/python.exe experiments/audit_reviewer_checks.py
D:/anaconda3/envs/py38/python.exe experiments/audit_dp_effect.py
```
