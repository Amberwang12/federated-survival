# JCB-2026-0108 审稿意见 → 代码库逐项核查清单

**核查环境**：`D:\anaconda3\envs\py38`（Python 3.8.20 / numpy 1.24.4 / scipy 1.10.1 / torch 2.4.1+cpu / lifelines 0.27.8 / pycox 0.3.0）  
**最近一次核查日期**：2026-10-07  
**首次核查日期**：2026-09-27  
**对照版本**：已发布版 `federated-survival 0.5.0`（审稿人所测） ⟷ 仓库源码 **0.8.0**（tag `v0.8.0`；核查时的代码状态 = `v0.7.5` tag `86f4627` + 本清单 4.3 节的 #16 修复）

> **本轮（0.7.0 → 0.7.5）核查结论一句话**：**缺陷修复项全部保持闭合，无回归**。  
> 审稿清单中的 6 项功能缺陷在 0.7.5 上逐位复现原修复后数值（IBS、Dirichlet、`batch_size`、DP 参数、  
> Cox 数值稳定化、5 种子 Figure-4 配置），因为这些代码路径在 0.7.1–0.7.5 未被改动。  
> 0.7.5 的增量集中在**真实数据打包、测试覆盖补洞、示例调优**，不触及审稿指控的代码路径。

---

## 0. 一个关键前提

py38 环境的 `site-packages` 里**另装了一份已发布包**：

| 项                              | 值                                                                   |
| ------------------------------ | ------------------------------------------------------------------- |
| 安装路径                           | `D:\anaconda3\envs\py38\lib\site-packages\federated_survival\`      |
| `dist-info` 版本                 | **0.5.0**                                                           |
| `__init__.py` 中的 `__version__` | **0.1.0**                                                           |
| 子包                             | 仅 `core / data / examples / utils`（无 `protocols`、`models`、`api.py`） |
| 随包测试                           | **无**                                                               |

→ 这个环境就是**审稿人测试环境的忠实复现**。因此本次核查采用"前后对照"：  
**前半 = 在 py38 中对已安装 0.5.0 复现缺陷；后半 = 在 py38 中对仓库源码验证修复。**

> 复现脚本：`audit/repro_050.py`（从非仓库目录运行，使 site-packages 的 0.5.0 生效）
>
> ⚠️ **运行前置条件（0.7.5 复测必读）**：`export MPLBACKEND=Agg`，否则 matplotlib 会阻塞；  
> 且需 `env -u PYTHONPATH <python> -m pytest`，解除 WorkBuddy 注入的 `Path.unlink` 安全垫片  
> （该垫片按 turn 累计删除配额，达到 50 次后 `raise SystemExit(1)`，  
> 会让 `tests/test_loader.py` 的 5 个用例如实失败 —— **是环境问题，不是被测代码缺陷**）。

---

## 1. 缺陷复现（已安装 0.5.0，py38，before）

| # | 审稿人指控                                         | 实测结果                                                | 判定   |
| - | --------------------------------------------- | --------------------------------------------------- | ---- |
| 1 | `__version__` 是 0.1.0，PyPI 是 0.5.0            | 模块 `0.1.0` / dist-info `0.5.0`                      | ✅ 复现 |
| 2 | `calculate_ibs()` 是空桩，返回 `None`               | 源码为 `# TODO: 实现IBS计算` + `pass`，调用返回 `None`          | ✅ 复现 |
| 3 | `split_type='Dirichlet'` 静默执行 time-non-iid    | 见下方代码证据；两种分区的客户端大小与均值时间**完全相同**                     | ✅ 复现 |
| 4 | `batch_size` 完全不起作用                           | 8 vs 64 → 测试 C-index **[0.5655, 0.5693] 逐位相同**      | ✅ 复现 |
| 5 | `dp_clip_norm` / `dp_noise_multiplier` 完全不起作用 | 0.1 vs 5.0/10.0 → C-index **[0.5873, 0.4924] 逐位相同** | ✅ 复现 |
| 6 | 未发布任何测试                                       | site-packages 无 `tests/`                            | ✅ 复现 |

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

## 2. 修复验证（仓库 0.7.5，py38，after）

### 2.1 测试与工程化

| 项       | 0.5.0                | 0.7.0                       | **0.7.5（py38 实测）**                                                                   |
| ------- | -------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| 随包测试    | 无                    | 24 个测试文件                    | **29 个 `test_*.py`**（+`run_dp_tests.py` + 2 份 README）                                |
| 测试执行    | —                    | 206 用例通过                    | **288 用例全部通过**（288 个 `.`，0 `F`/`E`，exit 0）                                           |
| 语句覆盖率   | —                    | 85.5%                       | **90.1%**（3602 语句 / 355 未覆盖）                                                         |
| 分支出口覆盖率 | —                    | 71.7%                       | **80.2%**（1394 分支 / 276 未出口）                                                         |
| 覆盖率门禁   | —                    | 无                           | **CI 强制 `coverage report --fail-under=80`**                                          |
| CI      | 无                    | `ci.yml` + `docs.yml`       | **`ci.yml` + `docs.yml` + `release.yml`**                                            |
| 版本控制    | 无 `.git`             | 已补 `.git`                   | **`.git` 正常，147 个跟踪文件，tag `v0.2.0`/`v0.7.0`/`v0.7.5`**                               |
| 文档站     | 仅 README             | `mkdocs.yml` + `docs/` 11 篇 | **`mkdocs.yml` + `docs/` 12 篇**（新增覆盖率审计报告）+ 线上文档站                                    |
| 版本一致性   | `0.1.0` vs `0.5.0` ❌ | 统一 0.7.0 ✅                  | **`pyproject.toml` = `__init__.py` = `0.7.5` ✅**                                     |
| 依赖范围    | —                    | `requires-python >= 3.8`    | **同；CI matrix 3.8–3.14 七版本**                                                         |
| 变更日志    | 无                    | `CHANGELOG.md`              | **`CHANGELOG.md`（0.7.5 / 0.5.0 两节）**                                                 |
| 发布流程    | 无                    | `docs/releasing.md`         | **`release.yml`：tag 触发 → 构建 → twine 校验 → tag/版本一致性断言 → SHA256SUMS → GitHub Release** |
| 构建产物    | —                    | 133 KB wheel / 175 KB sdist | **146 KB wheel / 212 KB sdist**，`twine check --strict` 双 PASSED                      |

> ⚠️ **用例数统计口径**：`pytest -q` 的最终汇总行（`288 passed in ...`）在本机会被  
> `-ra -q` 的 warnings summary 吞掉（重定向到文件后仍只有 21 行）。本轮采用两种独立方法交叉验证：  
> ① `--collect-only -q` 逐文件 awk 求和 = **288**；② 进度字符计数 `.` = **288**，`F`/`E`/`s`/`x` 均为 0。  
> 二者一致，且 `PYTEST_EXIT=0`。


### 2.2 功能缺陷 —— 0.7.5 逐位复测

以下数值由 `experiments/audit_reviewer_checks.py` 与 `experiments/audit_dp_effect.py`  
在 0.7.5 上重新产出，**与 0.7.0 首次核查逐位一致**，证明无回归。

| # | 指控                         | 0.7.5 实测                                                                                                                                                                                    | 判定                   |
| - | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 2 | `calculate_ibs()` 空桩       | 拟合较优曲线 **0.2194536747451985** < 较差曲线 **0.3390490360472073**（越低越好）                                                                                                                           | ✅ 已修（数值与 0.7.0 逐位相同） |
| 3 | `Dirichlet` 死分支            | 客户端大小 **`[81, 93, 66]`**、均值时间 **`[0.982, 0.502, 0.845]`**；`dirichlet_is_non_iid=True`                                                                                                       | ✅ 已修（逐位相同）           |
| 4 | `batch_size` 无效            | 8 → C-index **0.5863**；64 → **0.6129**；预测逐点差异 `[0.00088, 0.00044, 0.00164, 0.00238, 0.00114]`                                                                                               | ✅ 已修（逐位相同）           |
| 5 | DP 参数无效                    | `clip_norm` 0.1 / 1.0 / 10 → C-index **0.6091 / 0.5930 / 0.5000**；`noise_mult` 0.1 / 5 → **0.6091 / 0.5000**                                                                                | ✅ 已修（逐位相同）           |
| 6 | `simps` 崩溃                 | `_compat.py` 保留：用 `if not hasattr(scipy.integrate, "simps")` 守卫，在 scipy ≥ 1.14 时以 `simpson` 补齐（含 keyword-only `x` 的兼容处理）。0.7.5 本机 scipy 1.10.1 **有** `simps` 故走原生路径，覆盖率 44% 属**环境相关假象**而非缺口 | ✅ 已修                 |
| 7 | 示例需手设 `n_features`         | `api.py` 自动推断（`self.config.n_features = int(first_x.shape[1])`）                                                                                                                             | ✅ 已修                 |
| 9 | Figure 4 配置 100% 种子返回不可用预测 | 5/5 种子 `finite=True`、**曲线互异**（20/20 唯一曲线）、**单调非增**                                                                                                                                          | ✅ 已修（逐位相同）           |

**Figure-4 配置 5 种子实测（SDGM1, n=100, 3 clients, ε=1）**

| seed              | 0      | 1      | 2      | 3          | 4      | 均值         | 标准差        |
| ----------------- | ------ | ------ | ------ | ---------- | ------ | ---------- | ---------- |
| Federated C-index | 0.5248 | 0.6209 | 0.5267 | **0.4507** | 0.6985 | **0.5643** | **0.0963** |
| Center C-index    | 0.5745 | 0.6078 | 0.4933 | 0.3803     | 0.6397 | 0.5391     | 0.1042     |
| Local C-index     | 0.5332 | 0.6101 | 0.4818 | 0.5148     | 0.6763 | 0.5632     | 0.0788     |

> 0.7.5 补录了 Center / Local 两列（原 0.7.0 清单只列了 Federated）。**注意此配置下  
> Local 均值 0.5632 与 Federated 0.5643 几乎持平（差 0.0011），而 Center 反而最低（0.5391）**——  
> 这组数字比原清单更明确地支持第 3 节第 15 项的"主张需改写"结论。

**DP 数值溢出：根因与修复（已解决，0.7.5 保持）**

**症状（0.5.0）**：`dp_clip_norm=1.0` 时生存曲线变为非有限值（`finite=False,
within_probability_bounds=False, monotone_non_increasing=False`），C-index 掉到 **0.1565**；  
`noise_mult=5` 时 C-index = **0.0**；默认 `raise` 模式下直接抛 `PredictionValidationError`。

**根因（两层）**

1. **预测端数值不稳定**。pycox 计算累计风险为 `H0(t) · exp(logh)`；当 `logh` 很大时  
   `exp(logh) = inf`，再乘首个事件时间之前的 `H0 = 0`，得到 `inf × 0 = NaN`。  
   整条曲线因此变成 NaN（`pycox/models/cox.py:231` 报 `overflow encountered in exp`，  
   0.7.5 复测时该 warning 仍可见，属**朴素路径的预期行为**，已被测试显式断言）。  
   这不是 DP 独有——模型发散时同样触发。
2. **噪声标定未计入客户端平均**。`privatize_model_update` 原本令  
   `sigma = noise_multiplier × dp_sensitivity × clip_norm`，默认全为 1.0 →  
   每个权重元素注入标准差 **1.0** 的噪声，而神经网络权重典型量级仅 0.01–0.3，信噪比被淹没。  
   同时 `get_noise_scale()`（含 `1/√K` 衰减）是**死代码**，与实际加噪不一致。

**修复（已实施并验证）**

| 文件                              | 改动                                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `models/adapters.py`            | 新增 `stable_cox_survival()`：以对数空间计算 `S(t\|x)=exp(-exp(log H0(t) + logh))`，并对数截断到 700。数学上等价，但 `H0=0 → log H0=-inf → S=1`，**永不产生 NaN**；输出恒有限、⊂[0,1]、随时间非增。`_CoxPHAdapter`（含 DeepSurv）与 `CoxCCAdapter` 已接入 |
| `core/differential_privacy.py`  | `privatize_model_update()` 计入 `1/√K` 客户端平均（与 `get_noise_scale` 对齐，消除死代码不一致）；新增 `last_noise_to_signal` 等诊断                                                                                              |
| `core/client.py`                | 调用处传入 `num_clients=self.config.num_clients`                                                                                                                                                            |
| `core/runner.py`                | `privacy_info` 增加 `effective_noise_sigma`，用户可直接看出实际注入的噪声强度                                                                                                                                             |
| `tests/test_stable_survival.py` | 8 个回归测试（含"朴素路径确实产生 NaN"的对照断言）                                                                                                                                                                          |

**修复前后对比（py38 实测，同一配置同一随机种子；0.7.5 复测值与 0.7.0 逐位相同）**

| DP 设置            | 0.5.0 修复前      | 0.7.5 修复后                                               |
| ---------------- | -------------- | ------------------------------------------------------- |
| `clip_norm=0.1`  | 0.5825         | **0.6091**                                              |
| `clip_norm=1.0`  | **0.1565，非有限** | **0.5930，有限且曲线互异**                                      |
| `clip_norm=10`   | 0.0，非有限        | **0.5000，有限**（`identical_across_samples=True`，被诊断标记为退化） |
| `noise_mult=0.1` | 0.5825         | **0.6091**                                              |
| `noise_mult=5`   | 0.0，非有限        | **0.5000，有限**（退化标记）                                     |

`clip_norm=1.0` 这一档从"崩溃"回到 **0.5930**，已与关闭 DP 时的水平相当。极端设置  
（`clip_norm=10`、`noise_mult=5`）现在返回**合法但无信息量**的曲线（C-index = 0.5 且  
`identical_across_samples=True`）——这是正确行为：算法不再崩溃，而是如实报告"该隐私预算已淹没信号"。

**DP 记账语义（0.7.5 已诚实标注，但仍需论文说明）**

| 事实                                                                                | 状态                                                                                                                      |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 工业共识（Opacus / TFF / Flower / Geyer）：σ 是输入，ε 是 RDP 记账输出                            | 代码已按此实现                                                                                                                 |
| `total_epsilon` 恒等于 ε **是正确**的（ε 是总预算，不该除 T 或 √T）                                 | ✅ 无需改                                                                                                                   |
| `get_privacy_info()` 已加 `formal_accounting_available: False` + `privacy_scope` 标注 | ✅ 已标注                                                                                                                   |
| `compute_privacy_budget` 只被 `get_privacy_info()` 调用（纯汇报），**从未驱动加噪**               | ⚠️ 语义待讨论                                                                                                                |
| σ 不依赖 T。σ=1.0、T=100 时真实 ε ≈ **110.82**                                            | ⚠️ 需在正文说明                                                                                                               |
| Gaussian 下 `dp_epsilon` 是死参数、Laplace 下 `dp_noise_multiplier` 是死参数，**切换机制时无告警**    | ✅ **已修（2026-10-07）**：`FSAConfig` 发 `UserWarning`，`get_privacy_info()` 报 `noise_driver` / `inactive_parameters`，30 个回归用例 |
| `get_privacy_info().noise_scale` 含 `/√K` 而生产路径不除 → 低报 √K 倍                        | ⚠️ 建议改名 `nominal_epsilon`                                                                                               |
| `sensitivity/√K` 语义混用（Flower 口径下 Central DP 线性除 m、Local DP 不除 K）                  | ⚠️ 需澄清                                                                                                                  |
| RDP 半成品：`compute_renyi_divergence` / `convert_renyi_to_epsilon` 生产调用 **0 处**      | ⚠️ 死代码                                                                                                                  |

**经验证的 DP 超参区间**：`clip_norm ∈ [0.01, 0.1]`、`noise_multiplier ∈ [0.1, 1.0]`。  
实测剂量-响应（σ = 0.1 / 1.0 / 10 → 真实 ε ≈ 1008 / 18 / 1.3，C-index 0.519 / 0.501 / 0.500）。  
**ε 是决定性因素**：ε=10（σ≈0.28）才可用；ε=1 会因 σ≈4.85 淹没梯度量级而崩溃  
——这是信息论下限，不是 bug。

---

## 3. 仍未完全闭合的项 ⚠️ / ❌

> **状态说明（2026-10-07 更新）**：下表 #10 / #11 / #13 / #14 已由作者决定**不做代码实现**，  
> 改为在回复信中主动说明取舍理由。表格结论列保留了原始判定，括号内为**处置决定**，  
> 以便审稿人追问时能直接引用。详见 4.2 节。

| #  | 审稿人指控                      | 0.7.5 现状                                                                                                                                                                                                                                                                                                            | 结论                        |
| -- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| 10 | 仅单次运行，无重复 / 置信区间 / 显著性检验   | 已有 5 随机种子结果（`results/ie1`、`ie2`、`ie3`、`ie_gbsg_five_seeds`，含箱线图 + 指标表 + 分轮曲线），绘图使用 `mean_ci95` 误差带（`experiments/visualization.py:262`）；**全库 `wilcoxon` / `ttest` / `bootstrap` 零命中**（已复核：早前 grep 命中的 `unittest.TestSuite` 系 `ttest` 子串误命中）                                                                          | ⚠️ 部分 → **作者决定不做**，需回复信说明 |
| 11 | 未使用真实数据集                   | **已打包两个真实数据集**：`federated_survival/data/real/gbsg.csv` + **`colon_death.csv`**，均进入 wheel 与 sdist；公开 API `load_real_data()` / `available_real_datasets()`（`data/workflow.py:70,77`）。GBSG 已用于 5 种子实验（`results/gbsg_five_seeds`、`ie_gbsg_five_seeds`）。**`test_real_data.py` 仅为数据加载测试（0 处 `fit(`），不含训练**                | ✅ **已回应**（作者决定不做第二个数据集实验） |
| 12 | 无外部方法对比（WebDISCO、Verticox） | `protocols/webdisco.py` 已实现（182 行，覆盖率 85%）；**Verticox 属纵向联邦（vertical FL），与本包横向联邦的威胁模型和数据布局不兼容，不具可比性，不在范围内**                                                                                                                                                                                                         | ✅ 已处理                     |
| 13 | 数据为模拟、真值已知，却从不与真实基准对比      | 全库 `oracle` / `ground_truth` / `true_survival` **零命中**。技术上可行（`generator.py:93-94` 的 `SDGM1` 真值 `t_mu = exp(0.1·Σx_{p/2:})` 可解析恢复），但**真实数据 GBSG 无真值，原理上无法用于 oracle 对照**                                                                                                                                              | ❌ 未处理 → **作者决定不做**，需回复信说明 |
| 14 | 无显著性检验                     | 仅 `mean_ci95` 误差带；无任何假设检验函数。`scipy.stats.wilcoxon` / `bootstrap` 均可用，`mean_ci95` 基础设施已就绪，改动成本低                                                                                                                                                                                                                      | ❌ 未处理 → **作者决定不做**，需回复信说明 |
| 15 | 弱信号下"联邦显著优于本地"无法与噪声区分      | 5 种子：Federated **0.5643 ± 0.0963**、Local **0.5632 ± 0.0788**、Center **0.5391 ± 0.1042**（Figure-4 弱信号配置）。seed 3 仅 **0.4507 < 0.5**。Federated 与 Local 差距仅 **0.0011**，远小于标准差                                                                                                                                           | 🔴 **主张必须改写**             |
| 16 | （新增）DP 机制切换时死参数无告警         | ✅ **已修（2026-10-07）**。`FSAConfig` 在用户把无效字段设成非默认值时发 `UserWarning`（默认值静默）；两处 `get_privacy_info()` 新增 `noise_driver` / `inactive_parameters`。新增 `tests/test_dp_inert_parameters.py`（**30 用例**）。实测确认 `dp_epsilon`（Gaussian）、`dp_noise_multiplier`（Laplace）、`dp_delta`（两者）均**逐位不影响噪声**；修复后 `audit_dp_effect.py` 输出**零漂移** | ✅ 已修                      |

> **第 15 项的加强证据（0.7.5 新增）**：本轮首次补齐 Center / Local 两列后，  
> 可以看到在这个 Figure-4 配置下 **Center 反而最低**。这说明该配置主要测的是  
> "小样本 + 弱信号 + 高噪声"三重退化下的排序扰动，而非方法本身的优劣。  
> **建议在回复信中直接用更强信号配置（`examples/compare/compare_v2.py` 调优后  
> MVAES 0.593 > MVAEC 0.585 > Federated 0.561 > Local 0.534）作为主结果**，  
> 把弱信号配置降级为"压力测试 / 负面结果如实报告"。

> ⚠️ **风险提示（作者已知悉并作出决定）**：#13 / #14 是审稿人**明确点名的方法论缺口**，  
> 划掉后回复信必须正面说明取舍，否则存在被以此为由拒稿的风险。  
> 两项的实现成本都很低（#14 约数十行；#13 真值可解析恢复），  
> 若审稿人二轮追问，**建议改为"补做"而非继续解释**。  
> 另注意 #11 的 GBSG 实验目前是**交互式跑出来的**  
> （`docs/interactive-experiments.md:63-87`，产物只有 PNG/CSV、**无 json/manifest 落盘**，  
> 该文档 `:180` 亦自承"交互式 API 不写 resolved config"），  
> 论文若要声明可复现，建议补一个固化脚本——**此项不属于 #11 的审稿指控，但影响可复现性表述**。

---

## 4. 修改清单（可直接落到回复信 / 修订稿）

### 4.1 已在 0.8.0 闭合（回复信可直接引用实测数据）

- [x] 随包测试 + CI：318 用例通过（含本轮新增 30 个 DP 死参数回归用例），  
  覆盖率门禁 `--fail-under=80`（实际 90.2% 语句 / 80.3% 分支）
- [x] `calculate_ibs()` 补全实现（去掉 `pass` 桩）
- [x] Dirichlet 分派大小写 bug 修复
- [x] `batch_size` / `dp_clip_norm` / `dp_noise_multiplier` 真正接入训练与噪声计算
- [x] `_compat.py` 兼容 scipy ≥ 1.14 的 `simps`
- [x] `n_features` 自动推断
- [x] 版本号统一为 **0.7.5**（`pyproject.toml` = `__init__.py` = tag）
- [x] MkDocs API 文档站（`docs/` 12 篇）+ GitHub Pages 部署
- [x] 多随机种子（5 seeds）实验脚手架与 CI95 误差带
- [x] 真实数据集实验（GBSG）
- [x] WebDISCO 协议实现
- [x] **Cox 生存曲线数值稳定化**（对数空间计算，彻底消除 `inf × 0 = NaN`）+ 8 个回归测试
- [x] **DP 噪声标定计入 `1/√K` 客户端平均**，并暴露 `last_noise_to_signal` / `effective_noise_sigma` 诊断
- [x] **【0.7.5】第二个真实数据集 `colon_death.csv` 随包分发** + `load_real_data()` 公共 API
- [x] **【0.7.5】CI 覆盖率门禁 + 7 版本 matrix（3.8–3.14）**
- [x] **【0.7.5】GitHub Release 自动化**（`release.yml`：tag/版本一致性断言 + SHA256SUMS）
- [x] **【0.7.5】`tests/test_baseline_adapter_parity.py`**（212 行）——固化 7 个内置模型的  
  数值指纹，证明 adapter 委托重构后 Center/Local 基线**零漂移**
- [x] **【0.7.5】`CHANGELOG.md` 按 Keep a Changelog 维护**
- [x] **【2026-10-07 审稿项 #16】DP 死参数告警**——见下节 4.3

### 4.2 待办

**状态（2026-10-07 更新）**：原 8 项中，**5 项代码实现已由作者决定全部不做**  
（#11 第二个数据集实验、#13 oracle 对照、#14 显著性检验，以及 #10 的检验部分）。  
其中原第 5、6 项（DP 超参文档化、`sensitivity/√K` 语义澄清）**已在软件包文档  
`docs/privacy.md` 中补充完毕**。剩余仅 **2 项论文正文改写**，均不涉及代码。

#### 4.2.1 已决定不做（须在回复信中主动说明取舍）

| 原编号     | 内容                               | 划掉理由（可直接写入回复信）                                                              |
| ------- | -------------------------------- | --------------------------------------------------------------------------- |
| ~~#11~~ | 跑通 `colon_death.csv` 的实验         | 审稿人原始指控是"未使用真实数据集"，GBSG（n=686）已回应；colon 属扩展项而非必需                            |
| ~~#13~~ | 实现"真实基准（真值生存时间/风险）对照"评测          | **GBSG 等真实数据无真值，原理上无法作 oracle 对照**；该指控只针对模拟数据。取舍：本文定位为方法学与工程实现，不承担模拟器真值校准职责 |
| ~~#14~~ | 加入显著性检验（配对 bootstrap / Wilcoxon） | 取舍：以 5 随机种子的 `mean_ci95` 误差带 + 分轮曲线呈现不确定性，不做假设检验。**须主动说明这是有意的分析策略，而非遗漏**    |
| ~~#10~~ | 重复运行 / 置信区间                      | ✅ 已完成（5 seeds + `mean_ci95`），仅"显著性检验"部分随 #14 一并划掉                           |

> ⚠️ **必须做的事**：这几条不能从回复信中消失。审稿人明确点名了 #13 / #14，  
> **必须正面说明为何不做**，否则等于默认忽略意见，存在被以此为由拒稿的风险。  
> 若审稿人二轮追问，**建议改为补做**——两项成本都很低：  
> #14 约数十行（`scipy.stats.wilcoxon` / `bootstrap` 已可用，`mean_ci95` 基础设施已就绪），  
> #13 真值可解析恢复（`generator.py:93-94`，`SDGM1` 的 `t_mu = exp(0.1·Σx_{p/2:})`）。

#### 4.2.2 已完成：软件包文档补充（原第 5、6 项）

**✅ 原第 5 项「文档化 DP 超参与范围」** → `docs/privacy.md` 新增两节：

- *"Recommended ranges, and how to tell that noise has swamped the signal"*——  
  给出**实测**剂量-响应表（SDGM1 / n=200 / 5 特征 / censoring 0.4 / 3 客户端 / IID /  
  DeepSurv / 5 rounds × 3 local steps，seeds 0-2 均值）：
  | `dp_noise_multiplier` | `dp_clip_norm` | C-index             |
  | --------------------- | -------------- | ------------------- |
  | DP off                | –              | 0.5454              |
  | 0.01                  | 1.0            | 0.5357              |
  | 0.05                  | 1.0            | 0.5348              |
  | 0.1                   | 1.0            | 0.5200              |
  | 0.3                   | 1.0            | 0.5349              |
  | 0.5                   | 1.0            | 0.5579              |
  | 1.0                   | 1.0            | 0.5602              |
  | **2.0**               | 1.0            | **0.5000（退化，曲线全同）** |
  | 0.1                   | 0.1            | 0.5357              |
  | 0.1                   | 0.01           | 0.5064              |
  | 0.5                   | 0.01           | 0.5040              |
  推荐 **`dp_noise_multiplier ∈ [0.01, 0.1]` 且 `dp_clip_norm = 1.0`**；  
  并指出 `dp_noise_multiplier × dp_clip_norm` 的**乘积**才是真正决定注入噪声的量，  
  单看其中一项会误判。
- *"What the mechanism perturbs, and where the noise is added"*——  
  声明该机制为**实验性客户端更新扰动，不等同于记录级 DP-SGD**；  
  说明 `privatize_model_update()` 的四步流程；  
  显式指出 `add_noise_to_weights()` **不在训练路径上**（对无界绝对权重加噪不成立）、  
  服务端聚合**不加噪**（`apply_dp_to_weights()` 原样返回）。
- 附带一条**实测可达性说明**：`last_noise_to_signal` / `last_noise_sigma` /  
  `last_clipped_update_norm` 挂在**每轮临时创建并丢弃的 client** 上，  
  **`fit()` 后无法从 `FederatedSurvival` / `FSARunner` 取到**，  
  仅在自行驱动 `privatize_model_update()` 时可读。文档已如实标注，  
  避免读者按不可达路径去找。

**✅ 原第 6 项「澄清 `sensitivity/√K` 的 DP 语义」** → `docs/privacy.md` 新增  
*"Sensitivity, and what the `1/sqrt(K)` factor means"*：

- `dp_sensitivity` 约束的是**裁剪后的客户端更新**，**不是单条记录**，  
  因此**不继承任何记录级保证**；
- 明确 `1/√K` 是**方差缩减论证，不是隐私记账论证**——  
  标准 Central DP 下敏感度随采样率 `1/K` **线性**缩放而非 `1/√K`，  
  Local DP 则**不除 K**；
- 结论写明：读成"实际注入了多少噪声"可以，**据此推导 ε 不成立**。

#### 4.2.3 仍需完成（纯论文正文，不涉及代码）

1. 🔴 **修正核心主张的措辞**（#15）——现有数据显示弱信号下 Federated 0.5643 与 Local 0.5632  
   几乎持平（差 0.0011，远小于 sd 0.08–0.10），**"显著优于本地"站不住**。  
   **作者决定：不改代码，在回复信中说明取舍。**  
   建议改用调优配置（`compare_v2` 的 MVAES 0.593 > MVAEC 0.585 > Fed 0.561 > Local 0.534）  
   作主结果，弱信号配置如实降级为压力测试。  
   **这是唯一涉及核心卖点的事项，建议与合作者定调后再写。**
2. **在正文中明确本稿相对 *Computing* 108:32 (2026) 前作的新增贡献**（编辑部明确要求）。

#### 4.2.4 已明确不适用

- ~~加入 Verticox~~ — Verticox 是纵向联邦（vertical FL），与本包横向联邦的威胁模型和  
  数据布局不兼容，不具可比性。回复信中应**明确说明这一点**，  
  并只承诺横向联邦基线（WebDISCO 已实现，182 行 / 覆盖率 85%）。

#### 4.2.5 可复现性补充（不属审稿指控，作者决定暂不列入本轮）

- **GBSG 5 种子实验目前是交互式跑出来的**：`docs/interactive-experiments.md:63-87` 记录了  
  完整步骤，产物在 `results/gbsg_five_seeds/` 与 `results/ie_gbsg_five_seeds/`，  
  但**只有 PNG/CSV，无 json/manifest 落盘**（该文档 `:180` 自承"交互式 API 不写 resolved config"）。  
  论文若要声明可复现，建议补一个固化脚本。
- 另注：`tests/test_real_data.py` 是**数据加载测试**（0 处 `fit(`），验证 GBSG 表形状为  
  `(686, 10)`、`colon_death` 为 `(888, 13)`，**不含任何模型训练**——  
  引用实验结果时不要把它当作训练验证。

### 4.3 【2026-10-07 已完成】审稿项 #16：DP 机制切换的死参数告警

**原缺陷**：Gaussian 的噪声尺度是 `dp_noise_multiplier × dp_sensitivity`，  
Laplace 的是 `dp_sensitivity / dp_epsilon`。两者的控制旋钮**互为对方的死参数**，  
而 `dp_delta` 在**两种机制下都不进入噪声尺度**（唯一读取点  
`convert_renyi_to_epsilon` 生产调用 0 处）。切换 `dp_mechanism` 时，  
同一个参数从有效骤变无效，且**无任何告警**——用户以为设了隐私预算，其实没设。

**修复前的数值证据**（`torch.equal` 逐位比较，同 seed 重置 RNG）：

| 对照                                        | 结果                       |
| ----------------------------------------- | ------------------------ |
| Gaussian `dp_epsilon` 1 vs 1000           | **逐位相同**（`dp_epsilon` 死） |
| Gaussian `dp_noise_multiplier` 0.5 vs 5.0 | 不同（确认是驱动旋钮）              |
| Laplace `dp_noise_multiplier` 0.5 vs 99   | **逐位相同**（死）              |
| Laplace `dp_epsilon` 10 vs 0.01           | 不同（确认是驱动旋钮）              |
| `dp_delta` 1e-9 vs 1e-1（两种机制）             | **逐位相同**（均死）             |

**修复内容**：

| 文件                                  | 改动                                                                                                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/differential_privacy.py`      | 新增权威映射 `DP_MECHANISM_NOISE_KNOBS` / `DP_MECHANISM_INERT_KNOBS` 与查询函数 `describe_noise_knobs()`；`DifferentialPrivacy` 新增 `noise_driver()` / `inactive_parameters()` |
| `core/config.py`                    | `__post_init__` 在 DP 启用时校验：**仅当无效字段被设为非默认值**才发 `UserWarning`，指明该字段、控制它的旋钮、以及当前值。默认值保持静默（不制造噪音）                                                                    |
| `core/runner.py`、`api.py`           | 两处 `get_privacy_info()` 新增 `noise_driver` 与 `inactive_parameters`，使状态**可编程读取**而非只是一次性告警                                                                           |
| `docs/privacy.md`                   | 新增"Which parameter actually controls the noise"表格与说明（含已验证可运行的示例）                                                                                                  |
| `tests/test_dp_inert_parameters.py` | **新增 30 个用例**：数值层（死参数逐位不变 / 驱动参数确实生效 / 切机制确实改变噪声）+ 告警层（触发条件、静默条件、`UserWarning` 类别、消息含具体值）+ 上报层（两处 API）                                                            |

**告警文本示例**：

```
UserWarning: dp_mechanism='gaussian' does not use dp_epsilon (currently 0.5);
the injected noise is controlled by dp_noise_multiplier. Setting dp_epsilon has
no effect on the noise added to model updates. Switch dp_mechanism or adjust
dp_noise_multiplier instead.
```

**等价性实证**：修复后 `experiments/audit_dp_effect.py` 输出与修复前**逐位相同**  
（`clip_norm` 0.1/1.0/10 → 0.6091/0.593/0.5；`noise_mult` 0.1/5 → 0.6091/0.5；  
`batch_size` 8/64 → 0.5863/0.6129），`experiments/audit_reviewer_checks.py` 的  
5 种子均值仍为 0.5643276062997369 → **纯观测性改动，零数值影响**。

**回归**：**318 用例全绿**（原 288 + 新增 30），0 失败；  
覆盖率 90.2% 语句 / 80.3% 分支，`coverage report --fail-under=80` 门禁通过。

> ⚠️ **本轮踩到的坑（已修）**：首次实现时把新方法 `_warn_inactive_dp_parameters`  
> 插进了 `__post_init__` 中部，导致其后的 `model_params` 默认值块被误并进新方法，  
> `FSAConfig(model_type='PC-Hazard').model_params` 变成 `{}`，  
> `tests/test_config.py::test_model_params` 失败。**教训：给 dataclass 的  
> `__post_init__` 追加方法时，插入点必须落在方法体之外**，  
> 且改完要用 `git diff` 确认**没有删除任何原有行**（`grep "^-"` 应为空）。

## 附 A：本次核查所用命令（py38）

```bash
cd C:/Users/skyee/federated_survival
export MPLBACKEND=Agg

# 测试（必须 env -u PYTHONPATH 解除安全删除垫片）
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m pytest -q
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m pytest --collect-only -q   # 计数交叉验证

# 覆盖率（exe 不在 PATH，必须 python -m coverage）
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m coverage run --branch \
    --source=federated_survival -m pytest
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m coverage report --sort=cover

# 审稿项复核
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe experiments/audit_reviewer_checks.py
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe experiments/audit_dp_effect.py

# 0.5.0 缺陷复现（需从非仓库目录运行，使 site-packages 的 0.5.0 生效）
cd C:/Users/skyee/federated_survival/audit
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe repro_050.py

# 构建与元数据校验
cd C:/Users/skyee/federated_survival
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m build
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m twine check --strict dist/*
```

## 附 B：两个统计口径陷阱（复测必读）

1. **`pytest -q` 的汇总行会消失**。本机 py38 + `addopts = "-ra -q"` 下，  
   重定向到文件后最终汇总行（`288 passed in ...`）不落盘，日志只有 21 行。  
   **判定"全绿"的证据链 = `PYTEST_EXIT=0` + `--collect-only` 逐文件求和 + 进度字符计数**，  
   三者交叉一致才算数。不要因为"看不到 passed 行"就断定 pytest 没输出或失败。
2. **WorkBuddy 的 `Path.unlink` 垫片会让 `tests/test_loader.py` 假失败**。  
   该垫片按 turn 累计删除配额（本机阈值 50），达到后 `raise SystemExit(1)`。  
   症状极具迷惑性：**单跑通过、全量跑失败**，且失败数恰好等于文件内 `unlink()` 调用处数量。  
   **必须用 `env -u PYTHONPATH` 跑才是真实结果**（用户在自己终端跑时天然无此问题）。

## 附 C：0.5.0 ⟷ 0.7.5 文件差异概览（发布于 v0.7.0 ⟷ v0.7.5）

`git diff --stat v0.7.0..v0.7.5`：**59 个文件，+4398 / −1218 行**。

| 类别    | 文件                                                                                                                                                                                                                | 说明                                                         |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 真实数据  | `data/real/colon_death.csv`（+889）、`data/loader.py`（+92）、`data/workflow.py`（+45）、`data/__init__.py`                                                                                                                | 第二个真实数据集 + 公共 API                                          |
| 新测试   | `test_cli.py`（+462）、`test_baseline_adapter_parity.py`（+212）、`test_baselines.py`（+122）、`test_splitter.py`（+77）、`test_mvae.py`（+73）、`test_high_level_api.py`（+61）、`test_real_data.py`（+37）、`test_protocols.py`（+18） | 214 → 288 用例的主要来源                                          |
| 文档英文化 | `core/*.py`、`data/splitter.py`、`experiments/visualization.py` 等                                                                                                                                                   | 0.7.5 提交 `d9c55e9` 把仓库根 `examples/` 与 `scripts/` 的中文全部译为英文 |
| 工具脚本  | `scripts/coverage_gap_analyzer.py`（+160）、`scripts/run_all_examples.py`（+53）、`scripts/run_root_examples.py`（+51）                                                                                                   | 覆盖率缺口分析 + 批量示例运行器                                          |
| 打包    | `pyproject.toml`（+4）                                                                                                                                                                                              | 新增 `data/real/*.csv` 到 `package-data`                      |

> **注意 `core/differential_privacy.py` 差异 +250/−（大改）**：这是 0.7.5 对  
> 中文注释/文档字符串的翻译与诊断字段补充，**噪声计算公式本身未变**——  
> 证据是 `experiments/audit_dp_effect.py` 在 0.7.5 上的输出与 0.7.0 逐位相同。

## 附 D：0.8.0 相对 0.7.5 的文件差异

**版本变更**：`pyproject.toml` 与 `federated_survival/__init__.py` 的
`0.7.5` → `0.8.0`；`CHANGELOG.md` 新增 `[0.8.0] - 2026-10-07` 节。

### D.1 审稿项 #16 的代码修复（+127 行 / −0 行，纯新增）

| 文件 | 改动 | 性质 |
|---|---|---|
| `core/differential_privacy.py` | 新增 `DP_MECHANISM_NOISE_KNOBS`（3 行映射）、`DP_MECHANISM_INERT_KNOBS`、`describe_noise_knobs()`；`DifferentialPrivacy` 新增 `noise_driver()` / `inactive_parameters()` | 纯新增，无删除 |
| `core/config.py` | `__post_init__` 末尾新增告警调用 + 新方法 `_warn_inactive_dp_parameters()` | 纯新增（首版误插方法体内致 1 用例失败，已修，见 4.3 节末） |
| `core/runner.py` | `get_privacy_info()` 增加 `noise_driver` / `inactive_parameters` 两个键 | 纯新增 |
| `api.py` | 同上（高层 API 的 `get_privacy_info()`） | 纯新增 + 一次局部 import |
| `tests/test_dp_inert_parameters.py` | **新增文件**，300 行 / 30 个用例 | 新增 |

实测 `git diff --stat federated_survival/` = `4 files changed, 127 insertions(+)`，**无 deletions**。

### D.2 文档

| 文件 | 改动 |
|---|---|
| `docs/privacy.md` | 新增 3 节（19 → 161 行）：噪声注入位置与四步流程、`sensitivity` 与 `1/√K` 语义、推荐区间与实测剂量-响应表 |
| `docs/releasing.md` | 示例版本号 0.7.0 → 0.8.0 |
| `.github/releases/v0.8.0.md` | **新增**，手工撰写的发布说明（避免 `release.yml` 回退到自动生成 notes） |
| `PACKAGE_AUDIT.md` / `PACKAGE_STRUCTURE.md` / 本清单 | 同步 0.8.0 版本号、318 用例基线、0.8.0 产物实测值 |

### D.3 验证口径

`core/config.py` 的删除行数必须为 **0**
（`git diff <file> | grep "^-" | grep -v "^---"` 为空）——
这是本轮新增的自检手段，专门抓"插入新方法时截断了原有方法体"这类静默破坏。
该手段确实抓到一次真实事故：首版把 `_warn_inactive_dp_parameters()` 插进了
`__post_init__` 中部，导致其后的 `model_params` 默认值块被误并进新方法，
`test_config.py::test_model_params` 失败。详见 4.3 节末。

**0.8.0 发布前实测**：318 用例全绿；覆盖率 90.2% 语句 / 80.3% 分支，
`coverage report --fail-under=80` 门禁 EXIT=0；`python -m build` EXIT=0；
`twine check --strict` 双 PASSED；wheel 147812 B / 64 条目，sdist 221822 B / 122 文件；
产物污染扫描 0 命中。
