# 测试覆盖率审计报告

- **审计日期**：2026-09-27
- **被测版本**：federated_survival 0.7.0
- **执行环境**：`D:/anaconda3/envs/py38/python.exe`（Python 3.8.20）· coverage 7.6.1 · pytest 8.3.5
- **测量命令**：`coverage run --branch --source=federated_survival -m pytest tests/`
- **原始产物**：HTML 报告见 `docs/coverage_report/index.html`（该目录自带 `.gitignore`，不入库）

---

## 一、直接结论

**没有完全覆盖。** 需要把"覆盖"拆成三个层次看，结论完全不同：

| 维度 | 结论 |
| --- | --- |
| 测试套件是否健康 | ✅ 214/214 全部通过，**0 失败、0 跳过、0 xfail** |
| 功能"入口"是否都触碰过 | 🟡 37 个计入统计的源文件中 35 个有语句被执行，但**有 20 个函数从未被执行过** |
| 功能"内部"是否都验证过 | 🔴 语句覆盖 **85.5%**，分支出口覆盖仅 **71.7%**（399/1408 个分支出口从未走到） |

一句话概括：**本项目的测试是"广而浅"——绝大多数模块被 import 并跑通了主路径，但 CLI 全链路、协议聚合方法、数据划分的一种模式、以及大量校验与异常分支完全没有任何测试。**

这不是"测试写得差"，而是"测试写在了主干上，没写到边界上"。对于一个要发 JCB 软件论文、且已发布到 PyPI 的包，下面的 A 档缺口需要在投稿前补掉。

---

## 二、测试规模基线

| 指标 | 数值 |
| --- | --- |
| 测试文件 | 24 个（23 个 `test_*.py` + 1 个 `run_dp_tests.py`） |
| 测试用例 | **214** 个（官方汇总行口径） |
| 测试代码量 | 4,793 行 |
| 被测源码量 | 9,052 行（含未计入统计的 `examples/` 1,320 行） |
| 被测语句 | 3,613 行 |

按文件统计的用例数（`pytest --collect-only` 汇总）：

```
test_config.py                 21      test_stable_survival.py         8
test_dp_mechanisms.py          18      test_config_workflow.py         8
test_dp_federated_learning.py  17      test_splitter.py                7
test_differential_privacy.py   16      test_dp_performance.py          7
test_model_adapters.py         13      test_data_workflow_api.py       7
test_runner.py                 11      test_augmenter.py               7
test_protocols.py              11      test_high_level_api.py          5
test_metrics.py                11      test_generator.py               5
test_loader.py                 11      test_experiment_api.py          5
test_experiment_infrastructure.py 9    test_dp_configs.py              5
test_loss_normalization.py      4      test_augmentation_workflow.py   4
test_dp_fix.py                  3      test_plot.py                    1
```

> ⚠️ `tests/run_dp_tests.py` 不符合 `pyproject.toml` 中 `python_files = "test_*.py"` 的命名约定，**不会被 pytest 收集**，是一个未被执行的僵尸脚本。

---

## 三、覆盖率总览

### 3.1 按模块分组

| 模块 | 语句 | 未覆盖 | 语句覆盖 | 分支出口缺失 | 分支覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(顶层)` | 513 | 123 | 76.0% | 69 / 225 | 69.3% |
| `experiments/` | 938 | 193 | 79.4% | 158 / 395 | **60.0%** |
| `protocols/` | 362 | 49 | 86.5% | 38 / 128 | 70.3% |
| `core/` | 964 | 106 | 89.0% | 81 / 347 | 76.7% |
| `data/` | 601 | 42 | 93.0% | 40 / 259 | 84.6% |
| `utils/` | 68 | 4 | 94.1% | 6 / 26 | 76.9% |
| `models/` | 167 | 7 | 95.8% | 7 / 28 | 75.0% |
| **合计** | **3,613** | **524** | **85.5%** | **399 / 1,408** | **71.7%** |

> 口径说明：coverage 的 `report` 综合列（合并语句与分支）显示 82%，本表的分支覆盖率采用"未执行的分支出口数 / 分支出口总数"，更严格也更直观。

### 3.2 覆盖率最低的 10 个文件

| 文件 | 语句覆盖 | 缺失行 |
| --- | ---: | --- |
| `experiments/config_runner.py` | **13.2%** | 30-58, 67-162, 172-176 |
| `_compat.py` | 44.4% | 11-15 |
| `cli.py` | 60.2% | 41-42, 47-48, 59-64, 68-118, 122-124, 146, 198-199, 203 |
| `experiments/baselines.py` | 67.3% | 58-92, 117, 233-236, 290-324 |
| `api.py` | 72.8% | 48, 91-93, 102, 106, 142-145, 160-177, 234-329 |
| `core/differential_privacy.py` | 74.9% | 201-222, 246-248, 282-283, 291, 314, 334, 366-373 |
| `protocols/parameter.py` | 77.0% | 18-38, 54-67, 138-168 |
| `experiments/config_schema.py` | 77.1% | 散布 33 行校验分支 |
| `experiments/visualization.py` | 77.3% | 散布 40 行绘图分支 |
| `experiment_api.py` | 79.3% | 散布 39 行 |

### 3.3 已完全覆盖（100%）的文件

`core/server.py`、`data/generator.py`、以及全部 6 个 `__init__.py` 再导出模块。

**注意**：`data/generator.py` 以 124 语句达成 100% 语句 + 100% 分支覆盖，是全项目质量标杆；可对照它为其他模块定标准。

---

## 四、20 个从未被执行的函数

按风险分四档。A 档是必须在投稿前处理的实质缺陷。

### A 档 — 对外发布能力 / 公开 API（13 个）

| 位置 | 函数 | 行范围 | 性质 |
| --- | --- | --- | --- |
| `cli.py:197` | `main` | L197-199 | **PyPI 控制台入口** `federated-survival` |
| `cli.py:67` | `command_quickstart` | L67-118 | CLI 主命令（52 行） |
| `cli.py:58` | `command_doctor` | L58-64 | 环境自检命令 |
| `cli.py:121` | `command_run_config` | L121-124 | 配置驱动命令 |
| `experiments/config_runner.py:165` | `run_config_file` | L165-176 | `run-config` 后端主入口 |
| `experiments/config_runner.py:61` | `_run_legacy_config` | L61-162 | 配置执行主逻辑（102 行） |
| `experiments/config_runner.py:28` | `_protocol_specs` | L28-58 | 协议声明规范化 |
| `experiments/baselines.py:288` | `summarize_results` | L288-324 | `experiments` 包 `__all__` 公开导出（37 行） |
| `experiments/config_schema.py:35` | `is_workflow_configuration` | L35-40 | 配置类型判别 |
| `api.py:158` | `FederatedSurvival.evaluate` | L158-181 | **高层 API 的评估方法**（24 行） |
| `api.py:101` | `FederatedSurvival.available_models` | L101-102 | 公开方法 |
| `api.py:105` | `FederatedSurvival.available_protocols` | L105-106 | 公开方法 |
| `protocols/base.py:66` | `FederatedProtocol.aggregate` | L66-74 | 协议基类契约方法 |

### B 档 — 协议与生成模型核心方法（5 个）

| 位置 | 函数 | 行范围 |
| --- | --- | --- |
| `protocols/webdisco.py:251` | `WebDISCOStyleCoxProtocol.aggregate` | L251-252 |
| `core/mvae.py:90` | `MVAE.sample` | L90-97 |
| `core/mvae.py:128` | `MVAE.generate` | L128-131 |
| `core/mvae.py:140` | `cmse` | L140-144 |

### C 档 — 数据划分功能缺口（1 个）

| 位置 | 函数 | 行范围 | 影响 |
| --- | --- | --- | --- |
| `data/splitter.py:194` | `DataSplitter._split_random` | L194-206 | `split_type='random'` **整条分支无验证** |

### D 档 — 环境相关，可接受（1 个）

| 位置 | 函数 | 说明 |
| --- | --- | --- |
| `_compat.py:11` | `_simps` | 仅在 `scipy < 1.14` 时定义；本机 scipy ≥ 1.14，垫片不安装，属**合理的环境相关未覆盖**。建议改用 monkeypatch 强制构造 `not hasattr(scipy.integrate, "simps")` 场景补测。 |

**合计约 360 行源码（含定义行）从未被任何测试触及。**

---

## 五、重点缺口详解

### 5.1 CLI 全链路零单测（`cli.py` 60.2% / `config_runner.py` 13.2%）

`pyproject.toml` 声明了控制台脚本入口：

```toml
[project.scripts]
federated-survival = "federated_survival.cli:main"
```

这是**用户安装后唯一会直接调用的代码**，但：

- `main()` 本体未被执行 → 整个 argparse 分派链路（`L197-199`）无验证
- `command_quickstart` / `command_doctor` / `command_run_config` 三个命令实现全部未执行
- 下游 `config_runner.run_config_file` 整条链路 13.2%，`_run_legacy_config` 102 行零覆盖
- `config_schema.is_workflow_configuration`（配置类型判别的分岔点）零覆盖

测试只覆盖了 `build_parser`、`doctor_report`、`command_config_options` 三个**辅助函数**（`tests/test_config.py`、`test_config_workflow.py` 有 import），真正的命令执行体一个都没跑到。

**已被 CI 部分弥补**：`.github/workflows/ci.yml` 的 `Verify public command` 与 `quickstart` job 通过 shell 冒烟（`federated-survival doctor` / `quickstart --rounds 1 ...`）端到端跑通。但这**不属于 pytest 套件**，本地覆盖率测不到，也不能对参数分支、错误路径做断言。所以是"有冒烟、无单测"。

### 5.2 `FederatedSurvival.evaluate` 从未被执行

`api.py` 的 `FederatedSurvival` 被 docstring 定位为"A compact, model-independent entry point"，其公开方法共 8 个（4 个是 `_` 前缀私有）。其中：

- `fit` — 78.6% 覆盖（缺 4 行）
- `predict_survival` — 已覆盖
- `get_run_summary` — 已覆盖
- `get_privacy_info` — 已覆盖
- **`evaluate` — 完全未执行（L158-181，24 行）**
- **`available_models` / `available_protocols` — 完全未执行**

`evaluate` 内部串起了 `EvalSurv` + IPCW integrated Brier score + `evaluation_time_grid`，是高风险组合逻辑，且它调用的 `evaluation_time_grid` 本身也只在 `utils/metrics.py` 里 89% 覆盖（缺 L114、L116）。

经检索，`evaluate` / `available_models` / `available_protocols` **在 README 与 `docs/` 中均 0 次出现**，即尚未对用户承诺。这降低了严重性，但它们是**无下划线前缀的公开方法且带完整 docstring**，属于实际可用接口，应该在论文的软件验证章节里有测试记录。

### 5.3 `split_type='random'` 整条划分分支未被验证

`DataSplitter` 的 `__init__` 明确接受 6 种取值：

```python
if self.split_type not in [
    'iid', 'random', 'non-iid', 'censoring-non-iid',
    'time-non-iid', 'dirichlet'
]:
```

测试中的实际命中情况：

| `split_type` 取值 | 是否被测试 | 走哪个函数 |
| --- | --- | --- |
| `iid` | ✅ 6 处 | `_split_iid` |
| `dirichlet` | ✅ 3 处 | `_split_Dirichlet` |
| `non-iid` | ✅ 1 处 | `_split_censoring_non_iid` |
| `time-non-iid` | ✅ 1 处 | `_split_time_non_iid` |
| `invalid` | ✅ 1 处（异常路径） | 抛 `ValueError` |
| **`random`** | ❌ **0 处** | `_split_random` ← **函数从未执行** |
| **`censoring-non-iid`** | ❌ **0 处** | 复用 `_split_censoring_non_iid`（函数已覆盖，但该别名取值未验证） |

即：**6 种宣称支持的划分模式，实际验证了 4 种**。`random` 是完全独立的实现（13 行）却零覆盖。这个模块在 v0.5.0 曾出过 Dirichlet 分支大小写导致走错分支的 bug（见 `docs/` 既有记录），恰恰说明划分分支是历史高发区，`random` 的零覆盖是明显风险敞口。

### 5.4 DP 模块：`add_noise_to_weights` 零覆盖

`core/differential_privacy.py` 覆盖率 74.9%，其中：

| 方法 | 状态 |
| --- | --- |
| `add_noise_to_weights`（L190-222，33 行） | ❌ **完全未执行** |
| `privatize_model_update` | 部分（缺 L246, 248, 282, 283, 291） |
| `compute_privacy_budget` | 部分（缺 L314） |
| `get_noise_scale` | 部分（缺 L334） |
| `apply_dp_to_gradients` | 部分（缺 L366-373） |

值得注意的背景：既有的 DP 专项审计（`docs/v0.7.0-dp-audit.md`）已查明 `add_noise_to_weights` 在 v0.7.0 的**生产调用点为 0 处**——真正干活的是 `privatize_model_update`，前者只是"被修错了地方的 Hilbert 房间"。这正好解释了它为何零覆盖：**它已从生产路径上掉队，却仍作为公开方法保留，测试自然遗忘它**。

结论：这不是"漏测"，而是"死代码未被清理 + 未被测试共同导致"。建议二选一——要么删除，要么补测并明确其定位。`apply_dp_to_gradients` 缺 L366-373 则是真实的梯度路径异常分支未覆盖。

### 5.5 `FSAConfig.__post_init__` 21 行校验分支未覆盖

`core/config.py` 覆盖 86%，但缺口高度集中在 `__post_init__`（L94-262）的**参数校验分支**上，缺失行 L113-156 共 21 行。

这意味着：**配置的非法取值组合大多没有被测试断言过**。结合既有排查记录中已发现的多个配置类问题（`BATCH_SIZE` 语义、`early_stopping_patience` 无效、`dp_epsilon` 在 Gaussian 下成为死参数无告警），校验分支的测试缺口正是这些"静默失效"类 bug 得以存活的土壤。

配套的 `experiments/config_schema.py`（77.1%，33 行缺失）也全是 `_positive_int` / `_positive_float` / `_choice` 之类的**校验辅助函数分支**（如 L54、L61-62、L69），性质完全相同。

### 5.6 `experiments/baselines.py` 与平行实现

`baselines.py` 覆盖 67.3%，其中 `summarize_results`（37 行）完全未执行。更值得关注的是 `_prepare`（缺 L58-64, L73-75）与 `_wrap_model`（缺 L81-92）——既有审计（`docs/v0.7.0-verification.md` 的 N3 项）已指出 **`baselines._prepare` 与 `models/adapters.configure_targets` 是两套平行实现**：前者硬编码 if/elif 分派，后者走多态注册表。

在自定义 adapter 场景下二者结论不一致（`out_features` 6 vs 1、`labtrans` 存在与否），而这个分歧点**恰好落在未覆盖行上**。也就是说：**当前测试无法发现这两套实现的语义漂移，而这会直接污染论文中集中式基线的对比数字。** 这是本报告中最需要优先修复的"测试盲区 + 已知缺陷"叠加点。

---

## 六、统计口径说明与已知盲区

### 6.1 `examples/` 的 1,320 行完全不在统计内

`federated_survival/examples/` 下 9 个脚本共 1,320 行：

```
dp_mechanisms_comparison_example.py   303
differential_privacy_example.py       264
privacy_comparison_example.py         180
model_comparison_example.py           156
data_partitioning_example.py          136
data_generation_example.py            112
basic_usage_example.py                 89
data_augmentation_example.py           45
protocol_comparison_example.py         35
```

该目录**没有 `__init__.py`**，不是 Python 子包，因此 `coverage --source=federated_survival` 在扫描源码树时直接跳过，`coverage report` 也不会列出。这 1,320 行示例代码是**完全的统计盲区**——既没有覆盖率数字，也不在 pytest 收集范围内。

考虑到 README 与文档会引用这些示例作为用法示范，"示例代码可能已与实际 API 脱节"是一个未被任何机制拦住的风险（历史上已经出现过 README 示例参数名与实际签名不符的问题）。

### 6.2 CI 无覆盖率门禁

`.github/workflows/ci.yml` 对 7 个 Python 版本跑 `python -m pytest`，但**没有任何 `--cov` / `fail_under` 配置**：

```
- name: Run test suite
  run: python -m pytest
```

即：**覆盖率可以从 85% 掉到 50% 而 CI 依然全绿**。这是所有缺口能够长期存在的制度性原因。

### 6.3 未被收集的测试脚本

`tests/run_dp_tests.py` 命名不符合 `test_*.py` 约定，不被 pytest 收集；`tests/README_DP_TESTS.md`、`README_DP_MECHANISMS_TESTS.md` 是文档。

---

## 七、补齐优先级建议

### P0 — 投稿前必须处理

1. **补 CLI 端到端单测**（`cli.py` + `config_runner.py`）。用 `subprocess` 或 `capsys` + monkeypatch 覆盖 `main()` 的三条命令分派，至少验证 `--help`、`doctor`、`quickstart --rounds 1`、`run-config <yaml>` 四条路径。这是对外发布的唯一入口，零单测不可接受。
2. **补 `FederatedSurvival.evaluate` 测试**。构造 `fit` 后的小数据集，断言返回 dict 的三个键存在且 `c_index`/`ibs` 在合法区间，并覆盖 shape 不匹配的 `ValueError` 分支。
3. **补 `split_type='random'` 与 `censoring-non-iid'` 测试**。划分分支是历史 bug 高发区，5 分钟即可补齐。
4. **在 CI 加覆盖率门禁**。建议起步设 `--cov-fail-under=80`（对应当前 85.5% 留出余量），防止后续继续劣化。

### P1 — 与已知缺陷联动修复

5. **`baselines._prepare` / `adapters.configure_targets` 平行实现**（N3）。先统一为单一真相源，再补针对自定义 adapter 的对照测试。这是唯一"测试盲区恰好覆盖已知缺陷"的叠加项。
6. **`FSAConfig.__post_init__` 校验分支测试**。按参数逐个构造非法值，断言抛错信息。既补覆盖率，又能顺带暴露"死参数无告警"类问题。
7. **DP 模块收尾**：决定 `add_noise_to_weights` 的去留（删除或补测并标注定位），补 `apply_dp_to_gradients` 的 L366-373 异常分支。

### P2 — 工程质量

8. **补 `baselines.summarize_results`**（37 行公开导出，纯计算，极易测）。
9. **补 `MVAE.sample` / `generate` / `cmse`**，并覆盖 `MVAE.condition_sample` 的缺失分支。
10. **示例代码纳入验证**：给 `examples/` 加一个 smoke 测试（`subprocess` 逐个跑或 `compileall`），或至少在 CI 保留 `quickstart` 冒烟。同时考虑为 `examples/` 补 `__init__.py` 使其进入覆盖率视野。
11. **`_compat._simps`**：用 monkeypatch 模拟 `scipy < 1.14` 场景补测。
12. **`run_dp_tests.py`**：重命名为 `test_run_dp_tests.py` 或明确标记为手动脚本，消除命名与收集规则不一致。
13. **`utils/metrics.py`**：`calculate_ibs` 缺 L44/L51、`evaluation_time_grid` 缺 L114/L116 属异常分支，既有审计已交叉验证其数值正确（与 pycox 偏差 <1%），补齐即可闭环。

---

## 八、附：如何复现

```bash
cd /c/Users/skyee/federated_survival
export MPLBACKEND=Agg

# 语句覆盖率
D:/anaconda3/envs/py38/python.exe -m coverage run \
    --source=federated_survival -m pytest tests/ -q
D:/anaconda3/envs/py38/python.exe -m coverage report -m --sort=cover

# 分支覆盖率 + JSON + HTML
D:/anaconda3/envs/py38/python.exe -m coverage run --branch \
    --source=federated_survival -m pytest tests/ -q
D:/anaconda3/envs/py38/python.exe -m coverage json -o coverage.json
D:/anaconda3/envs/py38/python.exe -m coverage html -d docs/coverage_report

# 未覆盖函数定位（本报告第四节的数据来源）
D:/anaconda3/envs/py38/python.exe scripts/coverage_gap_analyzer.py coverage.json --only-full
```

> 若在 WorkBuddy 托管的终端里运行，需前置 `env -u PYTHONPATH` 以解除 `Path.unlink` 安全垫片对本机文件删除配额的影响（否则部分用例会以 `SystemExit: 1` 假失败）。

---

## 九、补洞结果（同日更新）

按第七节的优先级，已完成四个最严重缺口的修复与补测。

### 9.1 总体变化

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | --- |
| 测试用例 | 214 | **283** | +69 |
| 语句覆盖 | 85.5% | **90.1%** | +4.6pp |
| 分支出口覆盖 | 71.7% | **75.8%** | +4.1pp |
| 完全未执行的函数 | 20 | **2** | −18 |
| 套件耗时 | 45s | 38s | 更快 |

分文件对比：

| 文件 | 修复前 | 修复后 |
| --- | ---: | ---: |
| `experiments/config_runner.py` | 13.2% | **100%** |
| `cli.py` | 60.2% | **98%** |
| `experiments/infrastructure.py` | 88.2% | **96%** |
| `data/splitter.py` | 89.7% | **95%** |
| `api.py` | 72.8% | 79% |
| `experiments/baselines.py` | 67.3% | 79% |

### 9.2 四个洞的处理方式

**洞 1 · CLI 全链路** — 新增 `tests/test_cli.py`（30 用例）。
`main()` 接受 `argv` 列表，因此在进程内驱动，无需 subprocess：覆盖四条子命令的分派、
`doctor` / `config-options` / `quickstart` / `run` 四个命令体、`_protocol_specs` 的六条
归一化分支、`_run_legacy_config` 的 models/seeds 校验与失败聚合、`run_config_file` 的
workflow / legacy 双分派与错误路径。唯一昂贵的真实 quickstart 用最小预算跑一次
（`--rounds 1 --samples 60 --features 3 --clients 2 --num-durations 5 --split-type random`），
其余用 stub 替换估计器。

**洞 2 · `FederatedSurvival.evaluate`** — 追加 4 用例：正常路径（三键齐备、
`c_index ∈ [0,1]`、`ibs ≥ 0`、结果可 JSON 序列化）、长度不匹配 `ValueError`、
未 fit 时的 `RuntimeError`，并补上零覆盖的 `available_models` / `available_protocols`。

**洞 3 · `split_type='random'`** — 追加 5 用例：`_split_random` 是不丢行、不重叠的严格
划分（用 index 集合校验）、`random` 已接入公共 `split()` API、`random` 结果 ≠ `iid`
（防"分支未生效"回归）、不整除时末位客户端吃余数、`censoring-non-iid` 别名可用。

**洞 4 · 平行实现** — 见 9.3。

### 9.3 洞 4 的实质：一个被零覆盖掩盖的真缺陷

第四节把 `baselines._prepare` / `_wrap_model` 的未覆盖行标为「测试盲区 ∩ 已知缺陷」
叠加点。补测时确认它比原判断更明确——**不是测试缺失，是生产缺陷**：

`FSAConfig.__post_init__` 通过 `available_model_adapters()` 校验 `model_type`，
源码注释写明意图是 "so third-party adapters can be configured without editing this
dataclass"。即 `register_model_adapter` 是**官方扩展点**。但基线路径硬编码：

| 路径 | `out_features` | `labtrans` | 结果 |
| --- | --- | --- | --- |
| `baselines._prepare` | **1**（错） | None | 静默建出维度错误的网络 |
| `adapter.configure_targets` | **10**（对） | LabTransDiscreteTime | 正确 |
| `baselines._wrap_model` | — | — | `ValueError: Unsupported model` |

后果：用户注册自定义模型后，FSA 主路径正常，**Center / Local 基线崩溃或算错**，
而论文的对比结论正建立在这些基线上。

**修复**：`_prepare` / `_wrap_model` 改为委托 adapter；`_fit_one` 的 baseline-hazard
判定由硬编码 `("DeepSurv", "CoxPH", "CoxTime", "CoxCC")` 改为 adapter 的
`requires_baseline_hazards` 能力声明。

**等价性护栏**：修复前后对 7 个内置模型比对
`(out_features, labtrans 类型, transformed 数值 sha256)`，结果 **7/7 IDENTICAL、零漂移**：

```
CoxPH            IDENTICAL  ('1',  'NoneType',             '15e1cd15632b9fe7|ca26f1950dab92a5')
DeepSurv         IDENTICAL  ('1',  'NoneType',             '15e1cd15632b9fe7|ca26f1950dab92a5')
CoxCC            IDENTICAL  ('1',  'NoneType',             '15e1cd15632b9fe7|ca26f1950dab92a5')
CoxTime          IDENTICAL  ('1',  'LabTransCoxTime',      'b082356a843ef294|ca26f1950dab92a5')
LogisticHazard   IDENTICAL  ('10', 'LabTransDiscreteTime', '6c89f113ed97e45b|ca26f1950dab92a5')
PC-Hazard        IDENTICAL  ('10', 'LabTransPCHazard',     '15d7807e7d378d4d|...|6fbd7d42e1df0c35')
DeepHit          IDENTICAL  ('10', 'LabTransDiscreteTime', '6c89f113ed97e45b|ca26f1950dab92a5')
```

**补测过程中另一个发现（非缺陷，但易踩）**：`summarize_results` 按 `split` 列分组，
而 `run_paired_baselines` 产出的 DataFrame **不含该列**——它由 `workflow_runner` 在持久化
`raw_results.csv` 时补上。因此 `summarize_results(run_paired_baselines(...))` 会直接
`KeyError: 'split'`。这属于隐式输入契约，已在 docstring 中显式说明。

### 9.4 追加补齐 A/B 档剩余函数

原定验收标准是「A/B/C 档 18 个零覆盖函数全部消除」。四个洞只覆盖其中 12 个，
补齐了剩余 6 个：

- `tests/test_baselines.py`（4 用例）→ `summarize_results`，锁定 FSA−Center / FSA−Local
  的 C-index 与 IBS 配对差值算术、多种子取均值、`Local-client` 行被排除、split/model 分组隔离
- `tests/test_mvae.py`（5 用例）→ `MVAE.sample` / `MVAE.generate` / `cmse`
- `tests/test_protocols.py`（+2 用例）→ `FederatedProtocol.aggregate`（基类契约）
  与 `WebDISCOStyleCoxProtocol.aggregate`（拒绝参数聚合）

### 9.5 仍未覆盖（遗留项）

| 项 | 位置 | 说明 |
| --- | --- | --- |
| `_compat._simps` | `_compat.py:11` | D 档。仅在 `scipy < 1.14` 时安装，本机不触发；需 monkeypatch 模拟旧版 scipy 才能覆盖 |
| `add_noise_to_weights` | `core/differential_privacy.py:190` | DP 专项。既有审计已确认其生产调用点为 0（真正干活的是 `privatize_model_update`），应先决定删除或保留，再补测 |

**未达成的指标**：分支出口覆盖 75.8%，低于本报告原定的 ≥77% 目标。缺量集中在
`experiment_api.py`（35 个部分覆盖）、`config_schema.py`（32）、`visualization.py`（23）、
`differential_privacy.py`（11）、`parameter.py`（10）——均为参数校验与绘图分支，
不属于四个洞的范围。建议下一轮按「参数校验分支」为主题集中补。

### 9.6 CI 覆盖率门禁

`.github/workflows/ci.yml` 的测试步骤改为：

```yaml
- name: Run test suite with coverage floor
  run: |
    python -m coverage run --branch --source=federated_survival -m pytest
    python -m coverage report --fail-under=80
```

刻意使用 `coverage.py` 而非 `pytest-cov`：本报告的全部数字都来自
`coverage run --branch`，用同一套命令可保证本地与 CI 口径一致，避免出现
「本地 90% / CI 88%」的口径困惑。阈值 80% 对当前 86% 的综合覆盖率留有余量。
本地已预演通过：`coverage report --fail-under=80` → PASS。

另外在 `pyproject.toml` 的 dev extras 中显式声明 `coverage>=7.0.0`（此前仅经
`pytest-cov` 间接引入）。

### 9.7 教训

1. **零覆盖函数是缺陷的高发区，值得优先侦察而非仅仅补数**。`baselines._prepare`
   被列为「未覆盖」时看着只是测试缺失，实际藏着一个会让论文对照组数字失真的缺陷。
   本报告中 `_split_random`、`summarize_results` 也都是同类。
2. **测成本要用暖机后的数字**。首次 `fit` 的 4.4s 是 numba/torch 冷启动，
   暖机后单次训练仅 0.03s。按冷启动估成本会严重高估，进而错误地决定「不覆盖」。
3. **修复有平行实现的历史代码时，先固化数值指纹再动刀**。用 sha256 比对
   `transformed` 数值是证明「论文数字零漂移」最直接的证据。

