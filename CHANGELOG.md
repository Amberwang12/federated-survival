# Changelog

本文件记录项目的所有重要变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 新增 `docs/test-coverage-audit-2026-09-27.md`：全量测试覆盖率审计报告。
  基线 214 用例全绿，语句覆盖 85.5% / 分支出口覆盖 71.7%；定位 **20 个从未被
  执行的函数**（CLI 全链路、`FederatedSurvival.evaluate`、`split_type='random'`
  划分分支、`baselines.summarize_results`、WebDISCO `aggregate` 等），并指出
  `examples/` 1320 行为统计盲区、CI 缺覆盖率门禁。含 A/B/C/D 四档风险分级与
  P0/P1/P2 补齐建议。
- 新增 `scripts/coverage_gap_analyzer.py`：覆盖率缺口分析工具。读取 `coverage json`
  输出，用 AST 把未覆盖行映射到函数，区分「完全未执行」与「部分覆盖（分支缺失）」，
  解决 `coverage report` 只能给行号、无法直接看出哪个函数零覆盖的问题。
- 新增 `tests/test_cli.py`（30 用例）：覆盖 `federated-survival` 控制台入口全链路——
  `main` 子命令分派、`doctor` / `quickstart` / `run` / `config-options` 四个命令实现、
  `_protocol_specs` 六条归一化分支、`_run_legacy_config` 的校验与失败聚合、
  `run_config_file` 的 workflow / legacy 双分派。`cli.py` 覆盖率 60% → 98%，
  `config_runner.py` 13% → 100%。
- 新增 `tests/test_baseline_adapter_parity.py`（19 用例）：锁定 `experiments.baselines`
  与 adapter 注册表的行为一致性。7 个内置模型的 out_features / labtrans 类型 /
  transformed 数值必须与 `ModelAdapter.configure_targets` 完全一致；另用自定义
  adapter 证明 `register_model_adapter` 扩展点在基线路径上同样可用。
- 新增 `tests/test_baselines.py`（4 用例）：锁定 `summarize_results` 的配对差值算术
  （FSA−Center / FSA−Local 的 C-index 与 IBS）。
- 新增 `tests/test_mvae.py`（5 用例）：覆盖 `MVAE.sample` / `MVAE.generate` 与条件
  均方误差 `cmse` 的掩码语义。
- 新增 `docs/dp-accounting-audit.md`：差分隐私记账口径调研与 BUG-03 定案报告。
  含 Opacus / TensorFlow Federated / Flower / Geyer(DP-FedAvg) 四方开源实证、
  本地复刻 RDP accountant 的定量审计、两个修复方案对比、4 个待拍板决策点。
- 新增 `scripts/dp_privacy_audit.py`：DP 隐私记账独立交叉验证工具。
  复刻 Opacus RDP accountant（仅依赖标准库），可反算实际加噪路径的真实 ε、
  反解达标所需 σ、评估客户端子采样的隐私放大收益。建议纳入 CI 校验
  `get_privacy_info()` 报告值的正确性。
- 新增 `docs/v0.7.0-verification.md`：v0.7.0 版本验收报告。逐条核对 9-14 报告中
  17 项问题的修复状态（11 项已修 / 3 项改进 / 2 项未修 / 1 项不适用），
  并记录 v0.7.0 引入的 2 项新 DP 语义问题。**验收结论：通过，建议发布**。

### Changed

- **修复 `experiments/baselines.py` 与 adapter 注册表的双实现分歧（P1）**
  - 问题：`baselines._prepare` / `_wrap_model` 用硬编码 `if/elif` 链按 `model_type`
    分派 label transform 与 pycox 模型包装，而 `core.runner` 走 adapter 注册表。
    7 个内置模型下两者一致，但**在官方支持的 `register_model_adapter` 扩展点上会
    静默分歧**：实测自定义 adapter 下 `_prepare` 给出 `out_features = 1`（应为 10）、
    `_wrap_model` 直接抛 `ValueError: Unsupported model`。后果是 Center / Local
    基线算错或崩溃，而论文对比结论依赖这些基线。
  - 修复：两处改为委托 `get_model_adapter(...).configure_targets()` 与
    `build_model()`；`_fit_one` 的 baseline-hazard 判定也由硬编码模型列表改为
    adapter 的 `requires_baseline_hazards` 能力声明。净删除约 20 行硬编码分支。
  - **等价性已实证**：对 7 个内置模型比对修复前后的
    `(out_features, labtrans 类型, transformed 数值 sha256)`，**7/7 完全一致、零漂移**，
    论文已有数字不受影响。
  - 附带：`summarize_results` 补充 docstring，说明其输入需含 `split` 列
    （`run_paired_baselines` 本身不产出该列，由 `workflow_runner` 持久化时补上），
    避免使用者把两个公开导出直接串联时踩 `KeyError`。
- `.github/workflows/ci.yml` 增加覆盖率门禁：`coverage run --branch` +
  `coverage report --fail-under=80`。此前 CI 只跑 `pytest`，覆盖率从 85% 掉到 50%
  也不会失败——这是上述缺口得以长期存在的制度性原因。刻意沿用项目既有的
  `coverage.py` 命令而非 `pytest-cov`，以保证本地与 CI 的统计口径一致。
- `pyproject.toml`：dev extras 显式声明 `coverage>=7.0.0`（此前仅经 `pytest-cov`
  间接引入）。
- `.gitignore` 补充 `coverage.json`、`.coverage.*`，避免覆盖率中间产物入库。
- 更新 `bug-report-2026-09-14.md` 中 BUG-03 条目，更正此前的错误定性：
  - **撤销** 3c「`total_epsilon` 恒等于 ε 是数学错误」—— ε 作为全程总预算时，
    恒等于配置值是**正确**的；原判断有误。
  - **更正** 3b：`per_round = ε/√T` 在 RDP 下自洽（实测总 ε 恒为 0.9688，与轮数无关），
    **测试期望的 `ε/T` 才是错的**（过度保守 4~5 倍，T=100 时 σ=216 必然压垮模型）。

- **更正 BUG-06 / BUG-15 于 v0.7.0 的定性（二次复核，降级为「非运行时缺陷」）**
  - 作用范围：`federated-survival-v0.7.0-software-paper/`（验收结论修正，无代码改动）
  - 背景：初版 `docs/v0.7.0-verification.md` 将两项定为「未修复」，等级偏高。
  - **BUG-06**（`Server(FSAConfig())` 抛 `AttributeError: no attribute 'out_features'`）：
    现象**仍可复现**，源头为 `models/adapters.py:141`。但 `out_features` 的值必须等
    label transform `fit` 之后才能确定，原理上不可能是静态字段；`runner.py:121→127`
    与 `experiments/baselines.py:120→122` 两条生产路径均已正确执行两阶段初始化。
    且 `Server` **未出现在任何一层 `__init__.py` / `__all__`**，不是公开 API；
    v0.7.0 主推的 `register_model_adapter` / `register_federated_protocol` 两个扩展点
    均不需要调用方构造 `Server`；全仓 `tests/` **零处**构造 `Server`。
    → 降级为**低优先级 DX 瑕疵**，仅「报错信息未提示解法」值得修。
  - **BUG-15**（`splitter.py:111` 循环内定义 `feature_cols`、`:117/:121` 循环外使用）：
    源码形态**确实存在**，但**不可达** —— 全部 5 个 `_split_*` 均以
    `for i in range(self.n_clients)` 构建键，且 `__init__:42` 守卫 `n_clients > 0`
    → 循环至少执行一次。实测 200 组退化组合（n_samples × n_clients × 5 种划分 × alpha）
    **0 次触发 `NameError`**，130 组被既有 `ValueError` 守卫正常拦截。
    → 降级为**不可达 code smell**。
  - 同时修正初版一处硬错误：初版称「grep 全局无 `out_features`」有误，实际存在
    `adapters.py:123,127,141` 等 5 处，系 grep 路径写法（带尾斜杠 + `--include`）静默失效所致。
  - 新增 `docs/bug-06-15-recheck.md` 记录完整复核证据与修复建议。

- 更新 `docs/v0.7.0-verification.md`：新增 **N3 / N4 两项发现**（本节为增量补充）。
  - **N3（P1，新缺陷）**：`experiments/baselines.py::_prepare()` 与
    `models/adapters.py::configure_targets()` 是两套平行实现，`_prepare` 用硬编码
    if/elif 列举 model_type，与 adapters 的「注册表 + 多态」构成两个真相源。
    实测注册自定义 adapter 后：`configure_targets` 得 `out_features=6` + `labtrans`，
    而 `_prepare` 得 `out_features=1` + `labtrans=None`（连属性都没有），**无报错无警告**。
    现有 7 个内置模型两处恰好吻合故无症状，但 `register_model_adapter()` 是 v0.7.0
    官方推广的扩展点，一旦被使用即命中。因 baselines 是集中式基线（用于论文横向对比），
    基线算错会导致实验结论不可信。建议删除硬编码分支、统一走 adapters，并补一致性守护测试。
  - **N4（P2，测试盲区）**：首次用 `coverage` 量化 v0.7.0 —— 总体 **84%**（3578 stmts / 565 missed）。
    最低：`experiments/config_runner.py` 18%、**`utils/metrics.py` 25%**（缺失 37-90 行
    即 `calculate_ibs` 全部实现体）、`_compat.py` 43%、`cli.py` 61%、
    `experiments/baselines.py` 75%、`api.py` 77%、`core/differential_privacy.py` 81%。
    **`grep "calculate_ibs" tests/` = 0 命中**，`calculate_cindex` 与 `evaluation_time_grid`
    同样 0 命中 —— 而它们是 `__init__.py` 里唯二对外导出的评估工具。
    **已交叉验证实现正确**：以 pycox `EvalSurv.integrated_brier_score` 为参考，
    3 组样本（n=60/150/300）相对偏差 0.95% / 0.59% / 0.29%；6 类异常路径均正确抛 `ValueError`。
    → 非 bug，但需把交叉验证固化为测试防回归。
  - 覆盖率工具：`coverage 7.6.1` 已装入 `py38` 环境（`pip install coverage`，清华源）。

### Fixed

- 无代码改动。本次为调研与文档修正阶段，代码修复待口径拍板后进行。

- **差分隐私模块专项审计（v0.7.0）**：新增 `docs/v0.7.0-dp-audit.md`，
  对 `core/differential_privacy.py`（382 行）逐条核验原始报告的 5 条 DP bug。无代码改动。
  - **结论：能修的都修了且修得对，但隐私记账核心缺陷（BUG-03d）原封不动，另新增 3 项语义问题。**
  - ✅ BUG-01 机制分派：实测峰度 gaussian **2.993** / laplace **6.028**，exponential 显式拒绝。
    但注意其修复位置 `add_noise_to_weights` 在 v0.7.0 中**生产调用 0 处**，
    干活的是新增的 `privatize_model_update`（client.py:115），因照抄同款分派逻辑才一并受益。
  - ✅ BUG-03a/03b：σ 统一为 `noise_multiplier × sensitivity`（Opacus 口径），三方冲突根除，
    6 个历史失败用例消失。`compute_privacy_budget` 降级为纯名义汇报并加了诚实注释。
  - ✅ BUG-05 batch_norm+DP：非浮点守卫生效，实测 int64 `num_batches_tracked` 不再崩溃。
  - ✅ BUG-14 flaky：随口径统一消失。
  - ❌ **BUG-03d 未修**：生产路径 σ 完全不依赖轮数 T；σ=1.0 时 T=1/10/50/100 真实
    ε = 5.11 / 19.66 / 60.82 / **110.82**（低报最高 111 倍，v0.5.0 时是 32 倍，因默认 σ 变小而放大）。
  - 🔶 **BUG-02 半修 → N1 对称化**（比上一版定性更严重）：Gaussian 下 `dp_epsilon` 死参数、
    Laplace 下 `dp_noise_multiplier` 死参数，**两个机制各有各的死参数且恰是对方的控制旋钮**，
    切换 `dp_mechanism` 时同一参数会从有效骤变无效，**无告警**。
    根因：两机制噪声公式输入不重叠（`:52` σ=nm×sens vs `:81` b=sens/ε）。
  - 🆕 **N5**：`get_privacy_info()` 的 `noise_scale` 含 `/√K` 而生产路径不除 ——
    5 客户端下报告 0.4472 vs 实际 0.9991，**低报 √K 倍**；且该字段仅 gaussian 才有。
  - 🆕 RDP 半成品：`compute_renyi_divergence` / `convert_renyi_to_epsilon` **生产调用 0 处**（仅测试引用），
    零件已写好但未接入任何 accounting 路径。
  - 肯定 v0.7.0 的真实进步：`privatize_model_update()` 改为裁剪 client update 增量 Δw 再加噪，
    贴近 Geyer DP-FedAvg；docstring 诚实声明「非 record-level DP-SGD、不提供组合端到端保证」。
  - 复现手法备注：**用同形状零张量同时作 global 与 local**（Δw=0 → 输出即纯噪声）可直接测得 σ；
    勿用单元素张量，`torch.std` 在 n=1 时返回 `nan`（本次首版审计脚本即踩此坑）。

- **修复 0.7.0 软件论文版 `DataLoader` 静默丢特征（P2，本次已完成）**
  - 作用范围：`federated-survival-v0.7.0-software-paper/.../federated_survival/data/loader.py`
    （0.5.0 工作区版**未改动**，仍保留原问题，见 Security 段说明）
  - 根因：`_process_data()` 用 `sorted([col for col in data.columns if col.startswith('x')])`
    识别特征列，与文档承诺的「`feature_columns` 可映射到任意目标列名」冲突。
    映射到非 `x` 前缀列名时，特征被静默丢弃，返回仅含 time/status、无报错无警告的 DataFrame。
  - 变更：特征列改为「排除 time/status 后按原始顺序保留」；新增 0 特征列检测与非数值列报错。
  - 连带修复：移除 `sorted()` 字典序，≥10 个特征时列序不再错乱为 `x1,x10,x11,x2,...`。
  - 验证：`tests/test_loader.py` 新增 4 条回归测试（共 11 条）。
    全套件 **195 用例全部通过**（`195 passed, 9 warnings in 39.20s`）。

### Security

- **新发现（未修复）**：BUG-03d —— 隐私预算计算与实际加噪路径脱节，
  声称 ε=1 时，K=5 / T=100 的真实 ε ≈ **32.12**，隐私保证低报约 32 倍。
  详见 `docs/dp-accounting-audit.md` 第 3.2 节。
- **新发现（未修复）**：BUG-03e —— `add_noise_to_weights` 中 `sensitivity/√K` 缩放
  无隐私记账依据（Flower 官方口径为 Central DP 线性除 m、Local DP 不除 K），
  等价于把 ε 悄悄放大 √K 倍。
- **已知未修复（仅限 0.5.0 工作区版）**：`DataLoader` 的 `startswith('x')` 丢特征问题。
  按用户要求本次只修 0.7.0 软件论文版；0.5.0 版 `federated_survival/data/loader.py` 保持原样，
  若后续要同步，改动内容见上一条 Fixed 条目。
- **环境问题（已解决）**：0.7.0 仓库的 `tests/test_config_workflow.py`（8 条）与
  `tests/test_experiment_infrastructure.py`（9 条）依赖 `pyyaml`，本机 `py38` 环境原先缺失，
  导致这两个模块无法收集。现已安装 `pyyaml 6.0.3`，全套件 195 条可完整收集并全部通过。
- **v0.7.0 新发现 N1（未修复，发布前需在 README 声明）**：Gaussian 机制下 `dp_epsilon` 沦为
  **死参数**。σ 改为 `noise_multiplier × sensitivity` 后，`dp_epsilon` 从 1.0 调到 10.0
  实测噪声 std 恒为 ~0.999，无任何告警；Laplace 下则正常生效。两个机制参数语义不一致，极易误用。
- **v0.7.0 新发现 N2（部分缓解）**：`get_privacy_info()` 仍返回名义的 `total_epsilon`（= 配置值），
  与实际噪声无关。默认 `noise_multiplier=1.0`、σ=1.0、q=1 时，T=100 的真实 ε ≈ **110.82**
  （低报约 111 倍，v0.5.0 时为 32 倍）。
  缓解：作者已新增 `formal_accounting_available: False` 与 `privacy_scope` 字段显式声明无正式记账，
  属**已知限制的诚实标注**而非隐瞒；但 `total_epsilon` 字段本身仍具误导性，建议改名为 `nominal_epsilon`。
  定量方法与改善方案见 `docs/v0.7.0-verification.md` 第三节。

---

## [0.5.0]

历史版本变更未追溯记录，自本次起开始维护本文件。
