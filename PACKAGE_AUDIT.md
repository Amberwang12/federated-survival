# Python 软件包结构审计报告

**审计对象**：`federated-survival` **v0.8.0**（tag `v0.8.0`；本次审计的代码状态 = `v0.7.5` tag `86f4627` + 审稿项 #16 的 DP 死参数修复，GitHub `Amberwang12/federated-survival`）
**审计日期**：2026-10-07（首次审计 2026-09-27，对象为 0.7.0；0.8.0 沿用本次结论并同步版本号）
**仓库体积**：22 MB（其中 `results/` 5.6 MB、`docs/` 3.1 MB、`federated_survival/` 1.7 MB、`tests/` 812 KB、`examples/` 816 KB、`build/` 570 KB、`dist/` 432 KB）
**跟踪文件数**：147（`git ls-files`），**工作区干净**（`git status --porcelain` 为空）
**结论速览（对比 0.7.0）**：

| 维度 | 0.7.0 结论 | **0.8.0 结论** |
|---|---|---|
| 软件包本体 | 完整健康 | ✅ **完整健康**（61 个文件，wheel 64 条目） |
| 版本控制 | ❌ **全仓无 `.git`**（最大缺口） | ✅ **已补齐**，3 个 tag、147 跟踪文件、CI + Release 全自动 |
| 变更日志 | ❌ 缺 `CHANGELOG.md` | ✅ **已补**，Keep a Changelog 格式 |
| 构建产物 | ⚠️ `MANIFEST.in` 有死规则 + 告警 | ⚠️ **死规则仍在**（`requirements-paper-figure2-lock.txt`），无构建失败 |
| 打包一致性 | 4 项不一致 | ✅ **1 项**（wheel/sdist 不一致已撤销，license 已修）→ 现仅剩死规则 |
| 学术合规元数据 | ❌ 缺 `CITATION.cff` 等 4 项 | ❌ **仍缺** `CITATION.cff` / `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` |
| 仓库卫生 | 垃圾文件、生成物、`results/` | ⚠️ **仍有**：`build/`、`.idea/`、8 个根级 `*.log`、2 张散落 PNG、3 个临时 `.log` |
| 发行物纯净度 | 干净 | ✅ **干净**（sdist 121 文件，污染扫描 0 命中） |

---

## 一、必须保留 ✅

### A. 软件包本体（发行物核心）

| 路径 | 说明 |
|---|---|
| `federated_survival/__init__.py` | 公共导出 + `__version__ = "0.8.0"`（与 `pyproject.toml` 一致 ✅） |
| `federated_survival/api.py` | 高层 `FederatedSurvival` 估计器（158 行，79% 覆盖） |
| `federated_survival/experiment_api.py` | Center/Federated/Local 对比、模型-协议矩阵、可选结果绘图（227 行，79%） |
| `federated_survival/cli.py` | `federated-survival` 命令入口：`doctor` / `quickstart` / `run`（111 行，**98%**） |
| `federated_survival/_compat.py` | 兼容层（`scipy.integrate.simps` → `simpson`）；7 行，覆盖率 44%（scipy 1.10 环境相关） |
| `federated_survival/core/` | `config`(187) / `client`(57) / `server`(35) / `runner`(246) / `augmenter`(106) / `mvae`(120) / `differential_privacy`(137) / `_losses`(15) / `_minibatch`(58) |
| `federated_survival/models/adapters.py` | **7 个**生存模型适配器 + 注册表 + 生存预测校验（165 行，**93%**） |
| `federated_survival/protocols/` | `base`(70) / `parameter`（FedAvg, FedProx, FedOpt）(104) / `webdisco`(182) |
| `federated_survival/data/` | `generator`(124) / `loader`(47) / `splitter`(166) / `preprocessing`(31) / `workflow`(115) / `augmentation_workflow`(125) |
| `federated_survival/data/real/` | **【0.7.5 新增打包】**`gbsg.csv` + **`colon_death.csv`** — 0.7.0 时该目录是"孤儿文件"，现已通过 `package-data` 正确进 wheel/sdist |
| `federated_survival/experiments/` | `config_runner`(76) / `config_schema`(211) / `infrastructure`(105) / `baselines`(91) / `visualization`(252) / `workflow_runner`(176) |
| `federated_survival/utils/metrics.py` | C-index、IBS（66 行，89%） |
| 各子包 `__init__.py` | 共 **8 个**，齐全；`core`/`data`/`experiments`/`models`/`protocols`/`utils` + 顶层 |

**子目录 `federated_survival/examples/`（1320 行）** 无 `__init__.py`，**非子包**——
`coverage --source` 扫描直接跳过、`report` 也不列，因此覆盖率报告里看不到它。
这不是缺陷，但**读覆盖率时别以为"报告里的文件 = 全部源码"**。

### B. 打包与元数据

| 路径 | 说明 |
|---|---|
| `pyproject.toml` | 唯一权威构建配置（PEP 621），`version = "0.8.0"` |
| `setup.py` | 空壳 shim（3 行有效代码），`setup()` 转调；**建议保留**（兼容老工具链） |
| `MANIFEST.in` | sdist 清单，25 行 |
| `LICENSE` | MIT |
| `README.md` | 项目首页，被 `readme =` 引用；示例已改用 `load_real_data` |
| `PACKAGE_STRUCTURE.md` | 结构说明（被 MANIFEST `include`，且进入 sdist 顶层 ✅） |
| `CHANGELOG.md` | Keep a Changelog 格式，`0.8.0` / `0.7.5` / `0.5.0` 三节 |
| `.gitignore` | 105 行，**分层清晰**：含"本地保护规则（禁提交）"段，明确排除 `.pypirc` / `.workbuddy/` / `.archify/` / `audit/` / `bug-report-*.md` |

### C. 测试、文档、CI、可复现资产

| 路径 | 说明 |
|---|---|
| `tests/` | **30 个 `test_*.py`** + `run_dp_tests.py` + 2 份 README = 33 个文件，**318 用例全绿** |
| `docs/` | **12 篇 `.md`**（11 篇指南 + `test-coverage-audit-2026-09-27.md`）+ `mkdocs.yml`；另有 `docs/coverage_report/`（coverage.py 自动生成，自带 `.gitignore` 内容 `*`，天然不入库） |
| `.github/workflows/ci.yml` | **7 版本 matrix（3.8–3.14）** + quickstart job；覆盖率门禁 `--fail-under=80` |
| `.github/workflows/docs.yml` | MkDocs 严格模式构建 + GitHub Pages 部署（含 artifact 重跑修复） |
| `.github/workflows/release.yml` | **【0.7.5 新增】**tag 触发 → 构建 → `twine check --strict` → **tag/版本一致性断言** → SHA256SUMS → GitHub Release |
| `.github/releases/v0.8.0.md`、`v0.7.0.md` | 手工撰写的发布说明。`release.yml` 优先取 `.github/releases/<tag>.md`，缺失才回退到 GitHub 自动生成 notes —— **v0.7.5 曾因缺该文件走了回退路径，v0.8.0 起已补齐** |
| `federated_survival/examples/` | 9 个可运行示例 + 10 个 smoke/validation 配置（`.yaml`/`.json`/`.md`） |
| `experiments/`（根，4 个 `.py`） | `audit_reviewer_checks.py` / `audit_dp_effect.py` / `make_software_validation_tables.py` / `run_utility_smoke.py`；MANIFEST 已纳入 sdist ✅ |
| `examples/`（根，11 个编号脚本 + `compare/`） | 论文风格示例；`compare/` 下 `Models.py` / `central.py` / `client_local.py` / `compare_v2.py` + `result/` |
| `scripts/`（4 个 `.py`） | **【0.7.5 新增 3 个】**`coverage_gap_analyzer.py` / `run_all_examples.py` / `run_root_examples.py` / `dp_privacy_audit.py`（后者 gitignore，本地专用） |
| `data/real/gbsg.csv`、`data/real/colon_death.csv` | 仓库根副本，供文档/实验引用；**发行物中按设计排除**（`MANIFEST.in:15 recursive-exclude data *.csv`）。包内副本走 `package-data` 正常分发 |

---

## 二、建议删除 🗑️

> **说明**：以下删除均在"打包/仓库卫生"层面。`results/` 若为论文交付物请先归档再删。
> **0.7.5 现状**：macOS 垃圾（`.DS_Store`）与 `__pycache__` 已清理干净 ✅；
> 0.7.0 清单里的 `software_validation/` 目录已整体删除 ✅。
> 以下是本轮**实测仍存在**的项。

| 类别 | 路径 | 状态 / 理由 |
|---|---|---|
| 构建产物 | `build/`（570 KB，46 个 `.py` + `2 whl` 副本） | ⚠️ **仍存在**。`.gitignore:6` 已忽略 `build/`，但仍占体积；`python -m build` 可再生成 |
| 构建产物 | `federated_survival.egg-info/`（6 文件） | ⚠️ **仍存在**。已 gitignore。`SOURCES.txt` 现 118 行，随 0.7.5 已更新 |
| IDE | `.idea/` | ⚠️ **仍存在**。已 gitignore，个人设置 |
| 安装/调试日志 | `ex06_final.log`、`ex06_tuned.log`、`ex06_tuned2.log`、`ex07_final.log`、`ex08_final.log`、`probe_ex06.log`（**6 个**） | ⚠️ **仍存在**。`.gitignore:55 *.log` 已忽略，但堆在仓库根很碍事 |
| 散落生成图 | `dp_mechanisms_noise_distributions.png`、`dp_mechanisms_privacy_utility_tradeoff.png` | ⚠️ **仍存在且已被 git 跟踪**（`git ls-files` 命中）。应移入 `results/` 或 `docs/assets/` |
| 示例生成图（根 `examples/`） | `03_partition_comparison.png`、`04_partition_comparison.png`、`10_reproduce_paper.png`、`11_avoid_collapse.png` | ⚠️ **已被 git 跟踪**。论文插图候选，若确定要用应移入 `docs/assets/` |
| 生成结果 | `results/`（8 个子目录，20 PNG + 5 CSV + 5 JSON，5.6 MB） | `.gitignore:54 results/` 已忽略；占全仓 1/4 体积 |
| 非软件包内容 | `JCB-2026-0108_audit_checklist.md`、`PACKAGE_AUDIT.md` | ⚠️ **已被 git 跟踪**。审稿/审计过程文档，**已被 `MANIFEST.in` 排除在 sdist 外** ✅（污染扫描 0 命中） |
| 本地工具产物 | `.workbuddy/`、`.archify/`、`federated_survival-architecture.html`、`.coverage` | ✅ 全部已 gitignore，**未入库** |
| ~~孤儿文件~~ | ~~`federated_survival/data/real/gbsg.csv`~~ | ✅ **0.7.5 已修复**：不再是孤儿。`pyproject.toml` `package-data` 新增 `data/real/*.csv`，实测 **wheel 与 sdist 均含这 2 个 CSV** |
| ~~`software_validation/`~~ | — | ✅ 已整体删除，`MANIFEST.in` 对应行亦已删除 |

---

## 三、缺失的文件 ❌

### 1. 版本控制与合规

| 缺失项 | 0.7.0 状态 | **0.7.5 状态** |
|---|---|---|
| ~~`.git` / git 仓库~~ | ❌ 全仓无版本控制 | ✅ **已补齐**。147 跟踪文件；tag `v0.2.0` / `v0.7.0` / `v0.7.5`；工作区干净；`ci.yml`/`docs.yml`/`release.yml` 三工作流 |
| ~~`CHANGELOG.md`~~ | ❌ 缺失 | ✅ **已补**，Keep a Changelog 格式 |
| `CITATION.cff` | ❌ | ❌ **仍缺**。学术软件刚需（GitHub "Cite this repository"）；软件论文尤其必要。**优先级最高的剩余项** |
| `CONTRIBUTING.md` | ❌ | ❌ **仍缺**。CI 已跑 flake8/black/mypy（dev extras）但无成文规范 |
| `CODE_OF_CONDUCT.md` | ❌ | ❌ 仍缺 |
| `SECURITY.md` | ❌ | ❌ 仍缺 |

> **关于"CI 跑 flake8/mypy"的前提修正**：0.7.0 报告称"CI 已跑 flake8/mypy/black"，
> 但**实测 `ci.yml` 并不调用这三个工具**——dev extras 里装了，workflow 只用
> `coverage` + `pytest` + `federated-survival doctor` + `build` + `twine check`。
> 即 **lint/format 在 CI 中完全没有把关**。这与第三节的"无 lint 配置"是同一个缺口。

### 2. 工程规范

| 缺失项 | 说明 |
|---|---|
| `.pre-commit-config.yaml` | `dev` extras 已装 `black`/`flake8`/`mypy`，但无 pre-commit 钩子 |
| **`.flake8` / `ruff.toml`** | 🔴 **0.7.5 实测确认：项目无任何 lint 配置**。这直接导致 `BUG-15`（`splitter.py` 的不可达泄漏变量，`F821` 类问题）只能靠人工发现。建议补 ruff 并开启 `F821` |
| `.gitattributes` | 统一换行符（Win/Linux CI 均有）、标记二进制 |
| `.editorconfig` | 统一缩进/编码 |
| `tests/conftest.py` | **无共享 fixture**；CI 的 `NUMBA_*`/`OMP_*`/`MPLBACKEND` 环境变量目前散落在 workflow（`ci.yml` 与 `docs.yml` 重复定义） |
| `.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/` | 协作模板 |
| `docs/assets/` | 文档站图片无归属目录（现根目录 + `examples/` 的 PNG 无处安放） |
| `federated_survival/py.typed` | 若有类型标注意图（`mypy` 在 dev extras，但项目实际未用） |

### 3. 打包一致性

| 缺失项 | 0.7.0 状态 | **0.7.5 状态** |
|---|---|---|
| `requirements-paper-figure2-lock.txt` | `MANIFEST.in:14` 有 `exclude` 但文件不存在 → 死规则 + 构建告警 | ❌ **死规则仍在**（`MANIFEST.in:14`）。本轮 `python -m build` 仍产生 `no previously-included files found matching 'requirements-paper-figure2-lock.txt'` 告警（无害，但应清） |
| `pyproject.toml` 的 `license-files` | PEP 639 下建议显式声明 | ❌ 仍缺（`Metadata-Version: 2.1`，尚未到 2.4） |
| ~~`"License :: OSI Approved :: MIT License"` classifier~~ | ❌ 缺 | ✅ **已补**，wheel METADATA 实测含该 Classifier |
| `federated_survival/py.typed` | ❌ | ❌ 仍缺 |

---

## 四、打包配置本身的不一致（无需增删文件，需改配置）

1. ~~**wheel 与 sdist 内容不一致**~~ → ✅ **已撤销，结论：wheel/sdist 内容一致**。
   实测 0.7.5 wheel 64 个条目中 `examples/` 下 **19 个文件全部打入**（9 `.py` + 3 `.md` + 4 `.yaml` + 3 `.json`），
   与 `MANIFEST.in:9` 收录一致。原因是 setuptools 在**使用 pyproject.toml 配置时
   `include_package_data` 默认为 True**，`package-data` 并非唯一闸门。
2. **`package-data` 里的 `*.txt`**：**0.7.5 仍为空规则** —— 实测包内**无任何 `.txt` 文件**。
   （对比：`*.md` 有效，因为 `examples/` 下有 3 个 `.md`。）无害，可清理。
3. **【0.7.5 已消解】`namespaces = true` 与"该目录不入发行物"的冲突**：
   0.7.0 时 `namespaces = true` 会把无 `__init__.py` 的 `federated_survival/data/real/`
   视作隐式命名空间包，与"不入发行物"意图冲突。0.7.5 **主动改为纳入发行物**
   （`package-data` 加 `data/real/*.csv`），冲突自然消失。**但 `namespaces = true`
   本身仍是一个宽松设置**，会静默把未来任何缺 `__init__.py` 的子目录纳入包，建议后续评估。
4. ~~**`[project] license` 写法与 py3.8 不兼容**~~ → ✅ **已修复**。
   原 `license = "MIT"` 是 PEP 639 新写法（要求 setuptools ≥ 77），py38 最高只能装 75.x，
   会导致 py38 构建失败与 CI py3.8 job 挂掉。现为 `license = { text = "MIT" }`，
   wheel METADATA 实测 `License: MIT` + `License-File: LICENSE` + License Classifier。
   **0.7.5 的 7 版本 CI matrix（含 3.8）验证了此修复有效。**
5. **`addopts` 含 `--strict-markers`**：需确认全部测试只用已注册 marker。
   0.7.5 实测 318 用例通过且 `[tool.pytest.ini_options]` 未声明 `markers`，
   说明**当前无自定义 marker 使用**——一旦有人加 `@pytest.mark.slow` 就会 CI 失败。
6. **【0.7.5 新增观察】`.github/releases/` 只到 `v0.7.0.md`**：
   `release.yml` 在 tag 推送时查找 `.github/releases/<tag>.md`，找不到则回退到
   GitHub 自动生成 notes。功能正常，但 **0.7.5 的发布说明是自动生成的**，
   内容质量低于 0.7.0 的人工版本。建议补 `.github/releases/v0.7.5.md`。

---

## 五、建议的清理命令（**未执行，待确认**）

> ⚠️ 已在 Git 仓库内，`git rm --cached` / `git clean` 比裸 `rm` 更安全。
> 0.7.0 清单里的旧路径（`federated-survival-v0.7.0-software-paper/`、
> `install_py12.log`、`software_validation/`、`federated_survival/data/real/`）**已失效**，
> 因为仓库已 `git init` 并晋升为正式源码树。

```bash
cd /c/Users/skyee/federated_survival

# 0) 先备份（建议）
tar -czf ../federated-survival-0.7.5-backup-$(date +%Y%m%d).tar.gz \
    --exclude=.git --exclude=build --exclude=dist .

# 1) 构建产物 / IDE（可由 python -m build 再生成，.gitignore 已覆盖）
rm -rf build .idea federated_survival.egg-info

# 2) 根级调试日志（6 个，均已被 .gitignore 忽略）
rm -f ex06_final.log ex06_tuned.log ex06_tuned2.log \
      ex07_final.log ex08_final.log probe_ex06.log

# 3) 散落生成图 —— 注意这 4 个 PNG 已被 git 跟踪，需 git rm --cached
git rm --cached dp_mechanisms_noise_distributions.png \
              dp_mechanisms_privacy_utility_tradeoff.png
mkdir -p results/figures
mv dp_mechanisms_*.png results/figures/

# 4) 生成结果（如已归档）
# rm -rf results
```

---

## 六、执行顺序建议（0.7.5 更新）

1. ✅ 备份 → 2. ✅ 清理 `build/`、`.idea/`、根级 `*.log`（无风险，且已被 gitignore）
3. **补 `CITATION.cff`**（软件论文刚需，优先级最高的缺失项）
4. **补 lint 配置**（`ruff.toml` 开 `F821`）并接入 CI —— 这是唯一能自动拦住 `BUG-15` 类问题的手段
5. 补 `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` / `.gitattributes` / `.editorconfig`
6. 建 `docs/assets/`，把 4 张已跟踪 PNG 迁进去
7. 清 `MANIFEST.in:14` 死规则 + 删 `package-data` 的空规则 `*.txt`
8. 建 `tests/conftest.py`，把 `NUMBA_*`/`OMP_*`/`MPLBACKEND` 从 workflow 收敛到一处
9. 补 `.github/releases/v0.7.5.md`
10. 重跑 `python -m build && python -m twine check --strict dist/*` 验证

---

## 七、构建验证结果（2026-10-07 实测，py38，v0.8.0）

**命令**：
```bash
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m build          # EXIT=0
env -u PYTHONPATH D:/anaconda3/envs/py38/python.exe -m twine check --strict dist/*
```

**结果**：`BUILD_EXIT=0`；`twine check --strict` **双 PASSED**（`TWINE_EXIT=0`）。

| 产物 | 大小 | 对比 0.7.5 | 内容 |
|---|---|---|---|
| `dist/federated_survival-0.8.0.tar.gz` | **217 KB**（221822 B） | 212 KB（+2%） | **122 个文件**；顶层 `LICENSE` / `MANIFEST.in` / `PACKAGE_STRUCTURE.md` / `PKG-INFO` / `README.md` / `docs/`(12) / `experiments/`(4) / `federated_survival/` / `mkdocs.yml` / `pyproject.toml` / `setup.py` / `tests/`(33)，外加 setuptools 自动注入的 `setup.cfg`、`federated_survival.egg-info/` |
| `dist/federated_survival-0.8.0-py3-none-any.whl` | **144 KB**（147812 B） | 146 KB（+1%） | **64 个条目**：`federated_survival/` 58（含 `examples/` 19 + `data/real/` **2 CSV**）+ `dist-info/` 6 |

**wheel 元数据核验**：`Metadata-Version: 2.1`；`Version: 0.8.0`；`License: MIT`；
`License-File: LICENSE`；`Classifier: License :: OSI Approved :: MIT License`；
`Requires-Python: >=3.8`；`Description-Content-Type: text/markdown`；
入口 `federated-survival = federated_survival.cli:main`；
7 个 Python 版本 Classifier（3.8–3.14）。

**【0.7.5 新增核验】真实数据已正确打包**：
```
federated_survival/data/real/colon_death.csv
federated_survival/data/real/gbsg.csv
```
→ 0.7.0 报告列为"孤儿文件、永远打不进发行物"，**0.7.5 已闭环**：
`pyproject.toml` 的 `package-data` 新增 `data/real/*.csv`，wheel 与 sdist 双双命中。
注意 `MANIFEST.in:15 recursive-exclude data *.csv` 只排除**仓库根** `data/`，
不匹配 `federated_survival/data/`，两者不冲突（已实测确认）。

**发行物纯净度核验**：sdist 内**未混入** `__pycache__`、`*.pyc`、`.DS_Store`、`.idea`、
`*.log`、`PACKAGE_AUDIT.md`、`JCB-2026-0108_audit_checklist.md`、`bug-report-*.md`、
`results/`、`.pypirc`、`.coverage`。（`setup.cfg`、`federated_survival.egg-info/`
出现在 sdist 顶层属 setuptools 自动注入，正常。）

**构建告警（均为无害提示）**：
- `no previously-included files found matching 'requirements-paper-figure2-lock.txt'`
  → 死规则，**仍待处理**（见第三节 3）。
- `warning: build_py/install_lib: byte-compiling is disabled, skipping.`
  → wheel 构建规范行为（字节码由 pip 安装时编译），正常。
- 其余 `__pycache__` / `*.py[cod]` / `.DS_Store` / `*.so` / `*.xml` / `.pytest_cache` /
  `results` 的 "no previously-included" 告警，是清理规则生效的正常表现。

---

## 八、测试与覆盖率实测（2026-10-07，py38，v0.7.5）

| 指标 | 0.7.0 | **0.7.5** |
|---|---|---|
| 测试文件 | 24 | **30 个 `test_*.py`**（+2 份 README + `run_dp_tests.py`） |
| 用例数（通过） | 206 | **318**（318 个 `.`，0 `F`/`E`/`s`/`x`，`PYTEST_EXIT=0`） |
| 语句覆盖 | 85.5% | **90.2%**（3634 语句 / 355 未覆盖） |
| 分支出口覆盖 | 71.7% | **80.3%**（1406 分支 / 277 未出口） |
| coverage.py 总行 | — | `TOTAL 3602 355 1394 276 86%` |
| 零覆盖文件 | — | **0 个** |

**覆盖率最低的 8 个文件**（`report --sort=cover` 升序）：

| 文件 | 语句/未覆盖 | 分支/未出口 | 覆盖率 | 说明 |
|---|---|---|---|---|
| `_compat.py` | 7 / 4 | 2 / 1 | **44%** | **环境相关**：本机 scipy 1.10.1 **有** `simps`，走原生路径，`if not hasattr(...)` 的 fallback 体永不执行 |
| `protocols/parameter.py` | 104 / 18 | 35 / 10 | **77%** | FedOpt 的部分优化器分支 |
| `experiments/visualization.py` | 252 / 40 | 131 / 23 | **77%** | 大量绘图分支 |
| `experiments/config_schema.py` | 211 / 32 | 81 / 32 | **78%** | YAML/JSON schema 校验的错误分支 |
| `api.py` | 160 / 25 | 70 / 18 | **80%** | 高层估计器的异常路径 |
| `experiment_api.py` | 227 / 39 | 141 / 35 | **79%** | 对比矩阵的组合分支 |
| `core/differential_privacy.py` | 148 / 26 | 60 / 11 | **76%** | Laplace 分支 + `compute_privacy_budget`（生产 0 调用）；新增的死参数映射与 `describe_noise_knobs` 已覆盖 |
| `core/config.py` | 206 / 21 | 120 / 22 | **87%** | 大量互斥校验分支；DP 死参数告警的触发/静默条件均已覆盖 |
| `protocols/base.py` | 70 / 9 | 24 / 6 | **84%** | 能力声明的默认实现 |

> 注：`_compat.py` 的 44% 是**环境造成的统计假象**，不是测试缺口。该文件用
> `if not hasattr(scipy.integrate, "simps"):` 守卫，只在 scipy ≥ 1.14（`simps` 被移除）时
> 才定义 `_simps` 兜底函数。**本机 scipy 1.10.1 `hasattr(si,'simps') == True`**，
> 因此整个 `if` 块（4 条语句 + 1 条分支）永不执行 → 覆盖率必然是 3/7。
> CI 若跑在 scipy ≥ 1.14 的新 Python 版本上，比例会反过来。
> 要消除这个假象需用 `monkeypatch` 删属性后重新导入，属环境模拟测试，优先级低。

**满覆盖文件**（10 个）：`__init__.py`、`core/server.py`(100%)、
`data/generator.py`(100%)、`experiments/config_runner.py`(100%)、
以及 6 个子包 `__init__.py`。

**CI 门禁**：`ci.yml` 执行
```bash
python -m coverage run --branch --source=federated_survival -m pytest
python -m coverage report --fail-under=80
```
→ 实际 86%（coverage.py 综合行）/ 90.2%（语句），**高于门槛 6–10 个百分点**。
刻意用 `coverage.py` 而非 `pytest-cov`，保证本地与 CI 口径一致。

> ⚠️ **两个统计口径陷阱**（复测必读）：
> ① `federated_survival/examples/`（1320 行）**永远不进覆盖率**——该目录无 `__init__.py`，
>    非子包，`--source` 扫描直接跳过、`report` 也不列。
> ② `coverage html` 会在输出目录自动生成 `.gitignore`（内容 `*`）→
>    `docs/coverage_report/` 天然不入库，这是正确行为。
