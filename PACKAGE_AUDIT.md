# Python 软件包结构审计报告

**审计对象**：`federated-survival-v0.7.0-software-paper/`（`federated-survival` v0.7.0）
**审计日期**：2026-09-27
**仓库体积**：8.2 MB（其中 `results/` 5.5 MB、`federated_survival/` 1.2 MB、`tests/` 772 KB）
**结论速览**：软件包本体（`federated_survival/`）完整健康；仓库卫生（垃圾文件、生成物、`results/`）、版本控制（无 `.git`）、学术软件合规元数据（`CITATION.cff` 等）三处为主要缺口。

---

## 一、必须保留 ✅

### A. 软件包本体（发行物核心）

| 路径 | 说明 |
|---|---|
| `federated_survival/__init__.py` | 公共导出 + `__version__ = "0.7.0"` |
| `federated_survival/api.py` | 高层 `FederatedSurvival` 估计器 |
| `federated_survival/experiment_api.py` | Center/Federated/Local 对比与绘图 |
| `federated_survival/cli.py` | `federated-survival` 命令入口 |
| `federated_survival/_compat.py` | 兼容层（`scipy.integrate.simps` 等） |
| `federated_survival/core/` | `config / client / server / runner / augmenter / mvae / differential_privacy / _losses / _minibatch` |
| `federated_survival/models/adapters.py` | 7 个生存模型适配器 + 注册表 |
| `federated_survival/protocols/` | `base / parameter(FedAvg,FedProx,FedOpt) / webdisco` |
| `federated_survival/data/` | `generator / loader / splitter / preprocessing / workflow / augmentation_workflow` |
| `federated_survival/experiments/` | `config_runner / config_schema / infrastructure / baselines / visualization / workflow_runner` |
| `federated_survival/utils/metrics.py` | C-index、IBS |
| 各子包 `__init__.py` | 共 8 个，齐全 |

### B. 打包与元数据

| 路径 | 说明 |
|---|---|
| `pyproject.toml` | 唯一权威构建配置（PEP 621） |
| `setup.py` | 空壳 shim，`setup()` 转调；**建议保留**（兼容老工具链） |
| `MANIFEST.in` | sdist 清单 |
| `LICENSE` | MIT |
| `README.md` | 项目首页，被 `readme =` 引用 |
| `PACKAGE_STRUCTURE.md` | 结构说明（被 MANIFEST `include`） |

### C. 测试、文档、CI、可复现资产

| 路径 | 说明 |
|---|---|
| `tests/`（25 个 `test_*.py` + `run_dp_tests.py` + 2 份 README） | 测试套件，**必须随源码保留** |
| `docs/`（11 篇）+ `mkdocs.yml` | MkDocs 文档站 |
| `.github/workflows/ci.yml`、`docs.yml` | CI 与文档部署 |
| `federated_survival/examples/` | 可运行示例 + smoke 配置（`.yaml/.json/.md`） |
| `experiments/`（4 个 `.py`） | 审计/验证脚本，MANIFEST 已纳入 sdist |
| `data/real/gbsg.csv`、`data/real/colon_death.csv` | docs 示例引用（`data/real/gbsg.csv`）；发行物中按设计排除 |
| ~~`software_validation/README.md`~~ | 验证说明（XML 不入库）。**【2026-09-27 状态】** 该目录已被整体删除，含此 README；`MANIFEST.in` 对应行亦已删除。如需保留验证口径记录，建议后续恢复 |

---

## 二、建议删除 🗑️

> **说明**：以下删除均在"打包/仓库卫生"层面。`results/` 若为论文交付物请先归档再删。执行前建议先备份。

| 类别 | 路径 | 理由 |
|---|---|---|
| macOS 垃圾 | `.DS_Store`（6 处：根、`federated_survival/`、`data/`、`experiments/`、`results/`、`tests/`） | 系统文件，`.gitignore` 已忽略 |
| 字节码缓存 | 8 个 `__pycache__/`、111 个 `*.pyc` | 构建产物 |
| 测试缓存 | `.pytest_cache/` | 构建产物 |
| IDE | `.idea/`（6 文件） | 已 gitignore，个人设置 |
| 构建产物 | `federated_survival.egg-info/`（6 文件） | `pip install -e` 产物，已 gitignore；**且已过期**（`SOURCES.txt` 缺 `test_metrics.py`、`test_stable_survival.py`） |
| 安装日志 | `install_py12.log` | 临时日志 |
| 散落生成图 | `dp_mechanisms_noise_distributions.png`、`dp_mechanisms_privacy_utility_tradeoff.png` | 生成结果，应落 `results/` 或 `docs/assets/` |
| 验证报告 | `software_validation/pytest_180.xml` / `_183.xml` / `_184.xml` | 本地验证产物，`README.md` 明示"intentionally excluded" |
| 生成结果 | `results/`（17 PNG + 7 CSV，5.5 MB） | 已 gitignore；占全仓 2/3 体积 |
| 非软件包内容 | `JCB-2026-0108_audit_checklist.md` | 审稿核查清单，属过程文档；建议移出仓库或放入 `docs/` 之外的归档位 |
| **孤儿文件** | `federated_survival/data/real/gbsg.csv` | 全库零引用；`package-data` 不含 `*.csv`、MANIFEST 又 `recursive-exclude` → **永远打不进发行物**；且与根 `data/real/gbsg.csv` 重复 |

---

## 三、缺失的文件 ❌

### 1. 版本控制与合规（优先级最高）

| 缺失项 | 说明 |
|---|---|
| **`.git` / git 仓库** | **全仓无版本控制**。对可复现研究是最大缺口：无 commit 历史、无 tag、无法 `git archive` |
| `CITATION.cff` | 学术软件刚需（GitHub "Cite this repository"）；软件论文尤其必要 |
| `CHANGELOG.md` | 版本变更记录，`docs/releasing.md` 的发布流程实际需要 |
| `CONTRIBUTING.md` | 贡献指南（CI 已跑 flake8/mypy/black 但无成文规范） |
| `CODE_OF_CONDUCT.md` | 社区合规 |
| `SECURITY.md` | 安全披露渠道 |

### 2. 工程规范

| 缺失项 | 说明 |
|---|---|
| `.pre-commit-config.yaml` | `dev` extras 已装 `black/flake8/mypy`，但无 pre-commit 钩子 |
| `.gitattributes` | 统一换行符（跨 Win/Linux CI）、标记二进制 |
| `.editorconfig` | 统一缩进/编码 |
| `tests/conftest.py` | **无共享 fixture**；CI 的 `NUMBA_*/OMP_*` 环境变量目前只能散落在 workflow |
| `.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/` | 协作模板 |
| `docs/assets/` | 文档站图片无归属目录（现根目录 PNG 无处安放） |

### 3. 打包一致性

| 缺失项 | 说明 |
|---|---|
| `requirements-paper-figure2-lock.txt` | `MANIFEST.in:14` 有 `exclude` 规则，但**文件根本不存在** → 死规则，应删规则或补文件 |
| `pyproject.toml` 的 `license-files` | PEP 639 下建议显式声明 |
| `"License :: OSI Approved :: MIT License"` classifier | `classifiers` 缺失 License 项 |
| `federated_survival/py.typed` | 若有类型标注意图（`mypy` 在 dev extras） |

---

## 四、打包配置本身的不一致（无需增删文件，需改配置）

1. ~~**wheel 与 sdist 内容不一致**：`package-data` 只含 `examples/*.yaml`、`examples/*.json`，而 `MANIFEST.in` 还含 `examples/*.md`。→ `examples/*.md`、`examples/*.py` 进 sdist 但不进 wheel。~~
   **【2026-09-27 实测修正】** 此条**判断有误，已撤销**。实际解包 `federated_survival-0.7.0-py3-none-any.whl` 确认：`examples/` 下 **19 个文件全部打进 wheel**，含 9 个 `.py`、3 个 `.md`、4 个 `.yaml`、3 个 `.json`。原因是 setuptools 在**使用 pyproject.toml 配置时 `include_package_data` 默认为 True**，`MANIFEST.in` 收录的包内文件会同步进入 wheel，`package-data` 并非唯一闸门。结论：wheel/sdist 内容一致，无需调整。
2. **`package-data` 里的 `*.txt`**：包内并无 `.txt` 文件，属空规则（无害，可清理）。
3. **`namespaces = true`**：使无 `__init__.py` 的 `federated_survival/data/real/` 被视作隐式命名空间包；与"该目录不入发行物"的意图冲突，建议直接删除该目录（见二、孤儿文件）。
4. **`[project] license` 写法与 py3.8 不兼容（已修复）**：原 `license = "MIT"` 是 PEP 639 新写法，**要求 setuptools ≥ 77**；而 Python 3.8 最高只能装 setuptools 75.x（76 起放弃 3.8），导致 py38 下构建直接失败，**CI 的 py3.8 matrix job 也会挂**。已改为新旧皆兼容的 `license = { text = "MIT" }`，并补 `"License :: OSI Approved :: MIT License"` 分类器。
5. **`addopts` 含 `--strict-markers`**：需确认全部测试只用已注册 marker，否则 CI 会失败（当前 214 passed，暂无问题）。

---

## 五、建议的清理命令（**未执行，待确认**）

```bash
cd /c/Users/skyee/federated_survival/federated-survival-v0.7.0-software-paper/federated-survival-v0.7.0-software-paper

# 0) 先备份（强烈建议）
tar -czf ../federated-survival-v0.7.0-backup-$(date +%Y%m%d).tar.gz .

# 1) 系统垃圾 + 缓存（安全）
find . -name ".DS_Store" -delete
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

# 2) 构建产物 / IDE（可再生成）
rm -rf .idea federated_survival.egg-info install_py12.log

# 3) 生成物 / 验证产物（如已归档再删）
rm -rf results software_validation/pytest_18*.xml
rm -f dp_mechanisms_noise_distributions.png dp_mechanisms_privacy_utility_tradeoff.png

# 4) 孤儿文件（确认 docs 示例用的是根 data/real/gbsg.csv）
rm -rf federated_survival/data/real
```

---

## 六、执行顺序建议

1. 备份 → 2. 清理垃圾/缓存/构建产物（无风险）→ 3. 归档 `results/` 与验证 XML 后删除 → 4. 删除孤儿 `federated_survival/data/real/` + 修 `MANIFEST.in` 死规则 → 5. 补 `.git` 并首次 commit → 6. 补 `CITATION.cff`、`CHANGELOG.md`、`tests/conftest.py`、`.pre-commit-config.yaml` → 7. 对齐 wheel/sdist 的 `package-data` → 8. 重跑 `python -m build && python -m twine check --strict dist/*` 验证。

---

## 七、构建验证结果（2026-09-27 实测，py38）

**命令**：`D:\anaconda3\envs\py38\python.exe -m build` + `python -m twine check --strict dist/*`
**结果**：`EXIT=0`，sdist 与 wheel 均构建成功，`twine check --strict` **双 PASSED**。

| 产物 | 大小 | 内容 |
|---|---|---|
| `dist/federated_survival-0.7.0.tar.gz` | 175 KB | `federated_survival/` + `docs/` + `tests/` + `experiments/` + `README.md` / `PACKAGE_STRUCTURE.md` / `mkdocs.yml` / `pyproject.toml` / `setup.py` / `MANIFEST.in` / `LICENSE` |
| `dist/federated_survival-0.7.0-py3-none-any.whl` | 133 KB | 62 个文件：`federated_survival/` 56（含 `examples/` 19 个）+ `dist-info/` 6 |

**wheel 元数据核验**：`Metadata-Version: 2.1`；`License: MIT`；`License-File: LICENSE`；`Classifier: License :: OSI Approved :: MIT License`；`Requires-Python: >=3.8`；`Description-Content-Type: text/markdown`；入口 `federated-survival = federated_survival.cli:main`。

**发行物纯净度核验**：sdist 内**未混入** `.idea/`、`.workbuddy/`、`.DS_Store`、`__pycache__/`、`*.pyc`、`*.csv`、`results/`、`*.log`、`PACKAGE_AUDIT.md`、`JCB-2026-0108_audit_checklist.md`。（`setup.cfg`、`federated_survival.egg-info/` 出现在 sdist 顶层属 setuptools 自动注入，正常。）

**本次实际改动**：
- `pyproject.toml` — ① `license = "MIT"` → `license = { text = "MIT" }`；② 新增 License 分类器。**这是让 py3.8 能构建的必要修复。**
- `MANIFEST.in` — 删除原第 13 行 `recursive-include software_validation *.md`（该目录已被整体删除）。删后重新构建，sdist 体积 175583 → 175518 字节。

**构建告警（删行后复测，均为无害提示）**：
- ~~`no files found matching '*.md' under directory 'software_validation'`~~ → **已通过删除 MANIFEST 对应行消除** ✅
- `no previously-included files found matching 'requirements-paper-figure2-lock.txt'` → 该文件本就不存在，属死规则（见第三节），**仍待处理**：删 `MANIFEST.in:14` 或补文件。
- `warning: build_py/install_lib: byte-compiling is disabled, skipping.` → wheel 构建规范行为（字节码由 pip 安装时编译），正常。
- 其余 `__pycache__` / `*.py[cod]` / `.DS_Store` / `*.so` / `*.xml` / `.pytest_cache` / `results` 的 "no previously-included" 告警，是清理生效的正常表现。

**删行后复测结论**：`python -m build` EXIT=0；`twine check --strict` 双 **PASSED**；sdist 顶层干净（`LICENSE` / `MANIFEST.in` / `PACKAGE_STRUCTURE.md` / `PKG-INFO` / `README.md` / `docs/` / `experiments/` / `federated_survival/` / `mkdocs.yml` / `pyproject.toml` / `setup.py` / `tests/`，外加 setuptools 自动注入的 `setup.cfg`、`federated_survival.egg-info/`）。


