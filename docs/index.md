# federated-survival

`federated-survival` is an extensible Python toolkit for running multiple
survival objectives through one federated execution path.  The current release
provides adapters for CoxPH, DeepSurv, CoxCC, CoxTime, LogisticHazard,
PC-Hazard, and DeepHit. Four protocol adapters separate the communication and
server-update rule from model-specific behavior: FedAvg, FedProx, FedOpt, and
a linear-Cox-only WebDISCO-style risk-set-statistics protocol.

The software contribution is the common model contract, high-level API,
validation, and reproducible experiment infrastructure.  FSA-MVAE is retained
as an optional augmentation component and is not presented as a new method in
the software package.

## Installation check

```bash
pip install federated-survival
federated-survival doctor
```

## Design goals

- one public workflow across heterogeneous survival objectives;
- explicit, testable model-specific adapters;
- explicit, testable federated protocol adapters;
- fail-fast validation of invalid survival predictions;
- machine-readable configurations, metrics, failures, and manifests;
- one YAML workflow for real-data loading, client partitioning, paired
  Center/Federated/Local experiments, and automatic figures;
- backward compatibility with `FSAConfig` and `FSARunner`.

## Compose an experiment

For real or simulated data, partitioning, paired Center/Federated/Local
comparisons, and selectable plots without editing YAML, see
[Composable Python experiments](interactive-experiments.md).
For MVAEC/MVAES comparison plots and experimental update-perturbation
configuration, see [Augmentation and privacy calls](augmentation-privacy.md).
For recorded experiments with a manifest, see
[Configuration-driven experiments](configuration.md).

!!! warning "Privacy scope"
    The optional update-perturbation utilities do not currently provide a
    formal record-level, multi-round differential-privacy guarantee.  See the
    privacy page before using them with sensitive data.
