# Privacy scope

The package currently includes experimental clipping and perturbation of a
complete client model update.  This is not the same as record-level DP-SGD.

The current implementation does not provide:

- per-example gradient clipping;
- privacy amplification analysis;
- a multi-round privacy accountant;
- secure aggregation;
- an end-to-end `(epsilon, delta)` guarantee.

Consequently, `dp_epsilon` and `dp_delta` must not be reported as a formal
privacy guarantee for a federated training run.  Production use with sensitive
data requires a separately validated privacy mechanism and threat model.
For the current experimental configuration fields and a callable example, see
[Augmentation and privacy calls](augmentation-privacy.md). The high-level
`get_privacy_info()` method reports configured inputs, not an achieved budget.
