# Examples

This directory contains package-level examples and reproducible configurations.

## Recommended starting points

### High-level API

~~~bash
python -m federated_survival.examples.basic_usage_example
~~~

### Seven-model smoke configuration

~~~bash
federated-survival run \
  --config federated_survival/examples/seven_model_smoke.json \
  --output results/seven_model_smoke
~~~

### Four-protocol coverage configuration

~~~bash
federated-survival run \
  --config federated_survival/examples/four_protocol_smoke.json \
  --output results/four_protocol_smoke
~~~

This configuration runs FedAvg, FedProx, and FedOpt with all seven model
adapters and WebDISCO-style Cox with compatible CoxPH settings.

### Protocol comparison example

~~~bash
python -m federated_survival.examples.protocol_comparison_example
~~~

## Additional examples

- data_generation_example.py: simulated survival-data generators;
- data_partitioning_example.py: IID and non-IID client partitions;
- model_comparison_example.py: common model-adapter interface;
- data_augmentation_example.py: optional FSA-MVAE augmentation;
- files containing privacy, differential_privacy, or DP: experimental
  client-update perturbation utilities.

The perturbation examples do not provide a formal record-level, multi-round
differential-privacy guarantee. See docs/privacy.md before using them.
