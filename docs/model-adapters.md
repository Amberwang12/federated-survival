# Model adapters

An adapter isolates the operations that differ across survival objectives:
label transformation, output dimensionality, network construction, PyCox
wrapping, baseline-hazard estimation, and survival prediction.

Built-in adapters can be inspected with:

```python
from federated_survival import available_model_adapters

print(available_model_adapters())
```

## Register a custom model

```python
from federated_survival import ModelAdapter, register_model_adapter

class MyAdapter(ModelAdapter):
    name = "MySurvivalModel"

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return MyPyCoxCompatibleWrapper(network, optimizer)

register_model_adapter(MyAdapter.name, MyAdapter)
```

Once registered, the model name can be passed to `FSAConfig` or
`FederatedSurvival` without changing client, server, or aggregation code.

Every third-party adapter should be covered by the same contract checks as the
built-in models, including valid target preparation and finite, bounded,
monotone survival predictions.
