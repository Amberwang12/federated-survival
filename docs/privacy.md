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

## What the mechanism perturbs, and where the noise is added

The perturbation is applied by `DifferentialPrivacy.privatize_model_update()`,
which runs on the **client side only**.  For one client update the method:

1. forms the update delta `local - global` over all floating-point parameters;
2. clips it to `dp_clip_norm` in L2 norm, so the sensitivity bound holds
   regardless of how large the raw update was;
3. adds mechanism noise to the **clipped delta** with per-element scale
   `dp_noise_multiplier * dp_sensitivity * dp_clip_norm / sqrt(K)` (Gaussian) or
   the Laplace equivalent (see the table below);
4. writes back `global + perturbed_delta`.

Two consequences follow from step 4 and are worth stating plainly:

- **No noise is added during server-side aggregation.**  `apply_dp_to_weights()`
  is retained for API compatibility but returns its input unchanged, because
  adding noise after the average would perturb the aggregate itself rather than
  any individual contribution.
- **`add_noise_to_weights()` is not on the training path.**  It would perturb
  absolute weights, which is unsound because absolute weights are unbounded and
  carry no sensitivity bound.  `privatize_model_update()` is the function the
  federated loop actually calls.

## Sensitivity, and what the `1/sqrt(K)` factor means

`dp_sensitivity` is the sensitivity bound applied to the **clipped client
update**, not to a single record.  A single record's influence is never bounded
here, so this mechanism inherits no record-level guarantee.

The `1/sqrt(K)` factor deserves a caveat, because it is easy to over-read.
Averaging `K` independent client updates aligns their signal contributions while
their independent noise accumulates in quadrature, so dividing the per-client
noise by `sqrt(K)` is the scaling that keeps the aggregate noise-to-signal ratio
stable as `K` grows.  **This is a variance-reduction argument, not a privacy
accounting argument.**  Under the standard central-DP formulation a sensitivity
bound scales linearly with the sampling rate `1/K`, not with `1/sqrt(K)`, and a
local-DP formulation would not divide by `K` at all.  This package does not
implement either accounting, so the factor should be read as "how much noise is
actually injected", and the reported epsilon should not be derived from it.

If you need a defensible `(epsilon, delta)` statement, this mechanism is the
wrong starting point; the missing pieces are listed at the top of this page.

## Which parameter actually controls the noise

The two mechanisms are driven by **different** configuration fields, so
switching `dp_mechanism` changes which field is effective:

| `dp_mechanism` | Noise scale | Driver | Ignored fields |
|---|---|---|---|
| `gaussian` | `dp_noise_multiplier x dp_sensitivity` | `dp_noise_multiplier` | `dp_epsilon`, `dp_delta` |
| `laplace` | `dp_sensitivity / dp_epsilon` | `dp_epsilon` | `dp_noise_multiplier`, `dp_delta` |
| `exponential` | candidate selection only | `dp_epsilon` | `dp_noise_multiplier`, `dp_delta` |

Because of this, lowering `dp_epsilon` under the Gaussian mechanism does not
reduce the injected noise at all, and lowering `dp_noise_multiplier` under the
Laplace mechanism does not either.  `dp_delta` never affects the noise scale
under any mechanism: the noise is defined directly from the driver field, and
`dp_delta` is read only by the Renyi conversion helper, which has no caller in
the training path.

`FSAConfig` emits a `UserWarning` when an ignored field has been set away from
its default, so the misconfiguration surfaces at configuration time instead of
silently producing an unprotected model.  Fields left at their default stay
silent.  Both `get_privacy_info()` implementations report `noise_driver` and
`inactive_parameters` so the state can be inspected programmatically:

```python
info = FederatedSurvival(dp_mechanism="gaussian", dp_epsilon=0.5).get_privacy_info()
info["noise_driver"]         # 'dp_noise_multiplier'
info["inactive_parameters"]  # ['dp_epsilon', 'dp_delta']
```

Use `dp_noise_multiplier` to tune Gaussian noise and `dp_epsilon` to tune
Laplace noise.  Treat the reported epsilon as a nominal configuration value in
both cases, because this package performs no accounting that would convert the
noise into an achieved budget.

## Recommended ranges, and how to tell that noise has swamped the signal

Because the mechanism adds noise to **every** `global_rounds * local_steps`
update, the per-update noise compounds over training.  The practical constraint
is therefore not the nominal epsilon but the ratio of injected noise to the
clipped update it is added to.

That ratio is the quantity to watch.  `DifferentialPrivacy` records it on every
update as `last_noise_to_signal`, alongside `last_noise_sigma` (the per-element
standard deviation actually used) and `last_clipped_update_norm` (the norm the
noise is measured against).  Below roughly 1 the noise is smaller than the signal
it perturbs; as it climbs past 1 the published update is dominated by noise and
the model stops learning usefully.

Note that these three attributes live on the per-client `DifferentialPrivacy`
instance, and the client objects are created and discarded inside a training
round.  They are therefore **not reachable through `FederatedSurvival` or
`FSARunner` after `fit()`** — read them only if you drive
`privatize_model_update()` yourself.  For an end-to-end run the usable signal is
the behaviour of the predictions themselves: a C-index of exactly 0.5000
together with curves that are identical across samples means the privacy budget
drowned the signal at this sample size.  That is reported rather than hidden,
and it should be read as a diagnostic, not as a result.

Measured on `SDGM1` (n=200, 5 features, censoring 0.4, 3 clients, IID,
DeepSurv, 5 rounds x 3 local steps), averaged over seeds 0-2:

| `dp_noise_multiplier` | `dp_clip_norm` | C-index |
|---|---|---|
| DP off | - | 0.5454 |
| 0.01 | 1.0 | 0.5357 |
| 0.05 | 1.0 | 0.5348 |
| 0.1 | 1.0 | 0.5200 |
| 0.3 | 1.0 | 0.5349 |
| 0.5 | 1.0 | 0.5579 |
| 1.0 | 1.0 | 0.5602 |
| **2.0** | 1.0 | **0.5000** (degenerate) |
| 0.1 | 0.1 | 0.5357 |
| 0.1 | 0.01 | 0.5064 |
| 0.5 | 0.01 | 0.5040 |

Recommended starting points for this configuration:

- `dp_noise_multiplier` in **[0.01, 0.1]** with `dp_clip_norm = 1.0`.  Utility is
  within noise of the no-DP baseline across this range, and the collapse only
  begins above it.
- Avoid `dp_clip_norm` below **0.1** unless the noise multiplier is lowered to
  match.  `dp_noise_multiplier * dp_clip_norm` is the product that sets the
  injected noise, so shrinking the clip norm without shrinking the multiplier
  shrinks the signal faster than the noise.
- Treat C-index **0.5000 with `identical_across_samples=True`** as a failure
  signal, not a result.  It means the privacy budget drowned the signal at this
  sample size.

Two properties of that table are worth keeping in mind.  The C-index differences
across `dp_noise_multiplier` between 0.01 and 1.0 are small and not monotone,
which is expected: with this configuration and budget the perturbation is not
the dominant error term.  And small samples make DP far more expensive than the
nominal budget suggests — a genuinely private setting needs far fewer effective
rounds, so the useful lever is often **fewer, larger rounds** rather than a
smaller multiplier.  `dp_delta` does not appear in any of this because it never
reaches the noise scale.
