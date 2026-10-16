# configs/experiments/

One YAML file per ablation/run, containing only the keys being overridden. Loaded as the third
(optional) layer on top of `configs/default.yaml` and `configs/datasets/<name>.yaml`:

```
Config = merge(default.yaml, datasets/<dataset>.yaml, experiments/<experiment>.yaml)
```

Example -- `configs/experiments/bandpass_spec_preset.yaml`:

```yaml
preprocessing:
  bandpass_preset: spec_0_150
```
