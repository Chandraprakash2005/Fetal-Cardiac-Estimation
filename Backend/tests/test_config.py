"""tests/test_config.py

Implements: CLAUDE.md §8 item 9 -- config.py loading/merging.
"""

from fcac.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.preprocessing.window.size_samples == 5000
    assert cfg.pinn.loss_weights.data == 0.7
    assert cfg.pinn.loss_weights.physics == 0.3


def test_dataset_override_merges_on_top_of_default():
    cfg = load_config(dataset="ninfea")
    assert cfg.dataset.name == "ninfea"
    assert cfg.preprocessing.notch_enabled is True
    # default.yaml keys not touched by the dataset override survive the merge.
    assert cfg.preprocessing.window.size_samples == 5000


def test_adfecgdb_disables_notch():
    cfg = load_config(dataset="adfecgdb")
    assert cfg.preprocessing.notch_enabled is False


def test_config_is_frozen():
    cfg = load_config()
    try:
        cfg.seed = 0
        assert False, "Config should be immutable"
    except AttributeError:
        pass


def test_missing_key_raises_helpful_error():
    cfg = load_config()
    try:
        _ = cfg.this_key_does_not_exist
        assert False, "expected AttributeError"
    except AttributeError as exc:
        assert "configs/default.yaml" in str(exc)
