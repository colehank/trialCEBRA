"""Integration tests for TrialCEBRA sklearn wrapper — new 3-conditional design."""

import numpy as np

from trial_cebra import TrialCEBRA
from trial_cebra.distribution import TRIAL_CONDITIONALS, TrialAwareDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NTRIAL, NTIME, NNEURO, ND = 6, 40, 20, 8


def _make_epoch_data():
    X = np.random.randn(NTRIAL, NTIME, NNEURO).astype(np.float32)
    y2d = np.random.randn(NTRIAL, ND).astype(np.float32)
    y3d = np.random.randn(NTRIAL, NTIME, ND).astype(np.float32)
    return X, y2d, y3d


def _model(
    conditional, sample_fix_trial=False, sample_exclude_intrial=True, output_dim=3, max_iter=5, **kw
):
    return TrialCEBRA(
        model_architecture="offset1-model",
        conditional=conditional,
        time_offsets=5,
        delta=0.1,
        sample_fix_trial=sample_fix_trial,
        sample_exclude_intrial=sample_exclude_intrial,
        max_iterations=max_iter,
        batch_size=32,
        output_dimension=output_dim,
        **kw,
    )


# ---------------------------------------------------------------------------
# "time" conditional
# ---------------------------------------------------------------------------


class TestTimeFit:
    def test_fit_no_y(self):
        X, _, _ = _make_epoch_data()
        model = _model("time")
        model.fit(X)
        emb = model.transform(X.reshape(NTRIAL * NTIME, -1))
        assert emb.shape == (NTRIAL * NTIME, 3)

    def test_fit_and_transform_epochs(self):
        X, _, _ = _make_epoch_data()
        model = _model("time")
        model.fit(X)
        emb = model.transform_epochs(X)
        assert emb.shape == (NTRIAL, NTIME, 3)

    def test_loss_recorded(self):
        X, _, _ = _make_epoch_data()
        model = _model("time", max_iter=5)
        model.fit(X)
        assert len(model.state_dict_["loss"]) == 5

    def test_distribution_is_trial_aware(self):
        X, _, _ = _make_epoch_data()
        model = _model("time")
        model.fit(X)
        assert isinstance(model.distribution_, TrialAwareDistribution)
        assert model.distribution_.conditional == "time"

    def test_sample_fix_trial_ignored(self):
        X, _, _ = _make_epoch_data()
        model = _model("time", sample_fix_trial=True)
        model.fit(X)
        assert not hasattr(model.distribution_, "_locked_target_trials")


# ---------------------------------------------------------------------------
# "delta" conditional
# ---------------------------------------------------------------------------


class TestDeltaFit:
    def test_fit_and_transform(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta")
        model.fit(X, y2d)
        emb = model.transform(X.reshape(NTRIAL * NTIME, -1))
        assert emb.shape == (NTRIAL * NTIME, 3)

    def test_transform_epochs(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta")
        model.fit(X, y2d)
        emb = model.transform_epochs(X)
        assert emb.shape == (NTRIAL, NTIME, 3)

    def test_loss_recorded(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta")
        model.fit(X, y2d)
        assert len(model.state_dict_["loss"]) == 5

    def test_distribution_is_trial_aware(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta")
        model.fit(X, y2d)
        assert isinstance(model.distribution_, TrialAwareDistribution)
        assert model.distribution_.conditional == "delta"

    def test_sample_fix_trial_creates_locked_map(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta", sample_fix_trial=True)
        model.fit(X, y2d)
        assert hasattr(model.distribution_, "_locked_target_trials")

    def test_no_sample_fix_trial_no_locked_map(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta", sample_fix_trial=False)
        model.fit(X, y2d)
        assert not hasattr(model.distribution_, "_locked_target_trials")


# ---------------------------------------------------------------------------
# "time_delta" conditional
# ---------------------------------------------------------------------------


class TestTimeDeltaFit:
    def test_fit_and_transform(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta")
        model.fit(X, y3d)
        emb = model.transform(X.reshape(NTRIAL * NTIME, -1))
        assert emb.shape == (NTRIAL * NTIME, 3)

    def test_transform_epochs(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta")
        model.fit(X, y3d)
        emb = model.transform_epochs(X)
        assert emb.shape == (NTRIAL, NTIME, 3)

    def test_loss_recorded(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta")
        model.fit(X, y3d)
        assert len(model.state_dict_["loss"]) == 5

    def test_distribution_is_trial_aware(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta")
        model.fit(X, y3d)
        assert isinstance(model.distribution_, TrialAwareDistribution)
        assert model.distribution_.conditional == "time_delta"

    def test_sample_fix_trial_creates_locked_map(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta", sample_fix_trial=True)
        model.fit(X, y3d)
        assert hasattr(model.distribution_, "_locked_target_trials")

    def test_no_sample_fix_trial_no_locked_map(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta", sample_fix_trial=False)
        model.fit(X, y3d)
        assert not hasattr(model.distribution_, "_locked_target_trials")


# ---------------------------------------------------------------------------
# Native CEBRA conditionals pass through unchanged
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_native_time_passthrough(self):
        neural = np.random.randn(200, NNEURO).astype(np.float32)
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time",
            time_offsets=5,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural)
        assert model.transform(neural).shape == (200, 3)

    def test_native_time_delta_passthrough(self):
        neural = np.random.randn(200, NNEURO).astype(np.float32)
        continuous = np.random.randn(200, ND).astype(np.float32)
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time_delta",
            time_offsets=5,
            delta=0.1,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural, continuous)
        assert model.transform(neural).shape == (200, 3)

    def test_native_not_trial_aware(self):
        neural = np.random.randn(200, NNEURO).astype(np.float32)
        continuous = np.random.randn(200, ND).astype(np.float32)
        model = TrialCEBRA(
            model_architecture="offset1-model",
            conditional="time_delta",
            time_offsets=5,
            delta=0.1,
            max_iterations=5,
            batch_size=32,
            output_dimension=3,
        )
        model.fit(neural, continuous)
        assert not hasattr(model, "distribution_") or not isinstance(
            model.distribution_, TrialAwareDistribution
        )


# ---------------------------------------------------------------------------
# fit_epochs convenience wrapper
# ---------------------------------------------------------------------------


class TestFitEpochs:
    def test_delta_fit_epochs(self):
        X, y2d, _ = _make_epoch_data()
        model = _model("delta")
        model.fit_epochs(X, y2d)
        assert model.transform_epochs(X).shape == (NTRIAL, NTIME, 3)

    def test_time_delta_fit_epochs(self):
        X, _, y3d = _make_epoch_data()
        model = _model("time_delta")
        model.fit_epochs(X, y3d)
        assert model.transform_epochs(X).shape == (NTRIAL, NTIME, 3)


# ---------------------------------------------------------------------------
# All conditionals covered
# ---------------------------------------------------------------------------


def test_trial_conditionals_set():
    assert TRIAL_CONDITIONALS == {"time", "delta", "time_delta"}
