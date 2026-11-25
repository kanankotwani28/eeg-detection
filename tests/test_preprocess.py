import mne
import numpy as np
import pytest

from src.data.synthetic import MUMTAZ_EEG, TDBRAIN_EEG, TDBRAIN_EXTRA, synth_signal
from src.preprocess.pipeline import clean_channel_name, epoch_and_reject, preprocess_raw
from src.utils.config import load_config

CFG = load_config("preprocess")


def raw_from(chs, sfreq, dur=60, extra=(), seed=0):
    x = synth_signal(list(chs), sfreq, dur, 0, np.random.default_rng(seed))  # microvolts
    names = list(chs) + list(extra)
    data = np.vstack([x, np.random.default_rng(1).standard_normal((len(extra), x.shape[1])) * 10]) * 1e-6  # volts
    return mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose="ERROR")


def test_tdbrain_shape_order_units_sfreq():
    r = preprocess_raw(raw_from(TDBRAIN_EEG, 500.0, extra=TDBRAIN_EXTRA), CFG)
    assert r.epochs.shape[1:] == (19, 1024)
    assert r.epochs.dtype == np.float32
    assert r.ch_names == CFG["channels"]
    assert r.sfreq == 256
    assert not np.isnan(r.epochs).any()
    sd = r.epochs.std()
    assert 1.0 < sd < 200.0, f"expected microvolt scale, got sd={sd}"  # V would be ~1e-5
    assert len(r.epochs) <= CFG["max_epochs_per_person"]


def test_mumtaz_style_names_and_T3_rename():
    names = [f"EEG {c}-LE" for c in MUMTAZ_EEG]
    assert clean_channel_name("EEG T3-LE") == "T3"
    assert clean_channel_name("FP1") == "Fp1"
    r = preprocess_raw(raw_from(names, 256.0, extra=["EEG A2-A1", "EEG 23A-23R"]), CFG)
    assert r.ch_names == CFG["channels"]
    assert "T7" in r.ch_names and "T3" not in r.ch_names
    assert r.epochs.shape[1:] == (19, 1024)


def test_missing_channel_is_not_interpolated():
    chs = [c for c in TDBRAIN_EEG if c != "Pz"]
    with pytest.raises(ValueError, match="interpolation of missing channels is not allowed"):
        preprocess_raw(raw_from(chs, 500.0), CFG)


def test_rejection_and_cap():
    x = np.random.default_rng(0).standard_normal((19, 256 * 60)) * 10
    x[3, 256 * 10: 256 * 11] += 500  # artefact in epoch 2
    x[:, 256 * 20: 256 * 24] = 0.0     # flat epoch 5
    ep, rej, reasons, kept = epoch_and_reject(x, 256.0, CFG)
    assert ep.shape == (15, 19, 1024)
    assert rej[2] and reasons[2] == "ptp"
    assert rej[5] and reasons[5] == "flat"
    assert kept.sum() == min(CFG["max_epochs_per_person"], (~rej).sum())


def test_cache_from_synthetic_bids(synthetic_env):
    from src.preprocess.pipeline import load_cache

    X, idx, meta = load_cache("tdbrain", "restEC")
    assert X.shape[1:] == (19, 1024) and meta["sfreq"] == 256
    assert meta["channels"] == CFG["channels"]
    assert idx["kept"].sum() == X.shape[0]
