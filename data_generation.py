"""Synthetic CSI-like amplitude generator. This does not produce hardware CSI."""
from __future__ import annotations
import numpy as np
from config import CLASS_NAMES, N_GROUPS, N_SUBCARRIERS, N_TIMESTEPS, SAMPLE_RATE_HZ

def _make_scene(rng: np.random.Generator):
    """Return shared static multipath profile for one scene."""
    sub = np.arange(N_SUBCARRIERS, dtype=np.float32)
    gains = rng.uniform(0.25, 1.0, size=4)
    delays = rng.uniform(0.0, 2*np.pi, size=4)
    slopes = rng.uniform(-0.18, 0.18, size=4)
    profile = np.zeros(N_SUBCARRIERS, dtype=np.complex64)
    for g, d, s in zip(gains, delays, slopes):
        profile += (g * np.exp(1j * (d + s * sub))).astype(np.complex64)
    profile /= (np.sqrt(np.mean(np.abs(profile)**2)) + 1e-8)
    return profile

def _channel_window(rng, profile, kind: str):
    t = np.arange(N_TIMESTEPS, dtype=np.float32) / SAMPLE_RATE_HZ
    sub = np.arange(N_SUBCARRIERS, dtype=np.float32)[:, None]
    static = profile[:, None] * np.ones((1, N_TIMESTEPS), dtype=np.complex64)
    noise = (rng.normal(0, 0.025, (N_SUBCARRIERS, N_TIMESTEPS))
             + 1j*rng.normal(0, 0.025, (N_SUBCARRIERS, N_TIMESTEPS))).astype(np.complex64)
    h = static.copy()

    if kind == "Benign":
        freq = rng.uniform(0.3, 1.2)
        displacement = rng.uniform(0.05, 0.25) * np.sin(2*np.pi*freq*t + rng.uniform(0, 2*np.pi))
        phase = 0.5 * displacement[None, :] * (sub + 1.0)
        moving_path = 0.30 * np.exp(1j * (phase + rng.uniform(0, 2*np.pi)))
        h += moving_path.astype(np.complex64)
    elif kind == "Intrusion":
        speed = rng.uniform(0.6, 2.2)
        center = rng.uniform(0.35, 1.65)
        width = rng.uniform(0.08, 0.24)
        envelope = np.exp(-0.5*((t-center)/width)**2)
        phase = (2*np.pi*speed*t[None, :] * (sub + 1.0) / 30.0)
        crossing = (0.85 * envelope[None, :] * np.exp(1j*phase))
        attenuation = 1.0 - 0.48*envelope[None, :]
        h = h * attenuation + crossing.astype(np.complex64)
    # Empty stays static apart from receiver noise.
    h += noise
    return np.abs(h).astype(np.float32)

def generate_dataset(seed: int, n_groups: int = N_GROUPS):
    """
    Produce one sample per class per scene group.
    Returns:
      X: float32 [n_groups*4, 30, 200] amplitude arrays
      y: int64 class IDs
      groups: int64 scene group IDs
    Replay is derived from an Empty or Benign source in the same group.
    """
    rng = np.random.default_rng(seed)
    xs, ys, gids = [], [], []
    for gid in range(n_groups):
        scene = _make_scene(rng)
        empty = _channel_window(rng, scene, "Empty")
        benign = _channel_window(rng, scene, "Benign")
        intrusion = _channel_window(rng, scene, "Intrusion")
        source = empty if rng.random() < 0.5 else benign
        seg_len = int(rng.integers(20, 61))  # 0.2–0.6 s at 100 Hz
        start = int(rng.integers(0, N_TIMESTEPS - seg_len + 1))
        segment = source[:, start:start+seg_len]
        repeats = int(np.ceil(N_TIMESTEPS / seg_len))
        replay = np.tile(segment, (1, repeats))[:, :N_TIMESTEPS].copy()
        replay += rng.normal(0, 0.012, replay.shape).astype(np.float32)

        examples = [empty, benign, intrusion, replay]
        for label, x in enumerate(examples):
            xs.append(x)
            ys.append(label)
            gids.append(gid)
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    groups = np.asarray(gids, dtype=np.int64)
    # Assert one sample per class and scene; this is the grouping contract.
    for gid in range(n_groups):
        labels = y[groups == gid]
        assert sorted(labels.tolist()) == [0, 1, 2, 3]
    return X, y, groups

def extract_features(X: np.ndarray) -> np.ndarray:
    """Eight transparent engineered features per window."""
    eps = 1e-8
    feats = []
    for sample in X:
        mean = float(np.mean(sample))
        std = float(np.std(sample) + eps)
        z = (sample - mean) / std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3.0)
        temporal_diff = np.diff(sample, axis=1)
        doppler_proxy = float(np.mean(np.abs(temporal_diff)))
        max_excursion = float(np.max(np.abs(sample - np.mean(sample, axis=1, keepdims=True))))
        periodicity = []
        for row in sample:
            row = row - row.mean()
            denom = float(np.dot(row, row) + eps)
            lag = max(1, int(SAMPLE_RATE_HZ / 1.0))
            periodicity.append(float(np.dot(row[:-lag], row[lag:]) / denom))
        corr = np.corrcoef(sample)
        tri = corr[np.triu_indices_from(corr, k=1)]
        tri = tri[np.isfinite(tri)]
        mean_corr = float(np.mean(tri)) if tri.size else 0.0
        feats.append([
            mean, float(np.var(sample)), skew, kurtosis, mean_corr,
            doppler_proxy, float(np.mean(periodicity)), max_excursion
        ])
    return np.asarray(feats, dtype=np.float32)
