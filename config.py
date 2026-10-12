from dataclasses import dataclass

CLASS_NAMES = ["Empty", "Benign", "Intrusion", "Replay/Spoof"]
N_SUBCARRIERS = 30
N_TIMESTEPS = 200
SAMPLE_RATE_HZ = 100
N_GROUPS = 600

@dataclass
class ExperimentConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 7
    num_workers: int = 0
    noise_levels: tuple = (0.005, 0.03, 0.06, 0.12, 0.18, 0.25)
