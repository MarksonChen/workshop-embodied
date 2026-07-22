"""Small, explicit contracts for the Demo J spiking controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CONTROL_DT_MS = 20.0
FEATURE_DIM = 60
ACTION_DIM = 10
REFERENCE_FRAMES = 5
INTENTION_DIM = 32


@dataclass(frozen=True)
class SNNConfig:
    """Dynamics shared by training, rollout, and exported spike recordings."""

    neurons: int = 256
    adaptive_neurons: int = 128
    substeps: int = 4
    tau_membrane_ms: float = 20.0
    tau_synapse_ms: float = 10.0
    tau_adaptation_ms: float = 500.0
    tau_readout_ms: float = 20.0
    adaptation_strength: float = 1.6
    threshold: float = 1.0
    surrogate_alpha: float = 10.0
    recurrent_scale: float = 0.5

    @property
    def step_ms(self) -> float:
        return CONTROL_DT_MS / self.substeps

    def validate(self) -> None:
        if self.neurons <= 0:
            raise ValueError("neurons must be positive")
        if not 0 <= self.adaptive_neurons <= self.neurons:
            raise ValueError("adaptive_neurons must lie in [0, neurons]")
        if self.substeps <= 0 or CONTROL_DT_MS % self.substeps:
            raise ValueError("substeps must divide the 20 ms control period")
        for name in (
            "tau_membrane_ms",
            "tau_synapse_ms",
            "tau_adaptation_ms",
            "tau_readout_ms",
            "threshold",
            "surrogate_alpha",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict[str, int | float]:
        self.validate()
        return asdict(self)
