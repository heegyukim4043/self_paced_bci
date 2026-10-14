from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    native_sfreq: float = 1000.0
    target_sfreq: float = 250.0
    window_seconds: float = 2.0
    feature_step_seconds: float = 0.1
    decision_step_seconds: float = 0.5
    mi_start_seconds: float = 0.5
    mi_stop_seconds: float = 3.5
    fixation_start_seconds: float = -3.0
    fixation_stop_seconds: float = 0.0
    task_stop_seconds: float = 4.0
    iti_guard_seconds: float = 0.5
    max_task_iti_seconds: float = 6.0
    filter_order: int = 4
    transient_seconds: float = 2.0
    epsilon: float = 1e-12
    primary_channels: tuple[str, ...] = ("C3", "Cz", "C4")
    auxiliary_channels: tuple[str, ...] = ("FC3", "FC4", "CP3", "CP4")
    gate_channels: tuple[str, ...] = ("C3", "C4")
    sensorimotor_channels: tuple[str, ...] = (
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    )
    bands: tuple[tuple[str, float, float], ...] = (
        ("mu", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    )

    @property
    def selected_channels(self) -> tuple[str, ...]:
        return self.primary_channels + self.auxiliary_channels
