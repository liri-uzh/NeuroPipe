"""NeuroPipe settings — the one file to edit when adapting this tutorial to your own data.

Every researcher- and setup-specific choice the notebooks make is a field below, with a
comment on what it controls. Change the values, not the field names.

The settings are grouped, in the order they come up in the pipeline: where the files are
(Paths), what the electrodes are (Channels), what part of the recording to keep (Crop,
Resample, PowerLine), how artifacts are found (Ica), and what the analyses do with the
result (FilterBands, ConditionLabeling, Epochs, Aep, Tfr).
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CONFIG_PATH", "Config", "ConfigError", "config"]

# This file, which `1_prepare_raw_EEG.py` rewrites when a recording is picked.
CONFIG_PATH = Path(__file__).resolve()


def default(value):
    """Default value for a list or dict setting.

    Python insists these are built fresh per instance rather than shared, so they cannot
    be written as a plain `= [...]`. Only the wrapper is required; edit what is inside it.
    """
    return field(default_factory=lambda: deepcopy(value))


@dataclass(frozen=True)
class Paths:
    """Where the pipeline reads from and writes to.

    Folders are relative to the repository root unless given as full paths, and are
    created by the notebooks as needed.
    """

    # The recording to process. Both of these are rewritten by 1_prepare_raw_EEG.py
    # whenever a recording is picked in its file browser, which is how the notebooks that
    # follow know which participant to continue with.
    data_folder: Path = Path("data")
    raw_filename: Path = Path("p34_500.bdf")

    # Experiment log that accompanies the recording, read from data_folder
    log_filename: Path = Path("p34-matrix_senteces_order.txt")

    # Folders the notebooks write to
    cleaned_data_folder: Path = Path("cleaned_data")
    analyses_folder: Path = Path("analyses")
    logs_folder: Path = Path("logs")
    reports_folder: Path = Path("reports")
    figures_folder: Path = Path("figures")

    @property
    def raw_path(self) -> Path:
        """Full path of the recording."""
        return self.data_folder / self.raw_filename

    @property
    def log_path(self) -> Path:
        """Full path of the experiment log file that accompanies the recording."""
        return self.data_folder / self.log_filename


@dataclass(frozen=True)
class Channels:
    """Which electrode is what, and which montage describes their positions."""

    montage: str = "biosemi32"
    eog: list[str] = default(["EXG1", "EXG2"])
    reference: list[str] = default(["EXG3", "EXG4"])

    # Channels that were unused during recording. Usually known upfront from your system's
    # wiring/cap documentation. If that documentation is missing, you can still identify
    # them before analysis: unused channels typically show a dead/flat or pure-noise
    # signal when plotted, unlike real EEG channels.
    exclude: list[str] = default(["EXG5", "EXG6", "EXG7", "EXG8"])

    stim_channel: str = "Status"


@dataclass(frozen=True)
class Crop:
    """Segment of the recording to keep, in seconds.

    Set both to None if your file contains only one paradigm. If you don't know the exact
    timestamps, use mne.find_events() to locate the trigger boundaries of your paradigm of
    interest instead of hardcoding seconds here.
    """

    tmin: float | None = 2360
    tmax: float | None = None

    @property
    def is_set(self) -> bool:
        """Whether cropping was asked for at all."""
        return self.tmin is not None or self.tmax is not None


@dataclass(frozen=True)
class Resample:
    """Sampling rate the data is brought down to before cleaning, in Hz."""

    cleaning_hz: int = 500


@dataclass(frozen=True)
class PowerLine:
    """Power-line noise to notch out, which depends on where the data was recorded.

    The frequency is continent-dependent, so it is kept as a lookup table: set `region` to
    match where the data was recorded and the right frequencies are picked automatically.
    """

    frequencies_by_region: dict[str, list[float]] = default(
        {
            "EU": [50, 100, 150],
            "NA": [60, 120, 180],
        }
    )


@dataclass(frozen=True)
class Ica:
    """How the ICA is fitted, and how eagerly components are flagged as artifacts.

    eog_threshold and muscle_threshold both control how many components get flagged, and
    for both, HIGHER = STRICTER = FEWER components flagged. eog_threshold is a z-score on
    the correlation with the EOG channels (MNE default 3.0). muscle_threshold scores each
    component against a typical muscle topography and spectrum (MNE default 0.5, which is
    fairly permissive on 32-channel EEG - raise it toward 0.8-1.0 if too many components
    are being flagged as muscle).
    """

    method: str = "infomax"
    extended: bool = True
    random_state: int = 97
    filter_l_freq: float = 1
    filter_h_freq: float = 100
    eog_threshold: float = 3.0
    muscle_threshold: float = 0.5


@dataclass(frozen=True)
class FilterBands:
    """Named filter bands, and which of them this run uses.

    Which band to keep depends on the analysis you're running, not just the dataset. Add
    your own named option and point `active` at it instead of editing numbers inline. Keep
    the high-pass edge at 0.5 Hz or below unless you have a specific reason to go higher -
    increasing it also removes very low-frequency drift, but sweat artifacts already only
    produce drift on the order of 3+ seconds, so a higher high-pass buys little and risks
    distorting slow ERP components.
    """

    active: str = "auditory_erp"
    options: dict[str, list[float]] = default(
        {
            "auditory_erp": [0.5, 30],
            "alpha": [8, 13],
            "beta": [13, 30],
            "theta": [4, 8],
        }
    )

    @property
    def band(self) -> tuple[float, float]:
        """The `(l_freq, h_freq)` pair of the option named by `active`."""
        try:
            l_freq, h_freq = self.options[self.active]
        except KeyError:
            known = ", ".join(self.options) or "(none)"
            raise ConfigError(
                f"filter_bands.active is '{self.active}', which is not one of the "
                f"options in config.py (known bands: {known}). Add it under 'options', "
                "or point 'active' at one of the bands listed."
            ) from None
        return l_freq, h_freq


@dataclass(frozen=True)
class ConditionLabeling:
    """Rule that turns a column of the experiment log into a condition label.

    Rows whose `column` value starts with `keyword` get `keyword_label`; everything else
    gets `default_label`. Adjust to match how your own stimulus filenames or log columns
    encode condition.
    """

    column: str = "file"
    keyword: str = "random"
    keyword_label: str = "random"
    default_label: str = "context"


@dataclass(frozen=True)
class Epochs:
    """Trigger codes to epoch around, and the window cut around each, in seconds."""

    # event_id must match your own trigger codes; add more entries if you have multiple
    # event types. Usually known upfront from your experiment/trigger documentation. If
    # it's missing or unclear, run mne.find_events(raw) first (no event_id filter) - it
    # lists every trigger code actually present in the recording, and you can identify the
    # one(s) you need from there.
    event_id: dict[str, int] = default({"sentence onset": 256})

    # tmax should cover the longest stimulus/response window you need across all analyses
    # - individual analyses can crop back down (see Aep below).
    tmin: float = -0.2
    tmax: float = 7.0
    baseline: tuple[float, float] = (-0.2, 0)


@dataclass(frozen=True)
class Aep:
    """Window the auditory evoked potential is looked at, cut from the epochs above.

    It must fall inside Epochs.tmin/tmax - the epoch defines what data exists at all, this
    crop just narrows it for the AEP, where P1/N1/P2 all land within ~600 ms. Topomap time
    points are chosen automatically from the detected peaks, so there is no times setting.
    """

    crop_tmin: float = -0.2
    crop_tmax: float = 0.6


@dataclass(frozen=True)
class Tfr:
    """Frequencies and wavelet width for the time-frequency analysis."""

    freq_min: float = 1
    freq_max: float = 30
    n_freqs: int = 30
    n_cycles: int = 3


# ---------------------------------------------------------------------------------
# Machinery below: don't touch
# ---------------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a setting above points at something that isn't there."""


@dataclass(frozen=True)
class Config:
    """Every setting, one attribute per section."""

    eeg_system: str = "biosemi"

    paths: Paths = field(default_factory=Paths)
    channels: Channels = field(default_factory=Channels)
    crop: Crop = field(default_factory=Crop)
    resample: Resample = field(default_factory=Resample)
    power_line: PowerLine = field(default_factory=PowerLine)
    ica: Ica = field(default_factory=Ica)
    filter_bands: FilterBands = field(default_factory=FilterBands)
    condition_labeling: ConditionLabeling = field(default_factory=ConditionLabeling)
    epochs: Epochs = field(default_factory=Epochs)
    aep: Aep = field(default_factory=Aep)
    tfr: Tfr = field(default_factory=Tfr)


config = Config()


if __name__ == "__main__":
    print(config)
