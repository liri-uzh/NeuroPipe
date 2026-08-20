# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.24.0",
#     "mne==1.12.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load and prepare the raw EEG data of auditory experiment

    This notebook covers the first stage of the pipeline: reading the raw BioSemi file (a), cropping it to the paradigm of interest, setting the reference and resampling the data (b), and removing power line noise (c). Nothing beyond line noise is taken out of the signal here — bad channels, eye and muscle artifacts are dealt with in `2_ica_artifact_removal.ipynb`, which loads the file cached in the last cell of this notebook.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    All researcher- and setup-specific choices for this pipeline (which channels are EEG/EOG/reference, the EEG system, power-line frequency, filter bands, ICA settings, event codes, epoch windows, ...) live in **`config.py`** at the repository root, instead of being hardcoded throughout the notebook. This is the file you should edit when adapting this tutorial to your own recording — you should rarely need to change the code cells themselves.

    The recording itself is the exception: it is chosen with the file browser below, so it can live anywhere on the machine — an external drive, a shared network folder — rather than having to sit in one particular folder.

    Settings are grouped by topic and reached by name — `config.paths.data_folder`, `config.epochs.tmin` — so a misspelled setting is an error naming the field it belongs to, rather than something that surfaces halfway through a long run. Each group is a small dataclass at the top of `config.py`, and the value of every setting sits right next to its name there, with a comment on what it controls — so opening the file shows you what there is to change straight away.

    Open `config.py` next to this notebook to see every adjustable parameter and a short note on what it controls.
    """)


@app.cell
def _():
    from pathlib import Path

    import mne

    # Every researcher/setup-specific setting lives in config.py. Edit that file, not this cell, to adapt this tutorial to your own data.
    from config import config

    # Folder where the cleaned data will be stored
    cleaned_data_folder = config.paths.cleaned_data_folder
    cleaned_data_folder.mkdir(exist_ok=True)  # create the folder if it doesn't exist

    # Folder where figures will be saved
    figures_folder = config.paths.figures_folder
    figures_folder.mkdir(exist_ok=True)
    return Path, cleaned_data_folder, config, figures_folder, mne


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting the recording

    Pick the file to prepare in the browser below. Everything else about this participant follows from that one choice: the folder to read from, the participant name, and the prefix given to every file this notebook writes — there is no second setting that can fall out of sync with it.

    If nothing is selected, the notebook falls back to the recording named by `config.paths.data_folder` and `config.paths.raw_filename` — which is also where the choice is recorded, once the prepared data is saved in the last cell. The notebooks that follow read those two values, so picking a participant here points the whole pipeline at it, and an unattended run over a batch of participants still works without anyone clicking anything.
    """)


@app.cell
def _(Path, config, mo):
    # Open the browser in the folder named in config.py, falling back to the notebook's own folder
    # when it doesn't exist (data/ is gitignored, so a fresh clone of the repository has none yet).
    _start = config.paths.data_folder

    raw_browser = mo.ui.file_browser(
        initial_path=_start if _start.is_dir() else Path.cwd(),
        filetypes=[".bdf"],  # BioSemi recordings; widen this for other EEG systems
        selection_mode="file",
        multiple=False,
        label="Recording to prepare",
    )
    raw_browser
    return (raw_browser,)


@app.cell
def _(config, raw_browser):
    # Use the selected file, or the one named in config.py when the browser is still empty
    _selected = raw_browser.value
    raw_path = _selected[0].path if _selected else config.paths.raw_path

    data_folder = raw_path.parent  # folder the recording sits in
    participant_filename = raw_path.name

    # Filename without extension, used as a prefix for every file this notebook writes
    participant_stem = raw_path.stem
    return data_folder, participant_filename, participant_stem, raw_path


@app.cell(hide_code=True)
def _(Path):
    def remember_selection(raw_path):
        """Record this recording in config.py, so the next notebook opens the matching file.

        Only the two path values are rewritten, in place, which leaves the rest of the settings,
        their order and the comments in config.py exactly as they are. The folder is stored
        relative to the notebook when the recording sits inside the project, and as a full path
        when it lives somewhere else. Returns True when the file was changed, False when it
        already said this.
        """
        import re

        from config import CONFIG_PATH

        try:
            folder = raw_path.parent.relative_to(Path.cwd())
        except ValueError:
            folder = raw_path.parent  # recording lives outside the project folder

        original = CONFIG_PATH.read_text()
        updated = original

        for setting, new_value in [
            ("data_folder", str(folder)),
            ("raw_filename", raw_path.name),
        ]:
            # The field's default, e.g.   data_folder: Path = Path("data"). The leading \b keeps
            # 'data_folder' from matching 'cleaned_data_folder' as well.
            match = re.search(
                rf'(\b{setting}\s*:\s*Path\s*=\s*Path\()(?:"[^"]*"|\'[^\']*\')',
                updated,
            )
            if match is None:
                raise ValueError(
                    f"No '{setting}: Path = Path(...)' found in config.py."
                )

            quoted = '"' + new_value.replace("\\", "\\\\").replace('"', '\\"') + '"'
            updated = updated[: match.end(1)] + quoted + updated[match.end() :]

        if updated == original:
            return (
                False  # already points at this recording, so leave the file untouched
            )

        compile(
            updated, str(CONFIG_PATH), "exec"
        )  # refuse to write anything that would not parse

        tmp_path = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
        tmp_path.write_text(updated)
        tmp_path.replace(
            CONFIG_PATH
        )  # atomic: an interrupted write cannot truncate config.py
        return True

    return (remember_selection,)


@app.cell(hide_code=True)
def _(cleaned_data_folder, mo, participant_stem, raw_path):
    _output_path = cleaned_data_folder / f"{participant_stem}_prepared_raw.fif"

    mo.md(
        f"""
        Reading **`{raw_path}`**

        Participant **`{participant_stem}`** — the prepared data will be written to
        `{_output_path}`, and the log and figures are
        named after the participant in the same way.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logging and saved figures

    Alongside the config file, this notebook writes a plain-text log (via Python's `logging` module) to `config.paths.logs_folder`. This is separate from the notebook's own cell output: a log file gives you a permanent, greppable record of exactly what happened on a given run — which channels were excluded, how many events were found, which ICA components were removed, ... — which matters for provenance, and becomes essential once you're not watching the notebook run interactively (e.g. running it non-interactively via `jupyter nbconvert --execute` across many participants).

    The log file is opened in **append** mode, so re-running the notebook adds to the existing log rather than overwriting it. Each run starts with a timestamped `=== Starting ... ===` line, so you can always tell where one run ends and the next begins.

    The cell below also defines a small `save_fig()` helper. Every figure worth keeping is written to `config.paths.figures_folder` as a PNG named after the participant, so the QC plots survive after the notebook is closed. Two details it handles for you:
    - Some MNE functions (`plot_components` with many components, `plot_properties`) return a **list** of figures rather than one; each gets its own numbered file.
    - **Interactive browsers cannot be saved.** `raw.plot()`, `epochs.plot()`, and `ica.plot_sources()` return live Qt browser windows, not matplotlib figures — they have no `savefig`. Those are for on-screen inspection only; the `mne.Report` at the end of the notebook covers the raw data separately.
    """)


@app.cell
def _(config, figures_folder, participant_filename, participant_stem):
    import logging

    logs_folder = config.paths.logs_folder
    logs_folder.mkdir(exist_ok=True)

    logger = logging.getLogger(f"neuropipe.prepare.{participant_filename}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()  # avoid duplicate handlers if this cell is re-run

    log_path = logs_folder / f"{participant_stem}_preparation.log"
    file_handler = logging.FileHandler(
        log_path, mode="a"
    )  # append, so earlier runs are never overwritten
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"=== Starting preparation pipeline for {participant_filename} ===")

    def save_fig(fig, name, dpi=300):
        """Save a figure into the figures folder, prefixed with the participant name.

        Some MNE plotting functions return a *list* of figures rather than one
        (e.g. plot_components pages its output when there are many components, and
        plot_properties returns one figure per component), so both cases are handled.
        Interactive browsers such as raw.plot() cannot be saved this way.
        """
        figs = fig if isinstance(fig, (list, tuple)) else [fig]
        paths = []
        for i, f in enumerate(figs):
            suffix = f"_{i + 1}" if len(figs) > 1 else ""
            path = figures_folder / f"{participant_stem}_{name}{suffix}.png"
            f.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        logger.info(f"Saved figure '{name}' to {', '.join(str(p) for p in paths)}")
        return paths

    return logger, save_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (a) Reading EEG files using MNE-Python

    ### Read your raw file

    This notebook is written for **BioSemi** recordings, which are stored as `.bdf` files and read with `mne.io.read_raw_bdf`. MNE provides readers for many other EEG formats too, which you can explore in the [MNE documentation](https://mne.tools/stable/api/reading_raw_data.html).

    EEG files can easily be several GB, depending on sampling rate, duration, and channel count — the cell below prints the actual size of your file. Above a couple of GB, `preload=False` is set (as done here) to load the data in chunks instead of all at once; for smaller files, `preload=True` is simpler and loads everything into memory immediately. Additionally, the channels listed in `config.channels.exclude` are excluded because they were unused during recording (here, `EXG5`–`EXG8`). While these channels are present in the data, they contain no meaningful information — check your own system's channel list and update this entry in `config.py` accordingly.

    After loading the file, `raw.info` is printed. Key details include:
    - **Sampling frequency**: 16.384 kHz (set high for subcortical analyses, but not required for this tutorial).
    - **Filters**: Frequencies in raw data range from 0 to 3334 Hz.
    - **Channels**: 36 total:
      - 32 are EEG channels.
      - 4 are external channels (Electrooculogram (EOG) and mastoids).
      - 1 additional channel is labeled as Stimulus.

    Note: the external channels will be re-labeled and the appropriate montage assigned in the next step to ensure accurate channel definitions.

    Always have aproper look at the `raw.info` to ensure everything is correct.
    """)


@app.cell
def _(config, data_folder, logger, mne, participant_filename):
    # Print the actual size of the file we're about to read (see markdown above)
    file_size_gb = (data_folder / participant_filename).stat().st_size / 1e9
    print(f"File size: {file_size_gb:.2f} GB")

    # Read raw data
    raw = mne.io.read_raw_bdf(
        data_folder / participant_filename,
        preload=False,
        exclude=config.channels.exclude,
    )

    # Print duration of the recording
    print(f"Duration of the recording: {raw.times[-1]:.2f} seconds")
    logger.info(
        f"Loaded {participant_filename} ({file_size_gb:.2f} GB): {raw.times[-1]:.2f} s, {len(raw.ch_names)} channels"
    )
    return (raw,)


@app.cell
def _(raw):
    # Check info about the raw data
    raw.info


@app.cell
def _(config, logger, mne, raw):
    # Set the EOG channels named in config.py
    raw.set_channel_types(dict.fromkeys(config.channels.eog, "eog"))

    # Set the reference channels named in config.py---these do not directly exist as a channel type, so they are set as 'misc' (miscellaneous)
    raw.set_channel_types(dict.fromkeys(config.channels.reference, "misc"))

    # Now that we have the EEG channels properly set according to the cap used, set the montage named in config.py
    montage = mne.channels.make_standard_montage(config.channels.montage)
    raw.set_montage(montage)

    logger.info(
        f"Set channel types (eog={config.channels.eog}, reference={config.channels.reference}) and montage '{config.channels.montage}'"
    )

    # Check raw.info again
    raw.info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (optional) Cut out segment of interest from `.bdf` file

    This step is specific to this dataset because multiple paradigms were stored in a single `.bdf` file. For better data organization, it is recommended to record each paradigm in a separate file.

    Here, the file is cropped at the timestamp given by `config.crop.tmin` (2360 seconds for this participant), which marks the start of the random vs. context sentences paradigm. This ensures that only the relevant segment of data is retained for analysis. If your recording contains only one paradigm, set both `config.crop.tmin` and `config.crop.tmax` to `null` and this cell becomes a no-op.

    **If you don't know the exact timestamps** of your paradigm of interest (e.g. you didn't note them during recording), don't guess — use `mne.find_events(raw)` to look at the trigger codes on the Status/stim channel instead. Paradigm boundaries are usually marked by a distinctive trigger (e.g. a block-start code), and you can read off its timestamp from the events array and use that as `tmin`/`tmax` instead of a hardcoded guess.

    This is a bit unusual, but MNE works with seconds as standard unit of time. Always keep this in mind.
    """)


@app.cell
def _(config, logger, raw):
    # Crop to the segment of interest, as configured in config.py (tmax=None crops until the end)
    if config.crop.is_set:
        raw.crop(tmin=config.crop.tmin, tmax=config.crop.tmax)

    # Check the new duration of the recording
    print(f"New duration of the recording: {round(raw.times[-1], 2)} seconds")
    logger.info(f"Cropped to {config.crop}; new duration {round(raw.times[-1], 2)} s")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (b) Setting correct reference and resample data

    Now that everything is set up, cleaning of the EEG data can begin. Since running ICA at a high sampling rate would take a long time, the data is first **downsampled** using the `resample` method, to the rate given by `config.resample.cleaning_hz`. This step might not be necessary if your data was recorded at a lower sampling rate.

    ### Downsampling

    To modify the data, it is now loaded into memory. If your file is smaller, you can preload it from the beginning. At this stage, the reference is also set to the channels named in `config.channels.reference` (the two mastoid channels, for this dataset). While this step could be performed earlier (e.g., when setting the montage), it requires the data to be loaded into memory, so it is done here.

    **Note on referencing**: the BioSemi ActiveTwo system has its own internal reference/ground circuit and does not strictly require physical reference electrodes like mastoids — re-referencing to mastoids here is a *methodological choice* for this study, not a hardware requirement. If your system or paradigm calls for a different reference (e.g. average reference, or a single reference electrode), change `config.channels.reference`, or pass `'average'` to `set_eeg_reference` instead of a channel list.

    After resampling, `raw.info` shows:
    - **Sampling frequency** is now `config.resample.cleaning_hz` Hz (500 Hz here).
    - A **low-pass filter** at half that rate has been automatically applied by MNE. This anti-aliasing filter ensures no aliasing artifacts by filtering at half the target sampling rate.

    These steps prepare the data for efficient processing and further cleaning.
    """)


@app.cell
def _(config, logger, raw):
    # Resample data (this takes a few seconds to run)
    raw.load_data()  # load data into memory now since we will be modifying it
    raw.set_eeg_reference(
        config.channels.reference
    )  # set average of reference channels
    raw.resample(config.resample.cleaning_hz)

    logger.info(
        f"Re-referenced to {config.channels.reference} and resampled to {config.resample.cleaning_hz} Hz"
    )

    # Check the new sampling frequency
    raw.info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (c) Remove power line noise using notch filters

    In this step, a **notch filter** is applied to remove power line noise and its harmonics. Power line noise is a common source of interference in EEG recordings, caused by electrical systems operating at 50 Hz in most of the world (including the EU) or 60 Hz in North America. Filtering these frequencies helps clean the data without affecting the neural signals of interest.

    Because the power-line frequency is **continent-dependent**, `config.py` stores it as a lookup table, `config.power_line.frequencies_by_region`, keyed by region (`"EU"`: 50/100/150 Hz, `"NA"`: 60/120/180 Hz). Set `config.power_line.region` to match where the data was recorded and the correct frequencies are picked automatically — add more regions to the table if needed.
    """)


@app.cell
def _(raw):
    # '%matplotlib qt' command supported automatically in marimo
    # Switching to the interactive Qt backend here, right before the first plot — this stays in effect
    # for the rest of the notebook, so it only needs to be set once.

    raw.compute_psd().plot()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The PSD should look clean, displaying:
    - The expected **EEG shoulder**,  a noticeable increase in power is observed around 4–10 Hz, reflecting theta and alpha activity.
    - A general **1/f shape**, where power decreases with increasing frequency.
    - Evidence of notch filters applied at specific frequencies (e.g., 50 Hz and harmonics), as power line noise artifacts are no longer visible.
    """)


@app.cell
def _(config, logger, raw):
    # Apply notch filter at the power-line frequency and harmonics for the configured region
    notch_freqs = config.power_line.frequencies
    raw.notch_filter(notch_freqs)
    logger.info(
        f"Applied notch filter at {notch_freqs} Hz (region={config.power_line.region})"
    )


@app.cell
def _(raw, save_fig):
    fig_psd = raw.compute_psd().plot()
    save_fig(fig_psd, "psd")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Caching the prepared data for the next notebook

    At this point the data has been read, cropped, re-referenced, resampled and freed of power line noise, but no artifacts have been removed from it yet. It is cached here in MNE's native `.fif` format so that `2_ica_artifact_removal.ipynb` can begin directly at the ICA step: reading and resampling a multi-GB `.bdf` takes a while, and the ICA notebook is typically run more than once while tuning the thresholds in `config.ica`. The full measurement info travels with the file — channel types, montage, crop, reference, sampling frequency and the notch filter — so none of the setup above has to be repeated.

    Saving also records the participant in `config.py`, rewriting `paths.data_folder` and `paths.raw_filename` in place — the two values the next notebook reads to find this file. Nothing else in the file is touched, and re-running with the same recording selected leaves it alone entirely.

    Two files end up in `config.paths.cleaned_data_folder`, and it is worth keeping them apart:
    - **`*_prepared_raw.fif`** — written here, the input to the ICA notebook.
    - **`*_cleaned_raw.fif`** — written at the end of the ICA notebook, the fully cleaned data the analysis notebook loads.
    """)


@app.cell
def _(
    cleaned_data_folder,
    logger,
    participant_stem,
    raw,
    raw_path,
    remember_selection,
):
    # Save the prepared data for the next notebook
    prepared_path = cleaned_data_folder / f"{participant_stem}_prepared_raw.fif"
    raw.save(prepared_path, overwrite=True)
    logger.info(f"Saved prepared data to {prepared_path}")

    # Point config.py at this recording so the next notebook opens the matching file
    if remember_selection(raw_path):
        logger.info(f"Recorded {raw_path} in config.py")


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
