# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.24.0",
#     "mne[full]==1.12.1",
#     "mne-icalabel==0.9.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.function(hide_code=True)
def use_qt_browser(interval=0.02):
    """Select MNE's Qt browser and keep its windows responsive inside marimo.

    marimo cells run inside the kernel's asyncio event loop, and Qt windows are
    only interactive while a Qt event loop is running. Nothing runs one here, so
    this schedules a background task that repeatedly gives Qt a slice of time to
    process its pending events - the same trick IPython's `%gui qt` uses. Other
    cells keep working while the browser window is open.

    Safe to call repeatedly: at most one pump runs per kernel session. Returns the
    pump task, or None when the notebook is executed as a plain script (no asyncio
    loop) - there, pass `block=True` to `raw.plot()` instead.
    """
    import asyncio

    import mne
    from qtpy.QtWidgets import QApplication

    mne.viz.set_browser_backend('qt')

    async def _pump():
        while True:
            qt_app = QApplication.instance()
            if qt_app is not None:  # created by MNE on the first plot
                qt_app.processEvents()
            await asyncio.sleep(interval)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None  # running as `python 2_ica_artifact_removal.py`, not in marimo

    # Stash the task on the loop, which outlives any single cell run, so re-running
    # a plotting cell does not leave a second pump behind.
    task = getattr(loop, "_mne_qt_pump", None)
    if task is None or task.done():
        task = loop.create_task(_pump())
        loop._mne_qt_pump = task
    return task


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Remove artifacts from the prepared EEG data using ICA

    This notebook covers the second stage of the pipeline, starting from the data prepared in `1_prepare_raw_EEG.ipynb`: marking bad channels (d), removing eye and muscle artifacts with ICA (e), interpolating the bad channels (f), filtering the data to the frequency band of interest (g), caching the cleaned data for further analyses (h), and generating a cleaning report (i).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    All researcher- and setup-specific choices for this pipeline (which channels are EEG/EOG/reference, the EEG system, power-line frequency, filter bands, ICA settings, event codes, epoch windows, ...) live in **`config.json`** at the repository root, instead of being hardcoded throughout the notebook. This is the file you should edit when adapting this tutorial to your own recording — you should rarely need to change the code cells themselves. The sections this notebook reads most are `ica`, `channels` and `filter_bands`.

    Which participant to clean is not one of them: that is chosen with the file browser below.

    Open `config.json` next to this notebook to see every adjustable parameter and a short note on what it controls.
    """)
    return


@app.cell
def _():
    import json
    import mne
    from mne_icalabel import label_components
    from pathlib import Path

    # Load researcher/setup-specific settings. Edit config.json, not this cell, to adapt this tutorial to your own data.
    try:
        with open("config.json") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"config.json has invalid JSON syntax: {e}\n"
            "Common causes: a trailing comma after the last item in a list/object, "
            "single quotes instead of double quotes, or Python's None/True/False "
            "instead of JSON's null/true/false. Open config.json in a text editor "
            "and check the line/column mentioned above."
        ) from e

    # Folder where the prepared data is read from and the cleaned data will be stored
    cleaned_data_folder = Path(config["paths"]["cleaned_data_folder"])
    cleaned_data_folder.mkdir(exist_ok=True)  # create the folder if it doesn't exist

    # Folder where figures will be saved
    figures_folder = Path(config["paths"]["figures_folder"])
    figures_folder.mkdir(exist_ok=True)
    return (
        Path,
        cleaned_data_folder,
        config,
        figures_folder,
        label_components,
        mne,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The recording this notebook cleans

    The participant is read from `config['paths']['raw_filename']`, which follows the recording selected in `1_prepare_raw_EEG.py`. The matching `*_prepared_raw.fif` in `config['paths']['cleaned_data_folder']` is opened below, and the participant name is the prefix for every file this notebook writes.
    """)
    return


@app.cell
def _(Path, cleaned_data_folder, config):
    # Participant to clean, and the prefix for every file this notebook writes
    participant_stem = Path(config['paths']['raw_filename']).stem

    prepared_path = cleaned_data_folder / f"{participant_stem}_prepared_raw.fif"
    participant_filename = prepared_path.name
    return participant_filename, participant_stem, prepared_path


@app.cell(hide_code=True)
def _(cleaned_data_folder, mo, participant_stem, prepared_path):
    _output_path = cleaned_data_folder / f"{participant_stem}_cleaned_raw.fif"

    mo.md(
        f"""
        Cleaning **`{prepared_path}`**

        Participant **`{participant_stem}`** — the cleaned data will be written to
        `{_output_path}`, and the log, figures and
        cleaning report are named after the participant in the same way.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logging and saved figures

    Alongside the config file, this notebook writes a plain-text log (via Python's `logging` module) to `config['paths']['logs_folder']`. This is separate from the notebook's own cell output: a log file gives you a permanent, greppable record of exactly what happened on a given run — which channels were marked bad, how many blinks were detected, which ICA components were removed, ... — which matters for provenance, and becomes essential once you're not watching the notebook run interactively (e.g. running it non-interactively via `jupyter nbconvert --execute` across many participants).

    The log file is opened in **append** mode, so re-running the notebook adds to the existing log rather than overwriting it. Each run starts with a timestamped `=== Starting ... ===` line, so you can always tell where one run ends and the next begins.

    The cell below also defines a small `save_fig()` helper. Every figure worth keeping is written to `config['paths']['figures_folder']` as a PNG named after the participant, so the QC plots survive after the notebook is closed. Two details it handles for you:
    - Some MNE functions (`plot_components` with many components, `plot_properties`) return a **list** of figures rather than one; each gets its own numbered file.
    - **Interactive browsers cannot be saved.** `raw.plot()`, `epochs.plot()`, and `ica.plot_sources()` return live Qt browser windows, not matplotlib figures — they have no `savefig`. Those are for on-screen inspection only; the `mne.Report` at the end of the notebook covers the raw data separately.
    """)
    return


@app.cell
def _(Path, config, figures_folder, participant_filename, participant_stem):
    import logging

    logs_folder = Path(config["paths"]["logs_folder"])
    logs_folder.mkdir(exist_ok=True)

    logger = logging.getLogger(f"neuropipe.clean.{participant_filename}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()  # avoid duplicate handlers if this cell is re-run

    log_path = logs_folder / f"{participant_stem}_cleaning.log"
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

    logger.info(f"=== Starting cleaning pipeline for {participant_filename} ===")

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
    ## Loading the prepared data

    The cell below reads back the `.fif` file cached at the end of `1_prepare_raw_EEG.ipynb`. Run that notebook first for this participant — otherwise this cell stops with an error naming the file it expected to find.

    Because `.fif` stores the complete measurement info, everything configured in the previous notebook comes back with the data: the channel types (EEG/EOG/misc/stim), the montage, the sampling frequency, and a record of the filters already applied. `preload=True` loads the data into memory right away, since every step below (ICA, interpolation, filtering) modifies it. The file is already resampled to `config['resample']['cleaning_hz']` Hz, so it is far smaller than the original `.bdf` and reads in seconds.
    """)
    return


@app.cell
def _(logger, mne, prepared_path):
    # Load the prepared data: cropped, re-referenced, resampled and notch-filtered
    if not prepared_path.exists():
        raise FileNotFoundError(
            f"{prepared_path} not found. Run 1_prepare_raw_EEG.py to create it."
        )

    raw = mne.io.read_raw_fif(
        prepared_path, preload=True
    )  # preload: ICA and filtering modify the data

    logger.info(
        f"Loaded prepared data from {prepared_path}: {raw.times[-1]:.2f} s, "
        f"{len(raw.ch_names)} channels at {raw.info['sfreq']:.0f} Hz"
    )

    # Check info about the prepared data
    raw.info
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (d) Inspecting raw data

    The first step in cleaning the data is to visually inspect the recording and mark bad channels. This can involve reviewing notes taken during the recording session to identify bad electrode channels or directly examining the raw data for irregularities.

    Steps for inspection:
    1. **Plot the raw time series** using `raw.plot()` to visually check for channels with excessive noise or artifacts, and mark bad channels (see below).
    2. **Inspect the power spectrum density (PSD)** to evaluate the frequency content of the signal and identify noise or unexpected patterns.

    **How to decide if a channel is "bad"** — this is a judgment call, and the criterion matters more than it might seem:
    - **Mark the channel bad** (then interpolate — step (f) below) if it is **consistently** noisy throughout the recording.
    - A channel that is otherwise clean but has an isolated **jump** at one point in time does **not** need to be removed — that's a localized event, not a channel-level problem.
    - If a channel is noisy **only in a handful of trials**, don't mark the whole channel bad — instead, **you can drop bad epochs** later (notebook 3).

    For this participant, the raw data looks good overall, with no major issues. However, for practice purposes, **CP5** is marked as bad, as it shows minor jumping around second 360. Per the guidance above this is actually borderline — a single jump wouldn't normally warrant removal — but it demonstrates the process of identifying and handling bad channels.

    These steps ensure the data is ready for further processing and cleaning.
    """)
    return


@app.cell
def _(raw):
    # Selects MNE's Qt browser and keeps its window responsive — marimo does not run a Qt event
    # loop of its own, so this is called once before every interactive plot in this notebook.
    use_qt_browser()

    # You can also screen through the raw data by plotting it.
    # Show every channel the object contains (EEG + EOG + misc + stim), rather than a hardcoded count.
    raw.plot(n_channels=len(raw.ch_names), duration=10, scalings=dict(eeg=100e-6), block=True)

    # Note: You can already exclude some channels from the raw data if you see that they are noisy by clicking on the channel name in the plot
    # They will then be stored in the raw.info['bads'] attribute
    return


@app.cell
def _(logger, raw):
    # Mark bad channels found during the visual inspection above (not config-driven - see note above)
    raw.info["bads"] = ["CP5"]
    print(f"Bad channels: {raw.info['bads']}")
    logger.info(f"Marked bad channels: {raw.info['bads']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (e) Removing artifacts using Independent Component Analysis (ICA) and automatic component labeling

    ### Initializing the ICA instance

    The parameters for initializing the ICA instance, read from `config['ica']`, are selected as follows:

    - **`method='infomax'`**: Specifies the ICA algorithm to use. The Infomax method is chosen for its effectiveness in separating sources, particularly in EEG data. Additionally, this is combined with the extended Infomax approach, which improves the algorithm's ability to separate sub-Gaussian and super-Gaussian sources.

    - **`fit_params=dict(extended=True)`**: Activates the extended Infomax feature, enhancing the method's capacity to handle diverse source distributions, such as those present in EEG data.

    - **`random_state=97`**: Sets a fixed random seed for reproducibility. This ensures that the results of the ICA decomposition remain consistent across runs.

    These parameters are a solid default for EEG data, and are a reasonable starting point — but they are not the only option (`picard` and `fastica` are common alternatives), so they're kept fully editable in `config['ica']`.

    **Note**: channels already marked bad (step (d) above) are automatically excluded from the ICA fit — `ica.fit()` only uses "good" data channels by default, so make sure bad channels are marked *before* this step, not after.

    ### Preparing data for ICA

    Before fitting the ICA, a **copy of the raw data** is created and high-pass/low-pass filtered between `config['ica']['filter_l_freq']` and `config['ica']['filter_h_freq']` Hz (1–100 Hz by default). This filtering range is selected to:

    - Remove slow drifts and low-frequency noise, such as DC offsets, which can interfere with ICA's ability to separate independent components.
    - Focus on the frequency range where most artifacts, such as eye blinks, muscle noise, and other physiological or environmental artifacts, are present.
    - Improve ICA's ability to separate signal from noise while preserving relevant brain activity.

    This range is an **advanced, editable** setting — the values above work well as a general default, but if you know your artifacts of interest live outside 1–100 Hz you can adjust `config['ica']['filter_l_freq']`/`['filter_h_freq']`.

    This preparation step ensures that ICA operates on data optimized for identifying and removing artifacts while leaving the original raw data untouched for subsequent analyses.
    """)
    return


@app.cell
def _(config, logger, mne, raw):
    # Initialize ICA instance
    ica = mne.preprocessing.ICA(
        method=config["ica"]["method"],  # picard, fastica
        fit_params=dict(extended=config["ica"]["extended"]),
        random_state=config["ica"]["random_state"],
    )

    # Create a copy of the raw data to work with ICA, high-pass/low-pass filtered as configured. This is to enhance the detection of artifacts.
    raw_ica = raw.copy().filter(
        l_freq=config["ica"]["filter_l_freq"], h_freq=config["ica"]["filter_h_freq"]
    )

    logger.info(
        f"Fitting ICA (method={config['ica']['method']}, filter={config['ica']['filter_l_freq']}-{config['ica']['filter_h_freq']} Hz)"
    )

    # Fit ICA (takes a few minutes to run)
    ica.fit(raw_ica)

    logger.info(f"ICA fit complete: {ica.n_components_} components")
    return ica, raw_ica


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visually inspect ICA components

    ICA returns as many components as there are **good** channels it was fitted on — so with 32 EEG channels and `CP5` marked bad in step (d), this run produces **31** components, not 32. If you mark a different number of bad channels, that count changes accordingly.

    In MNE, you can both plot the components (scalp topographies) and their sources (time series).
    """)
    return


@app.cell
def _(ica, save_fig):
    # Inspect ICA components.
    # With many components MNE returns a list of figures (paged 20 at a time), which save_fig handles.
    fig_components = ica.plot_components()
    save_fig(fig_components, "ica_components")
    return


@app.cell
def _(ica, raw_ica):
    use_qt_browser()

    # Inspect ICA sources
    ica.plot_sources(raw_ica)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Automatically detecting artifact components

    Rather than deciding by eye which components are artifacts, MNE can flag them automatically. Two complementary detectors are used here, each targeting a different artifact type:

    - **Eye blinks / eye movements** — `ica.find_bads_eog()` correlates every component's time course against the actual **EOG electrodes** (`config['channels']['eog']`). Because it compares against a real recorded signal, this is the most trustworthy of the automatic methods.
    - **Muscle activity** — `ica.find_bads_muscle()` doesn't need a dedicated channel. It scores each component on its **power spectrum slope** (muscle activity is broadband and dominates high frequencies, unlike the 1/f falloff of genuine brain signal) combined with how **spatially focal** the component's topography is.

    **Both detectors are threshold-driven, and both thresholds live in `config['ica']`.** For each one, **higher = stricter = fewer components flagged**:
    - `eog_threshold` (default **3.0**) is a z-score on the correlation with the EOG channels.
    - `muscle_threshold` (default **0.5**) scores each component against a typical muscle profile. MNE's default is fairly permissive on 32-channel EEG — if too many components are being flagged as muscle, raise this toward 0.8–1.0.

    Everything flagged by either detector ends up in `ica.exclude`, so these two numbers directly control how much gets removed *and* how many inspection plots get generated further down. Check the printed count below before moving on.

    **What about heart-beat artifacts?** MNE's equivalent function, `find_bads_ecg()`, needs a real **ECG channel** to correlate against. It can synthesize one from MEG sensors, but *not* from EEG — this recording has 32 EEG + 2 EOG + 2 mastoid + 1 stim channel and no ECG, so cardiac components cannot be detected this way. In practice this matters less for scalp EEG than for MEG: the cardiac field is far weaker at EEG electrodes, and blinks and muscle dominate the artifact budget. If your own setup does record ECG, you can add `ica.find_bads_ecg(raw_ica, ch_name='<your ECG channel>')` alongside the two detectors below.

    As an **independent point of comparison**, the notebook also runs `mne_icalabel`, a neural network that classifies each component from its topography and spectrum alone ('brain', 'eye blink', 'muscle artifact', 'heart beat', ...). Its labels are only *printed* for you to compare against — they do not drive the exclusion decision.
    """)
    return


@app.cell
def _(config, logger, mne, raw, save_fig):
    # Detect blink events directly in the raw data, using the EOG channel(s) named in config.json.
    # Note: this does NOT select ICA components - it finds when blinks happened, which is a useful
    # sanity check that blinks are present and detectable before trusting any automatic component labelling.
    eog_events = mne.preprocessing.find_eog_events(
        raw, ch_name=config["channels"]["eog"]
    )
    logger.info(f"Detected {len(eog_events)} EOG (blink) events")

    # Cut epochs around each detected blink and average them. A clean blink-evoked average with a
    # clear deflection confirms the EOG channels are working and the blinks are well time-locked.
    # The baseline (-0.5 to -0.2 s) is deliberately well before the blink onset: a window like
    # (-0.2, 0) risks already containing the start of the eye movement.
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw,
        ch_name=config["channels"]["eog"],
        baseline=(-0.5, -0.2),
    )
    eog_evoked = eog_epochs.average()
    logger.info(f"Created {len(eog_epochs)} blink epochs")

    fig_eog = eog_evoked.plot(show=True)
    save_fig(fig_eog, "eog_evoked")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flagging components for exclusion

    With the blink signal confirmed above, the two detectors now run against the ICA decomposition. Each returns the indices of the components it flags, plus a per-component score you can inspect.

    The union of both sets becomes `ica.exclude` — the components that will be subtracted from the data when `ica.apply()` runs. Nothing is removed until that step, so you can still inspect and adjust the list first.
    """)
    return


@app.cell
def _(config, ica, logger, raw_ica):
    # Eye components: correlate each component's time course against the EOG channel(s).
    # threshold is a z-score - higher means stricter, so fewer components get flagged.
    eog_inds, eog_scores = ica.find_bads_eog(
        raw_ica,
        ch_name=config["channels"]["eog"],
        threshold=config["ica"]["eog_threshold"],
    )

    # Muscle components: score each component on spectrum slope + topography focality (no extra channel needed).
    # Also higher = stricter. Both index lists come back sorted by score, strongest first.
    muscle_inds, muscle_scores = ica.find_bads_muscle(
        raw_ica,
        threshold=config["ica"]["muscle_threshold"],
    )

    print(
        f"Eye (EOG-correlated) components  [threshold={config['ica']['eog_threshold']}]:",
        eog_inds,
    )
    print(
        f"Muscle components                [threshold={config['ica']['muscle_threshold']}]:",
        muscle_inds,
    )

    # Components to remove: everything either detector flagged
    ica.exclude = sorted(set(eog_inds) | set(muscle_inds))
    print(
        f"\n{len(ica.exclude)} of {ica.n_components_} components marked for exclusion:",
        ica.exclude,
    )
    print(
        "If that looks like too many, raise the thresholds in the 'ica' section of config.json."
    )

    logger.info(
        f"find_bads_eog flagged (threshold={config['ica']['eog_threshold']}): {eog_inds}"
    )
    logger.info(
        f"find_bads_muscle flagged (threshold={config['ica']['muscle_threshold']}): {muscle_inds}"
    )
    logger.info(
        f"Components marked for exclusion ({len(ica.exclude)}/{ica.n_components_}): {ica.exclude}"
    )
    return eog_inds, muscle_inds


@app.cell
def _(eog_inds, ica, label_components, logger, muscle_inds, raw_ica):
    # Independent comparison only - iclabel does NOT drive the exclusion decision above.
    # It classifies each component from its topography and spectrum alone, so agreement with the
    # EOG-correlation and muscle detectors is reassuring, and disagreement is worth a closer look.
    ica_labels = label_components(raw_ica, ica, method="iclabel")
    labels = ica_labels["labels"]
    print(f"{'comp':>5}  {'iclabel says':<18}  flagged by")
    print("-" * 50)
    for _comp in range(ica.n_components_):
        flagged_by = []
        if _comp in eog_inds:
            flagged_by.append("find_bads_eog")
        if _comp in muscle_inds:
            flagged_by.append("find_bads_muscle")
        if flagged_by or labels[_comp] != "brain":
            print(
                f"{_comp:>5}  {labels[_comp]:<18}  {', '.join(flagged_by) or '-'}"
            )  # Show every excluded component, plus any component iclabel considers non-brain
    logger.info(f"iclabel labels (info only): {dict(enumerate(labels))}")
    return


@app.cell
def _(ica, logger, raw):
    # Apply cleaning to initial raw instance
    ica.apply(raw)
    logger.info("ICA applied to raw data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Closer look at the flagged components

    `plot_components` shows only a topography, and `plot_sources` only a raw time series — neither is enough on its own to *justify* excluding a component. `plot_properties` gives a fuller per-component view (time course, power spectrum, topography, and an epochs-image of the component activity) for each excluded component, so you can confirm why it looked like a blink or muscle artifact rather than just trusting the automatic detectors.

    A genuine **blink** component typically shows a frontal topography, large slow deflections in the time course, and power concentrated at low frequencies. A **muscle** component typically shows a focal topography near the edge of the montage (temporalis, neck) and a spectrum that stays high or rises toward the upper frequencies instead of falling off as 1/f.

    These plots are drawn from `raw_ica`, the unmodified copy the ICA was fitted on, so they show the components as they were regardless of when `ica.apply()` runs. If a component here looks like genuine brain activity, adjust `ica.exclude` and re-run `ica.apply()` on a fresh copy of the raw data.
    """)
    return


@app.cell
def _(ica, logger, raw_ica, save_fig):
    logger.info(f"Inspecting properties of excluded components: {ica.exclude}")
    for _comp in ica.exclude:
        figs = ica.plot_properties(raw_ica, picks=_comp)
        save_fig(
            figs, f"ica_properties_comp{_comp}"
        )  # plot_properties returns a list (one figure per picked component)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (f) Interpolating bad channels

    Now, the raw instance is cleaned from artifacts. The data is clean now, so in a last step, the bad channels can be interpolated.

    `interpolate_bads()` reconstructs each bad channel as a weighted combination of its neighbors, using the sensor positions from the montage set in step (a) of the previous notebook — this is why getting the montage right matters even for channels you never mark bad. It also prints the fitted head radius/origin it used for the spline interpolation. This estimate depends on head size: EEG caps commonly come in standard sizes (S/M/L or by head circumference), and using the wrong cap size for a participant subtly worsens interpolation and source localization. Some labs go further and digitize each participant's actual 3D head shape (e.g. with a Polhemus digitizer) for more accurate electrode positions — that's an advanced/optional step this tutorial doesn't cover, but worth knowing about if interpolation quality matters a lot for your analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before interpolating, it's worth visually confirming where the bad channel(s) actually sit relative to the rest of the montage — `plot_sensors` highlights any channel currently in `raw.info['bads']` in red.
    """)
    return


@app.cell
def _(raw, save_fig):
    # 2D sensor layout, bad channels shown in red (use kind='3d' instead if you have pyvista installed)
    fig_sensors = raw.plot_sensors(kind="topomap", show_names=True)
    save_fig(fig_sensors, "sensors")
    return


@app.cell
def _(logger, raw):
    logger.info(f"Interpolating bad channels: {raw.info['bads']}")
    raw.interpolate_bads(reset_bads=True)
    logger.info("Interpolation complete")
    return


@app.cell
def _(raw):
    use_qt_browser()

    # Optional: if you browse the data now, you can see that it is cleaned
    raw.plot(n_channels=len(raw.ch_names), duration=10, scalings=dict(eeg=100e-6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (g) Filtering the data in auditory-relevant frequencies

    In this step, the data is filtered to the band given by `config['filter_bands']`, which for this study's auditory speech-processing analysis is 0.5–30 Hz.

    **Which band is "correct" depends on the analysis, not just the dataset.** Rather than hardcoding one band, `config['filter_bands']` stores a **dictionary of named options** (`auditory_erp`, `broadband`, `alpha`, ...) plus an `active` key selecting which one this run uses — add your own named band and point `active` at it instead of editing numbers inline.

    **Why the high-pass edge matters**: sweat and other slow-drift artifacts typically only distort the signal over a timescale of 3+ seconds, so they show up as very low-frequency drift. Increasing the high-pass cutoff removes this kind of drift, but it also increasingly distorts genuine slow components of the ERP. As a rule of thumb, keep the high-pass edge at **0.5 Hz or below** unless you have a specific, understood reason to go higher.
    """)
    return


@app.cell
def _(config, logger, raw):
    # Final filter using the active band selected in config.json
    l_freq, h_freq = config["filter_bands"]["options"][config["filter_bands"]["active"]]
    raw.filter(l_freq, h_freq)
    logger.info(
        f"Applied final filter: {l_freq}-{h_freq} Hz (band='{config['filter_bands']['active']}')"
    )
    return


@app.cell
def _(raw):
    use_qt_browser()

    # Final inspection before saving
    raw.plot(n_channels=len(raw.ch_names), duration=10, scalings=dict(eeg=100e-6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (h) Caching cleaned data for further analyses

    In this final step, the cleaned raw data is cached in MNE’s native .fif format to a folder created at the beginning of this script. This file will be loaded in the next notebook for further processing.
    """)
    return


@app.cell
def _(cleaned_data_folder, logger, participant_stem, raw):
    # Save the cleaned raw data
    save_path = cleaned_data_folder / f"{participant_stem}_cleaned_raw.fif"
    raw.save(save_path, overwrite=True)
    logger.info(f"Saved cleaned data to {save_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (i) Generating a cleaning report

    As a final step, `mne.Report` bundles the key QC outputs from this run — the cleaned raw data with its PSD, and the ICA decomposition together with which components were excluded — into a single, self-contained HTML file. This is saved to `config['paths']['reports_folder']` and can be opened in any browser or shared with a collaborator without needing to re-run the notebook, which is a nicer artifact to keep per participant than a folder of loose screenshots.
    """)
    return


@app.cell
def _(
    Path,
    config,
    ica,
    logger,
    mne,
    participant_filename,
    participant_stem,
    raw,
    raw_ica,
):
    reports_folder = Path(config["paths"]["reports_folder"])
    reports_folder.mkdir(exist_ok=True)

    report = mne.Report(title=f"EEG cleaning report — {participant_filename}")
    report.add_raw(raw=raw, title="Cleaned raw data", psd=True, butterfly=False)
    report.add_ica(ica=ica, title="ICA", inst=raw_ica, picks=ica.exclude)

    report_path = reports_folder / f"{participant_stem}_cleaning_report.html"
    report.save(report_path, overwrite=True)

    logger.info(f"Saved cleaning report to {report_path}")
    return


if __name__ == "__main__":
    app.run()
