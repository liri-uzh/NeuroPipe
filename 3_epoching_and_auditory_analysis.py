# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib==3.11.1",
#     "mne[full]==1.12.1",
#     "mne-qt-browser @ git+https://github.com/larsoner/mne-qt-browser@main",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
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
    ## (a) Creating epochs from cleaned raw files

    Now that the raw data has been cleaned, the next step is to create **epochs**—segments of EEG data time-locked to specific events. This is done to isolate the neural responses to specific stimuli and exclude unrelated portions of the recording.

    ### Finding events

    The `find_events` function extracts all the recorded events from the stim channel named in `config.channels.stim_channel` — on a BioSemi system this is the `Status` channel. These events correspond to the triggers recorded during the experiment, indicating when specific stimuli or actions occurred. As you can see in the output of this function, there are other triggers present in the data, but they are not crucial for this analysis.

    ### Creating epochs

    Using the `Epochs` class, the cleaned data is segmented into epochs. Here's an explanation of the key parameters, all read from `config.epochs` so they can be **fully adapted to your own trigger scheme**:
    - **`events`**: The extracted events from the stim channel.
    - **`event_id`**: A dict mapping a label to the trigger code(s) of interest — `{'sentence onset': 256}` here, since the trigger code 256 marks the onset of speech. "Soft triggers" were used, so the same code (256) was used for every trial, independent of the experimental condition. If your paradigm uses different codes per condition, add more entries to this dict.
    - **`tmin=-0.200`**: Includes 200 ms of data before the stimulus onset as a baseline period. This is a standard choice for EEG analyses to capture pre-stimulus activity.
    - **`tmax=7.000`**: Captures 7 seconds of data following the stimulus onset. This duration was chosen because the sentences in the experiment had an average length of 7 seconds, ensuring the epochs cover the entire stimulus duration. Set `tmax` to cover the longest window any of your downstream analyses need — individual analyses can always crop back down (see the AEP step below), but they can't recover data outside the original epoch.
    - **`baseline=(-0.200, 0)`**: Sets a baseline correction window from -200 to 0 ms to account for drift or other slow fluctuations in the data.

    This step creates time-locked epochs that align the EEG signals to the presentation of each stimulus, preparing the data for further analyses.

    As you can seen from the output, there are 120 epochs, which correspond to the 120 sentences presented in the experiment.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd

    # mne.set_log_level('WARNING')  # suppresses MNE messages; only show warnings and errors; activate if you want to see all messages
    # Every researcher/setup-specific setting lives in config.py (shared with the earlier notebooks). Edit that file, not this cell, to adapt this tutorial to your own data.
    from config import config

    # Folder where cleaned data is stored
    cleaned_data_folder = config.paths.cleaned_data_folder

    # Participant's cleaned filename, derived the same way notebook 1 named it when saving
    participant_stem = config.paths.raw_filename.stem
    participant_filename = f"{participant_stem}_cleaned_raw.fif"

    # Folder where the analysis data will be stored
    analyses_folder = config.paths.analyses_folder
    analyses_folder.mkdir(exist_ok=True)  # create the folder if it doesn't exist

    # Folder where figures will be saved
    figures_folder = config.paths.figures_folder
    figures_folder.mkdir(exist_ok=True)

    # Load participant's log file (we need this to work with conditions)
    # It's a txt file
    participant_log = pd.read_csv(config.paths.log_path, sep="\t")

    # Load the cleaned data
    raw_cleaned = mne.io.read_raw_fif(
        cleaned_data_folder / participant_filename, preload=True
    )
    return (
        analyses_folder,
        config,
        figures_folder,
        mne,
        np,
        participant_filename,
        participant_log,
        participant_stem,
        pd,
        plt,
        raw_cleaned,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logging

    As in notebook 1, this notebook writes a plain-text log to `config.paths.logs_folder` alongside its normal cell output — a permanent, greppable record of what this run actually did (how many events/epochs were found, condition counts, ...), independent of whether anyone was watching it run.

    The log file is opened in **append** mode, so re-running the notebook adds to the existing log rather than overwriting it. Each run starts with a timestamped `=== Starting ... ===` line marking where one run ends and the next begins.
    """)
    return


@app.cell(hide_code=True)
def _(config, figures_folder, participant_filename, participant_stem):
    import logging

    logs_folder = config.paths.logs_folder
    logs_folder.mkdir(exist_ok=True)

    logger = logging.getLogger(f"neuropipe.epoching.{participant_filename}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()  # avoid duplicate handlers if this cell is re-run

    log_path = logs_folder / f"{participant_stem}_epoching.log"
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

    logger.info(
        f"=== Starting epoching/analysis pipeline for {participant_filename} ==="
    )

    def save_fig(fig, name, dpi=300):
        """Save a figure into the figures folder, prefixed with the participant name.

        Some MNE plotting functions return a *list* of figures rather than one, so both
        cases are handled. Interactive browsers such as raw.plot() or epochs.plot() cannot
        be saved this way - under the Qt backend they are live browser windows, not
        matplotlib figures.
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


@app.cell
def _(config, mne, raw_cleaned):
    print(config.channels.stim_channel)

    events1 = mne.find_events(raw_cleaned, stim_channel=config.channels.stim_channel)
    events2 = mne.find_events(
        raw_cleaned, stim_channel=config.channels.stim_channel, initial_event=True
    )
    raw_cleaned.plot(picks="stim", block=True)
    return


@app.cell
def _(config, logger, mne, raw_cleaned):
    # First, find all the events recorded by the stim channel named in config.py
    events = mne.find_events(raw_cleaned, stim_channel=config.channels.stim_channel)

    epochs = mne.Epochs(
        raw_cleaned,
        events=events,
        event_id=config.epochs.event_id,
        tmin=config.epochs.tmin,
        tmax=config.epochs.tmax,
        baseline=config.epochs.baseline,
        initial_event=True,
        preload=True,
    )

    logger.info(
        f"Found {len(events)} events; created {len(epochs)} epochs ({config.epochs.tmin}-{config.epochs.tmax} s)"
    )

    epochs
    return epochs, events


@app.cell
def _(events, raw_cleaned):
    # '%matplotlib qt' command supported automatically in marimo
    # Switching to the interactive Qt backend here, right before the first plot — this stays in effect
    # for the rest of the notebook, so it only needs to be set once.

    # Show every channel the object contains (EEG + EOG + misc + stim), rather than a hardcoded count
    raw_cleaned.plot(
        n_channels=len(raw_cleaned.ch_names),
        duration=10,
        scalings=dict(eeg=100e-6),
        events=events,
    )
    return


@app.cell
def _(epochs):
    epochs.plot(n_channels=len(epochs.ch_names), n_epochs=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (b) Adding metadata from the log file to the epochs to work with conditions

    In this step, metadata from the participant log (read in from the `.txt` file) is added to the epochs. The participant log contains information about all 120 trials in the experiment, including filenames of the sentences, response accuracy (`hit`), and reaction time (`RT`). The number of entries in the log (120) matches the number of epochs in the dataset, and MNE automatically ensures that the dimensions of the metadata align with the epochs.

    ### Modifying the participant log

    The participant log initially includes only filenames and trial-specific details. To make it more informative, a new column, **`condition`**, is added, which classifies each trial as "random" or "context" based on the filename. This rule is configured in `config.condition_labeling` — it labels rows whose `column` value starts with `keyword` as `keyword_label`, and everything else as `default_label`. Adapt this to however your own log/filenames encode condition (it doesn't have to be a filename prefix — you can change this cell's logic entirely if your rule is more complex).

    ### Adding metadata to epochs

    Once the participant log is updated, it is assigned to the `metadata` attribute of the epochs. This associates the condition, response accuracy, and reaction time information with each epoch.

    ### Importance of matching dimensions

    It is crucial that the participant log and the epochs have the same number of entries (120 in this case). If the dimensions do not match, MNE will throw an error, as it relies on a one-to-one correspondence between the metadata and the epochs.

    ### Potential uses of metadata

    The metadata enables several analyses and filtering options, such as:
    - Dividing epochs into conditions (e.g., "random" vs. "context") for condition-specific analyses.
    - Excluding trials based on criteria like:
      - Incorrect responses (`hit` equals 0).
      - Reaction times (`RT`) that exceed a certain threshold.
    """)
    return


@app.cell
def _(epochs, raw_cleaned):
    n_channels, n_samples = raw_cleaned.get_data(picks="eeg").shape
    n_epochs, _, n_samples_per_epoch = epochs.get_data(picks="eeg").shape

    print(
        f"Continuous data: {n_channels} EEG channels, {n_samples} samples each "
        f"({raw_cleaned.times[-1]:.1f} s at {raw_cleaned.info['sfreq']:.0f} Hz)"
    )
    print(
        f"Epoched data:    {n_epochs} epochs, {n_channels} EEG channels, "
        f"{n_samples_per_epoch} samples each ({epochs.tmin} to {epochs.tmax} s around each trigger)"
    )
    return


@app.cell
def _(config, epochs, logger, np, participant_log):
    # Add new column in participant_log to store the condition information, using the rule configured in config.py
    condition_cfg = config.condition_labeling
    participant_log["condition"] = np.where(
        participant_log[condition_cfg.column].str.startswith(condition_cfg.keyword),
        condition_cfg.keyword_label,
        condition_cfg.default_label,
    )

    # Add the condition to the epochs
    epochs.metadata = participant_log

    logger.info(
        f"Condition counts: {epochs.metadata['condition'].value_counts().to_dict()}"
    )

    # Check the metadata
    epochs.metadata.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Dividing epochs by condition

    The next cell demonstrates how to divide the `epochs` object into separate subsets based on the condition (e.g., "context" or "random"). This can be useful for performing condition-specific analyses.
    """)
    return


@app.cell
def _(epochs):
    # Create epochs containing only segments from "context" and "random" conditions
    epochs_context = epochs[epochs.metadata["condition"] == "context"]
    epochs_random = epochs[epochs.metadata["condition"] == "random"]

    # Check number of epochs
    epochs_context
    return epochs_context, epochs_random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (c) Calculating and inspecting auditory evoked potentials (AEPs) locked to stimulus onset

    For the sake of brevity and illustration, the auditory evoked potential (AEP) is calculated across all conditions combined. If relevant to your hypothesis, this analysis could also be performed separately for each condition (e.g., "random" and "context").

    **Why crop for this analysis at all?** The epochs created in step (a) span the full `config.epochs.tmax` (7 s here), because that's long enough to cover every downstream analysis, including the time-frequency analysis in step (d) which needs the extra length for good low-frequency resolution. But an AEP is only interested in the early, transient response — the epoch length used for an analysis should match what that analysis actually needs, not be reused unthinkingly from step (a). That's why this step crops back down to a short window (`config.aep`) rather than analyzing the full 7 s epoch.

    ### Steps:
    1. **Cropping epochs**: AEPs are typically analyzed within shorter time windows to focus on key components such as P1 (~50 ms), N1 (~100 ms), and P2 (~200 ms). To do this, the `.crop()` method is used to reduce the epochs to the window given by `config.aep.crop_tmin`/`['crop_tmax']` (-200 ms to 600 ms by default).

    2. **Averaging epochs**: To compute the evoked response, the `.average()` method is used, which averages the EEG signal across all 120 epochs.

    3. **Visualizing the AEP**: The `.plot_joint()` method is used to visualize the evoked response. This method combines a standard AEP plot with topographic maps showing the spatial distribution of activity across the scalp. No explicit time points are given, so MNE selects them automatically from the peaks it detects in the evoked response — this adapts to whatever your data actually shows, instead of assuming where the peaks should be. The figure is saved to `config.paths.figures_folder`.

    ### Observations:
    - **P1 (~50 ms)**: A distinct P1 peak is visible shortly after timepoint 0, with a flat baseline before the stimulus onset. This indicates that the trigger system functioned correctly.
    - **N1 (~100-125 ms)**: An N1 component is present but not strongly pronounced across all channels.
    - **P2 (~200-250 ms)**: A clear P2 peak is observed following the N1.

    ### Conclusion:
    Using `.average()` and `.plot_joint()`, a reliable AEP is observed in participant p34, with well-defined P1 and P2 peaks and a weaker N1. This indicates good data quality and confirms that the experimental setup is functioning as expected.
    """)
    return


@app.cell
def _(config, epochs, logger, save_fig):
    # Crop the epochs to the time window of interest, as configured for the AEP analysis
    epochs_cropped = epochs.copy().crop(
        tmin=config.aep.crop_tmin, tmax=config.aep.crop_tmax
    )

    # Average across all epochs
    evoked = epochs_cropped.average()

    logger.info(
        f"Computed AEP: cropped to {config.aep.crop_tmin}-{config.aep.crop_tmax} s, averaged over {len(epochs_cropped)} epochs"
    )

    # Plot the evoked response with topomaps. Without an explicit `times` argument, MNE picks the
    # topomap time points automatically from the peaks it finds in the evoked response.
    fig_joint = evoked.plot_joint()
    save_fig(fig_joint, "aep_joint")
    return evoked, fig_joint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Storing evoked AEP data for further analysis

    This example demonstrates how the averaged auditory evoked potential (AEP) data can be stored for further analysis.

    Using the evoked response object, a **DataFrame** is created where:
    - Each row corresponds to a **time point** in the AEP.
    - Each column corresponds to a specific **channel**.

    The DataFrame is saved as a `.csv` file, which stores the data in Volts. This format makes the data accessible for other analyses in external software or programming environments.
    """)
    return


@app.cell
def _(analyses_folder, evoked, logger, participant_stem, pd):
    # Create a DataFrame with the evoked data
    # Rows correspond to time points, and columns correspond to channel names
    evoked_df = pd.DataFrame(evoked.data.T, columns=evoked.ch_names, index=evoked.times)

    # Rename the index to 'time' for clarity
    evoked_df.index.name = "time"

    # Save the DataFrame as a .csv file
    # Note: The data is in Volts
    evoked_csv_path = analyses_folder / f"{participant_stem}_evoked_data.csv"
    evoked_df.to_csv(evoked_csv_path)
    logger.info(f"Saved evoked data to {evoked_csv_path}")

    evoked_df.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing conditions with Global Field Power (GFP)

    The AEP above was computed across both conditions combined. Now that conditions are split (`epochs_context`/`epochs_random`, from step (b)), a quick way to compare them without picking a single channel or ROI upfront is **Global Field Power** — the standard deviation across channels at each time point. It summarizes "how much is going on across the whole scalp" at every moment, independent of any assumption about *where* an effect should appear, which makes it a good first-pass condition comparison before committing to a specific channel selection.
    """)
    return


@app.cell
def _(config, epochs_context, epochs_random, plt, save_fig):
    fig_gfp, ax = plt.subplots(figsize=(8, 4))

    for label, cond_epochs in [("context", epochs_context), ("random", epochs_random)]:
        evoked_cond = (
            cond_epochs.copy()
            .crop(tmin=config.aep.crop_tmin, tmax=config.aep.crop_tmax)
            .average()
        )
        gfp = evoked_cond.data.std(axis=0)
        ax.plot(evoked_cond.times, gfp, label=label)

    ax.set_title("Global Field Power by condition")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GFP (std across channels)")
    ax.axvline(0, linestyle="--", color="gray")
    ax.legend()

    save_fig(fig_gfp, "gfp_by_condition")
    fig_gfp
    return (fig_gfp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (d) Performing time-frequency analysis of epochs

    In this step, a **time-frequency analysis** (TFR) is performed on the epochs using the **Morlet wavelet transform**. TFR provides insight into how power at different frequencies evolves over time, offering a more detailed view of the dynamics of neural responses compared to standard time-domain analyses.

    Unlike the AEP step above, this analysis uses the **full-length epochs** from step (a) rather than a cropped version — low-frequency wavelets need a longer time window to resolve, so shortening the epoch here would degrade frequency resolution at the low end.

    ### Explanation of the parameters (`config.tfr`)

    - **`freqs`**: Specifies the frequencies of interest for the analysis, built from `freq_min`/`freq_max`/`n_freqs`. In this example, frequencies between 1 Hz and 30 Hz are analyzed, using a total of 30 frequency steps.
    - **`n_cycles`**: Determines the number of cycles in each Morlet wavelet. A value of 3 ensures a good balance between time and frequency resolution. Higher values provide better frequency resolution but worse temporal resolution.
    - **`method='morlet'`**: Indicates that the Morlet wavelet transform is used for the TFR computation. This is one of the most common methods for EEG time-frequency analyses.
    - **`return_itc=False`**: Specifies that only the power of the signal is of interest, not the inter-trial coherence (ITC).
    - **`average=True`**: Averages the TFR across all epochs to obtain a single power estimate for each time-frequency point.
    """)
    return


@app.cell
def _(config, epochs, logger, np):
    # Define frequencies and resolution from config.py, but this can be more fine-grained if needed
    tfr_cfg = config.tfr
    frequencies = np.linspace(tfr_cfg.freq_min, tfr_cfg.freq_max, tfr_cfg.n_freqs)
    n_cycles = tfr_cfg.n_cycles

    logger.info(
        f"Computing TFR: {tfr_cfg.freq_min}-{tfr_cfg.freq_max} Hz, {tfr_cfg.n_freqs} steps, n_cycles={n_cycles}"
    )

    # Compute time-frequency representation
    power = epochs.compute_tfr(
        method="morlet",
        freqs=frequencies,
        return_itc=False,
        n_cycles=n_cycles,
        average=True,
    )
    return (power,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing time-frequency analysis results

    To explore the time-frequency representation of the data, `power.plot_topo()` is used to visualize the power of frequencies across time for each electrode channel. This visualization provides an overview of how power at different frequencies evolves over time across the scalp.

    #### Interactive plot

    In this plot:
    - Each subplot corresponds to an individual electrode channel.
    - You can **click on any subplot** to view a larger, more detailed time-frequency representation for that specific channel.

    #### Observations

    Typically, you would observe:
    - An increase in power in the **5–10 Hz frequency range** after speech onset, which is expected based on neural activity related to speech processing.
    - Limited resolution for frequencies below 4 Hz due to the relatively short epoch duration.

    #### Use cases for time-frequency data

    The time-frequency data can be used for various analyses, such as:
    - **Condition-based comparisons**: Analyzing power differences in specific frequency bands (e.g., delta, theta, alpha) between conditions, such as "random" vs. "context."
    - **Group-level comparisons**: Investigating differences in power between experimental groups, such as younger vs. older adults, or groups with varying degrees of hearing loss.
    - **Regions of interest (ROI) analysis**: Focusing on specific clusters of electrodes to study localized neural activity in predefined regions of interest (e.g., frontal, central, or parietal regions).

    #### Customizing frequency resolution

    The resolution of the time-frequency analysis can be adjusted by modifying the `freqs` parameter in the TFR computation step. Higher resolution allows for more detailed frequency analysis but may increase computational cost.
    """)
    return


@app.cell
def _(power, save_fig):
    fig_tfr = power.plot_topo(
        title="Power of frequencies across time for each electrode channel",
        baseline=(-0.200, 0),
        mode="logratio",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    save_fig(fig_tfr, "tfr_topo")
    return (fig_tfr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Storing time-frequency power data for further analysis

    This step demonstrates how to save the time-frequency power data for future analyses. The power data is converted into a **DataFrame**, making it easy to inspect and manipulate. Each row corresponds to a time point, and columns include frequencies and channel information.

    ### Steps:
    1. **Convert power data to a DataFrame**:
       - The `.to_data_frame()` method is used to convert the `power` object into a pandas DataFrame. This format is flexible and well-suited for further processing in Python or external software.

    2. **Set time as the index**:
       - The `time` column is set as the index to make the data easier to navigate, especially for time-based analyses.

    3. **Save the DataFrame to a `.csv` file**:
       - The data is stored as a `.csv` file for portability. This ensures the power data can be used in other environments or tools.
    """)
    return


@app.cell
def _(analyses_folder, logger, participant_stem, power):
    # Create a DataFrame with the power data
    power_df = power.to_data_frame()

    # Make column 'times' the index
    power_df.set_index("time", inplace=True)

    # Store the power data in a .csv file
    power_csv_path = analyses_folder / f"{participant_stem}_power_data.csv"
    power_df.to_csv(power_csv_path)
    logger.info(f"Saved power data to {power_csv_path}")

    power_df.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (e) Generating an analysis report

    As in notebook 1, `mne.Report` bundles the key outputs of this run — the epochs summary, the AEP, the GFP condition comparison, and the TFR topo plot — into one self-contained HTML file, saved to `config.paths.reports_folder`.
    """)
    return


@app.cell
def _(
    config,
    epochs,
    evoked,
    fig_gfp,
    fig_joint,
    fig_tfr,
    logger,
    mne,
    participant_filename,
    participant_stem,
):
    reports_folder = config.paths.reports_folder
    reports_folder.mkdir(exist_ok=True)

    report = mne.Report(title=f"EEG analysis report — {participant_filename}")
    report.add_epochs(epochs=epochs, title="Epochs")
    report.add_evokeds(evokeds=evoked, titles="AEP (all conditions)")
    report.add_figure(fig_joint, title="AEP joint plot")
    report.add_figure(fig_gfp, title="GFP by condition")
    report.add_figure(fig_tfr, title="Time-frequency power (topo)")

    report_path = reports_folder / f"{participant_stem}_analysis_report.html"
    report.save(report_path, overwrite=True)

    logger.info(f"Saved analysis report to {report_path}")
    return


if __name__ == "__main__":
    app.run()
