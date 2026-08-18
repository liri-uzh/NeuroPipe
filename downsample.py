import mne
raw = mne.io.read_raw_bdf("data/p34.bdf")
raw = raw.resample(500)
raw.export("data/p34_500.bdf")
