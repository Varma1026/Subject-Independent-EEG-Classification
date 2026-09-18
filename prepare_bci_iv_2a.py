# %% [markdown]
# # BCI Competition IV 2a → S³-Mamba-DA `.npz` preparation
#
# 1. Download the official BCI Competition IV Dataset 2a after accepting its
#    terms: https://www.bbci.de/competition/iv/
# 2. Place **only** the labelled training files `A01T.gdf` ... `A09T.gdf` in
#    `data/BCICIV_2a_gdf/` (or change `RAW_DIR` below).
# 3. Run every cell.  It writes `data/mi_eeg.npz`, with:
#    `X [N, 22, 1000]`, `y [N]` (0=left, 1=right, 2=feet, 3=tongue), and
#    `subjects [N]` (0–8).
#
# This uses each 4-second imagery interval beginning at cue onset. Dataset 2a
# is sampled at 250 Hz, so every output trial has exactly 1000 samples.
#
# This is a subject-independent dataset construction: it uses no target data
# during an individual LOSO fold. Do **not** put `A??E.gdf` files in RAW_DIR
# unless you explicitly decide to use released evaluation labels in a separate
# protocol.

# %%
# Run once in the notebook kernel if MNE is not installed.
# %pip install -U mne numpy

from __future__ import annotations

from pathlib import Path
import json
import re

import mne
import numpy as np

# Keep MNE output readable in notebooks.
mne.set_log_level("WARNING")

RAW_DIR = Path("data/BCICIV_2a_gdf")
OUTPUT_PATH = Path("data/mi_eeg.npz")
SFREQ = 250
N_EEG_CHANNELS = 22
TRIAL_SECONDS = 4.0
EXPECTED_SAMPLES = int(SFREQ * TRIAL_SECONDS)

# Official GDF cue event codes. Output labels are deliberately zero-based for
# PyTorch CrossEntropyLoss, while `class_names` preserves their meaning.
CUE_TO_LABEL = {769: 0, 770: 1, 771: 2, 772: 3}
CLASS_NAMES = np.array(["left_hand", "right_hand", "feet", "tongue"])


# %%
def annotation_code(description: str) -> int | None:
    """Return a numeric GDF event code, tolerating MNE's string annotations."""
    match = re.search(r"\d+", str(description))
    return int(match.group()) if match else None


def extract_trials_from_gdf(gdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract [trials, 22, 1000] trials and zero-based cue labels from one T file.

    The Dataset 2a specification places the 22 EEG channels before its three
    EOG channels. EOG is intentionally excluded from the model input.
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], SFREQ):
        raise ValueError(f"{gdf_path.name}: expected {SFREQ} Hz, got {raw.info['sfreq']} Hz")
    if len(raw.ch_names) < N_EEG_CHANNELS:
        raise ValueError(f"{gdf_path.name}: found only {len(raw.ch_names)} channels")

    # GDF event annotation labels are normally strings such as '769'.
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # Match numeric codes instead of relying on a particular string rendering
    # (e.g. MNE versions may expose descriptions as '769' or '769.0').
    requested = {description: internal_id for description, internal_id in event_id.items()
                 if annotation_code(description) in CUE_TO_LABEL}
    found_codes = {annotation_code(description) for description in requested}
    if found_codes != set(CUE_TO_LABEL):
        missing = set(CUE_TO_LABEL) - found_codes
        raise RuntimeError(f"{gdf_path.name}: missing cue codes {sorted(missing)}; available={event_id}")

    # tmax is inclusive in MNE. Subtract one sample to obtain precisely 1000
    # time points: cue sample through cue + 999 samples.
    epochs = mne.Epochs(
        raw,
        events,
        event_id=requested,
        tmin=0.0,
        tmax=TRIAL_SECONDS - 1.0 / SFREQ,
        baseline=None,
        picks=list(range(N_EEG_CHANNELS)),
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    x = epochs.get_data(copy=True).astype(np.float32) * 1e6  # volts → microvolts
    id_to_label = {internal_id: CUE_TO_LABEL[annotation_code(description)]
                   for description, internal_id in requested.items()}
    y = np.asarray([id_to_label[event] for event in epochs.events[:, 2]], dtype=np.int64)

    if x.shape[1:] != (N_EEG_CHANNELS, EXPECTED_SAMPLES):
        raise RuntimeError(f"{gdf_path.name}: got {x.shape}; expected [trials,22,1000]")
    if len(x) != 288:
        raise RuntimeError(
            f"{gdf_path.name}: extracted {len(x)} trials, expected 288. "
            "Check that this is a complete labelled A??T.gdf file."
        )
    return x, y


# %%
training_files = [RAW_DIR / f"A{subject:02d}T.gdf" for subject in range(1, 10)]
missing_files = [str(path) for path in training_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Download Dataset 2a and place these labelled files in "
        f"{RAW_DIR.resolve()}:\n" + "\n".join(missing_files)
    )

all_x, all_y, all_subjects = [], [], []
for subject_index, gdf_path in enumerate(training_files):
    x, y = extract_trials_from_gdf(gdf_path)
    all_x.append(x)
    all_y.append(y)
    all_subjects.append(np.full(len(y), subject_index, dtype=np.int64))
    print(f"{gdf_path.name}: X={x.shape}, class counts={np.bincount(y, minlength=4).tolist()}")

X = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects, axis=0)

assert X.shape == (9 * 288, N_EEG_CHANNELS, EXPECTED_SAMPLES), X.shape
assert np.array_equal(np.bincount(y, minlength=4), np.full(4, 9 * 72)), np.bincount(y, minlength=4)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y,
    subjects=subjects,
    class_names=CLASS_NAMES,
    sfreq=np.asarray(SFREQ),
    window_seconds=np.asarray([0.0, TRIAL_SECONDS], dtype=np.float32),
)

metadata = {
    "source": "BCI Competition IV, Dataset 2a; labelled A01T.gdf–A09T.gdf",
    "trials": int(len(X)),
    "shape": list(X.shape),
    "labels": {str(i): name for i, name in enumerate(CLASS_NAMES.tolist())},
    "subject_encoding": {str(i): f"A{i + 1:02d}" for i in range(9)},
    "sampling_hz": SFREQ,
    "epoch_seconds_relative_to_cue": [0.0, TRIAL_SECONDS],
    "unit": "microvolts",
    "eog_channels_used": False,
}
metadata_path = OUTPUT_PATH.with_suffix(".metadata.json")
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

print(f"\nSaved {OUTPUT_PATH.resolve()}")
print(f"Saved {metadata_path.resolve()}")
print("Final arrays:", X.shape, y.shape, subjects.shape)


# %% [markdown]
# ## Next step
#
# In `S3_Mamba_DA_LOSO.ipynb`, leave `CFG.data_path = 'data/mi_eeg.npz'` as-is
# and run the training cells. The class labels are already zero-based, which is
# required by the classifier loss.
