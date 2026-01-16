# eeg/preprocessing.py
import numpy as np

try:
    from utils.autopreprocessing import dataset
    from utils.interprocessing import interdataset
except ImportError:
    from autopreprocessing import dataset
    from interprocessing import interdataset


def dbg(print_debug, *args):
    if print_debug:
        print("[DEBUG]", *args)


def autopreprocess(csv_path, fs, print_debug=False):
    """
    STEP 1: Load raw EEG & Auto-preprocessing
    Returns: interdataset object (data in (1, channels, samples))
    """

    dbg(print_debug, f"Loading RAW: {csv_path}")
    ds = dataset(csv_path, Fs=fs)
    ds.loaddata()

    dbg(print_debug, "Raw:", ds.data.shape, ds.data.dtype)

    # Auto pipeline (artifact correction)
    ds.bipolarEOG()
    ds.apply_filters()
    ds.correct_EOG()
    ds.detect_emg()
    ds.detect_jumps()
    ds.detect_kurtosis()
    ds.detect_extremevoltswing()
    ds.residual_eyeblinks()
    ds.define_artifacts()

    dbg(print_debug, "Auto-preprocess INFO:")
    for k, v in ds.info.items():
        dbg(print_debug, f"  {k}: {v}")

    # Expand for interdataset compatibility (1, ch, time)
    ds.data = np.expand_dims(ds.data, axis=0)

    # Wrap into interdataset
    ds = interdataset(ds.__dict__)

    return ds


def segment_and_reference(ds, epoch_sec, fs, expected_channels=26, print_debug=False):
    """
    STEP 2-3: Rereference + segmentation
    Returns segmented: (epochs, channels, samples)
    """

    # Rereference
    dbg(print_debug, "Applying avg reference...")
    ds.rereference('avgref')

    # Segment
    dbg(print_debug, f"Segment: {epoch_sec}s, artifact=yes")

    try:
        ds.segment(
            trllength=float(epoch_sec),
            remove_artifact="yes",
            marking="no",
        )
    except Exception as e:
        dbg(print_debug, "SEGMENT FAILED:", e)
        return None, "segment-failed"

    # Extract segmented
    segmented = ds.data[:, :expected_channels, :]   # (epochs, 26, samples)
    labels = ds.labels[:expected_channels]

    dbg(print_debug, "Segmented:", segmented.shape)

    if segmented.ndim != 3:
        return None, "bad-segment-dim"

    n_epochs, n_ch, n_samp = segmented.shape

    if n_epochs == 0:
        return None, "no-epochs"

    if n_ch != expected_channels:
        return None, f"unexpected-channels-{n_ch}-expected-{expected_channels}"

    # Duration check
    raw_dur = n_samp / fs
    if raw_dur < epoch_sec:
        return None, "duration-too-short"

    return segmented, None
