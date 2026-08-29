"""Pure-numpy utilities shared by dataset and other data modules.

No torch or heavy dependencies here so these are importable and testable
in any environment, including CI runners without a GPU stack.
"""

from __future__ import annotations

import numpy as np


def center_sample(vol: np.ndarray, n: int) -> np.ndarray:
    """Return the center `n` slices from a (D, H, W) volume.

    If the volume has fewer than `n` slices, pads with edge replication so the
    output is always exactly (n, H, W). This keeps batch collation simple.

    Why center sampling as the default:
        The relevant knee anatomy (ACL in the notch, meniscal body, articular
        cartilage) is generally near the center of each series. Center sampling
        is a strong enough heuristic for a baseline and requires no learned
        slice-selection module. Upgrade to attention-based sampling in Week 3-4
        if the per-label AUC analysis reveals problems.
    """
    d = vol.shape[0]
    if d >= n:
        start = (d - n) // 2
        return vol[start : start + n]
    # Pad: repeat first and last slice alternately from both ends.
    pad_total = n - d
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    before = np.repeat(vol[[0]], pad_before, axis=0)
    after = np.repeat(vol[[-1]], pad_after, axis=0)
    return np.concatenate([before, vol, after], axis=0)
