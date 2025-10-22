"""
Utility functions for the MTF extraction pipeline.

This module provides small helpers such as reproducible RNG configuration,
file handling, and numeric helpers. It is deliberately lightweight to avoid
introducing unnecessary dependencies into the core modules.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def setup_logging():
    """Configure a basic logger for the application.

    The logger writes to stdout at INFO level and includes the module name
    and message. Downstream modules should use logging.getLogger(__name__).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (if available) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def next_power_of_two(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def stratified_sample(indices: List[int], groups: List[int], n: int) -> List[int]:
    """
    Sample n items from indices stratified by group labels.

    Parameters
    ----------
    indices: list of candidate indices.
    groups: list of group labels (same length as indices).
    n: total number of samples to draw.

    Returns
    -------
    A list of selected indices (subset of input indices) of length <= n.

    Notes
    -----
    The function splits the candidates into unique group buckets and samples
    roughly equal numbers from each. If the total number of candidates is
    smaller than n, all indices are returned. If n is not divisible by the
    number of groups, extra samples are drawn at random from the remaining
    population.
    """
    if n <= 0 or not indices:
        return []
    from collections import defaultdict

    buckets: defaultdict[int, List[int]] = defaultdict(list)
    for idx, g in zip(indices, groups):
        buckets[g].append(idx)

    # Determine number per group
    unique_groups = list(buckets.keys())
    if not unique_groups:
        return []
    per_group = n // len(unique_groups)
    remainder = n % len(unique_groups)
    selected: List[int] = []
    rng = np.random.default_rng()
    for g in unique_groups:
        bucket = buckets[g]
        if per_group > 0:
            k = min(per_group, len(bucket))
            selected.extend(rng.choice(bucket, size=k, replace=False).tolist())

    # Distribute remainder across groups with available candidates
    remaining = []
    for g in unique_groups:
        bucket = buckets[g]
        unused = set(bucket) - set(selected)
        remaining.extend(list(unused))
    if remainder > 0 and remaining:
        k = min(remainder, len(remaining))
        selected.extend(rng.choice(remaining, size=k, replace=False).tolist())

    return selected


def bin_edges_to_groups(bin_edges: Sequence[float], values: Sequence[float]) -> List[int]:
    """
    Assign each value to a bin index given bin edges.

    Parameters
    ----------
    bin_edges : sequence of floats of length m+1 (monotonic)
        Edges of bins. Each value is placed into the bin index i such that
        bin_edges[i] <= value < bin_edges[i+1]. Values below the first edge
        are assigned to bin 0, values >= last edge assigned to the last bin.
    values : sequence of floats
        Input values to assign.

    Returns
    -------
    list of int
        Bin indices for each input value.
    """
    bins = []
    m = len(bin_edges) - 1
    for val in values:
        idx = 0
        while idx < m and val >= bin_edges[idx + 1]:
            idx += 1
        bins.append(idx)
    return bins