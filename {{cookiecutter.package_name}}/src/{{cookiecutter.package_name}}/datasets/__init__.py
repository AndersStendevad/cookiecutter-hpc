"""Datasets
========
Module for loading datasets into torch.
All of these functions relate to actions doing training.

This includes the class MusicDataset, which is subclassed from the Dataset class from torch.

This submodule contains functions and classes from the following files:

* MusicDataset.py
* transforms.py

"""

from {{cookiecutter.package_name}}.datasets.MusicDataset import MusicDataset
from {{cookiecutter.package_name}}.datasets.transforms import (
    ToNumpy,
    ToTensor,
    trnsfrm_idx_to_note,
    trnsfrm_note_to_idx,
    pad_EB,
    pad_EB_fixed,
    split_tracks_src_trg,
    CropEnding,
    # TriGram,
    get_single_track,
    get_populated_single_track,
    unsqueeze_list,
    cap_list,
    remove_tracks,
    split_tracks_include_instrument,
    split_tracks_equal_srctrg,
    get_single_track_ispecific,
    remove_wt_tokens,
)

__all__ = [
    "MusicDataset",
    "ToNumpy",
    "ToTensor",
    "trnsfrm_idx_to_note",
    "trnsfrm_note_to_idx",
    "pad_EB",
    "pad_EB_fixed",
    "split_tracks_src_trg",
    "CropEnding",
    # "TriGram",
    "get_single_track",
    "get_populated_single_track",
    "unsqueeze_list",
    "cap_list",
    "remove_tracks",
    "split_tracks_include_instrument",
    "split_tracks_equal_srctrg",
    "get_single_track_ispecific",
    "remove_wt_tokens",
]
