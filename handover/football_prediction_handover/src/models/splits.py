"""Deterministic, chronological (never random) train/validation/test
splitting for the walk-forward strategy approved in
docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md §5-6.

Every split is defined by season membership (never by row index, never
by a random seed), and every split this module produces can be checked
by `verify_temporal_safety` for the two properties that actually matter:
no fixture appears in both sides of a split, and every training
fixture's kickoff time is strictly earlier than every evaluation
fixture's kickoff time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    FINAL_TEST_SEASONS,
    FINAL_TRAIN_SEASONS,
    SEASON_NAME_TO_IDS,
    WALK_FORWARD_FOLDS,
    Fold,
)
from .data import SupervisedDataset


class SplitSafetyError(RuntimeError):
    """Raised when a split would violate temporal or overlap safety.
    This is deliberately a hard failure, not a warning -- a caller
    should never be able to silently proceed with an unsafe split.
    """


def season_ids_for(season_names: tuple[str, ...]) -> set[int]:
    missing = [s for s in season_names if s not in SEASON_NAME_TO_IDS]
    if missing:
        raise KeyError(f"Unknown season name(s), not found in config/competitions.json: {missing}")
    ids: set[int] = set()
    for name in season_names:
        ids.update(SEASON_NAME_TO_IDS[name])
    return ids


def _subset_by_season_ids(dataset: SupervisedDataset, season_ids: set[int]) -> SupervisedDataset:
    mask = dataset.metadata["season_id"].isin(season_ids)
    return SupervisedDataset(
        metadata=dataset.metadata.loc[mask].reset_index(drop=True),
        X=dataset.X.loc[mask].reset_index(drop=True),
        y=dataset.y.loc[mask].reset_index(drop=True),
    )


def split_by_seasons(
    dataset: SupervisedDataset, train_seasons: tuple[str, ...], eval_seasons: tuple[str, ...]
) -> tuple[SupervisedDataset, SupervisedDataset]:
    train = _subset_by_season_ids(dataset, season_ids_for(train_seasons))
    evald = _subset_by_season_ids(dataset, season_ids_for(eval_seasons))
    return train, evald


def verify_temporal_safety(train: SupervisedDataset, evald: SupervisedDataset) -> None:
    """Raises SplitSafetyError if the split is unsafe. Checks:
      1. max(train.unix) < min(eval.unix)
      2. no fixture_id appears in both train and eval
    Both are checked (and reported together if both fail) rather than
    stopping at the first problem, so a caller sees the full picture.
    """
    problems: list[str] = []

    if len(train) > 0 and len(evald) > 0:
        max_train_unix = train.metadata["unix"].max()
        min_eval_unix = evald.metadata["unix"].min()
        if not (max_train_unix < min_eval_unix):
            problems.append(
                f"max(train.unix)={max_train_unix} is not strictly before "
                f"min(eval.unix)={min_eval_unix}"
            )

    overlap = set(train.metadata["fixture_id"]) & set(evald.metadata["fixture_id"])
    if overlap:
        problems.append(f"{len(overlap)} fixture_id(s) appear in both train and eval: {sorted(overlap)[:10]}")

    if problems:
        raise SplitSafetyError("; ".join(problems))


def iter_walk_forward_folds(dataset: SupervisedDataset):
    """Yields (Fold, train_subset, validation_subset) for each of the 3
    approved walk-forward folds, each already verified temporally safe.
    2025/26 never appears in any train_seasons/validation_seasons here
    -- see config.WALK_FORWARD_FOLDS, which structurally excludes it.
    """
    for fold in WALK_FORWARD_FOLDS:
        train, val = split_by_seasons(dataset, fold.train_seasons, fold.validation_seasons)
        verify_temporal_safety(train, val)
        yield fold, train, val


def final_split(dataset: SupervisedDataset) -> tuple[SupervisedDataset, SupervisedDataset]:
    """The untouched final train/test split: train on everything through
    2024/25, test on 2025/26 only. Verified temporally safe before being
    returned, same as every walk-forward fold.
    """
    train, test = split_by_seasons(dataset, FINAL_TRAIN_SEASONS, FINAL_TEST_SEASONS)
    verify_temporal_safety(train, test)
    return train, test
