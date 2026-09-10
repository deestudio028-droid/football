"""V1 model artifact: save/load for the trained production model (D-29).

No artifact convention existed in this repository before D-29 -- nothing
imported joblib, pickle or cloudpickle, and no model file was ever
written. This module defines ONE minimal convention so a second,
incompatible one is never invented alongside it.

WHY THIS IS ALLOWED NOW. Earlier phases carried a standing rule against
persisting an estimator, and Gates 5, 7 and 9 each assert that *those
test modules* import no serialiser and make no write call. Those
assertions are self-scoped: they constrain the gate scripts, not the
project. D-29 explicitly authorises persisting a trained V1 artifact,
which supersedes the earlier standing rule for this purpose. Nothing in
this module touches a frozen source, a pinned artifact, or a Gate test.

FORMAT. A single pickle holding a dict with five keys:

    model            fitted LogisticRegression from V1's frozen path
    preprocessor     fitted train.LogisticRegressionPreprocessor
    feature_columns  tuple, the exact 80-column contract in stored order
    model_version    "v1.0"
    class_order      ["H", "D", "A"]

stdlib `pickle` is used rather than joblib so the artifact carries no
dependency beyond what training already required. `save` refuses to
write anything whose contract does not match `ablation.MODEL_B_COLUMNS`
exactly, and `load` re-verifies on the way back in, so a drifted or
hand-edited artifact fails loudly instead of silently predicting with
the wrong feature order.

This module fits nothing and evaluates nothing.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ablation import MODEL_B_COLUMNS
from .baselines import CLASS_ORDER
from .config import MODEL_VERSION

#: Default location. `data/` already holds the project's other generated
#: artifacts (audit/, processed/, raw/, samples/), so a sibling directory
#: keeps the convention consistent. No "v2" appears in the name, per the
#: governance invariant asserted by nine test modules.
DEFAULT_ARTIFACT_PATH = Path("data/models/v1_logreg.pkl")

ARTIFACT_KEYS = ("model", "preprocessor", "feature_columns", "model_version", "class_order")


class ArtifactError(RuntimeError):
    """Raised rather than silently loading or writing an artifact whose
    contract does not match the frozen V1 definition."""


@dataclass(frozen=True)
class V1Artifact:
    model: Any
    preprocessor: Any
    feature_columns: tuple[str, ...]
    model_version: str
    class_order: list[str]


def _verify(feature_columns, model_version, class_order, where: str) -> None:
    cols = tuple(feature_columns)
    if cols != tuple(MODEL_B_COLUMNS):
        if set(cols) == set(MODEL_B_COLUMNS):
            raise ArtifactError(
                f"{where}: feature_columns match MODEL_B_COLUMNS as a set but the "
                "ORDER differs. Refusing to proceed -- column order is part of the "
                "contract and a reordered matrix would silently mispredict.")
        raise ArtifactError(
            f"{where}: feature_columns do not match ablation.MODEL_B_COLUMNS "
            f"({len(cols)} vs {len(MODEL_B_COLUMNS)} columns; "
            f"added={sorted(set(cols) - set(MODEL_B_COLUMNS))[:5]}, "
            f"removed={sorted(set(MODEL_B_COLUMNS) - set(cols))[:5]}).")
    if model_version != MODEL_VERSION:
        raise ArtifactError(
            f"{where}: model_version is {model_version!r}, expected {MODEL_VERSION!r}.")
    if list(class_order) != list(CLASS_ORDER):
        raise ArtifactError(
            f"{where}: class_order is {list(class_order)!r}, expected {list(CLASS_ORDER)!r}.")


def save(model: Any, preprocessor: Any, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    """Persist a fitted V1 model + preprocessor. The contract, version and
    class order are taken from the frozen sources, never from the caller,
    so an artifact cannot be written with a contract that was not V1's."""
    _verify(MODEL_B_COLUMNS, MODEL_VERSION, CLASS_ORDER, "save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": tuple(MODEL_B_COLUMNS),
        "model_version": MODEL_VERSION,
        "class_order": list(CLASS_ORDER),
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: Path = DEFAULT_ARTIFACT_PATH) -> V1Artifact:
    """Reload an artifact and re-verify its contract before returning it."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(
            f"no model artifact at {path}. Train one first: python train_v1_model.py")
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ArtifactError(f"load: artifact at {path} is not a dict payload.")
    missing = [k for k in ARTIFACT_KEYS if k not in payload]
    if missing:
        raise ArtifactError(f"load: artifact at {path} is missing key(s): {missing}.")
    _verify(payload["feature_columns"], payload["model_version"],
            payload["class_order"], "load")
    return V1Artifact(
        model=payload["model"],
        preprocessor=payload["preprocessor"],
        feature_columns=tuple(payload["feature_columns"]),
        model_version=payload["model_version"],
        class_order=list(payload["class_order"]),
    )


__all__ = ["DEFAULT_ARTIFACT_PATH", "ArtifactError", "V1Artifact", "save", "load"]
