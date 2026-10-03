"""
Load the frozen M3 models across scikit-learn versions.

The frozen M3 pipelines were serialized with scikit-learn 1.6.1. In scikit-learn >= 1.7,
SimpleImputer renamed the private attribute `_fill_dtype` to `_fit_dtype`, so the
unpickled imputer fails at predict time. This loader applies the compatibility patch
recorded in the pre-test verification (gate 2): copy `_fit_dtype` <-> `_fill_dtype`.
That gate verified numerical equivalence against scikit-learn 1.6.1 on the validation
features (max |diff| = 2.2e-16). Under scikit-learn 1.6.1 the patch is a no-op.
"""
import warnings
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer


def _patch(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
            obj._fill_dtype = obj._fit_dtype
        if not hasattr(obj, "_fit_dtype") and hasattr(obj, "_fill_dtype"):
            obj._fit_dtype = obj._fill_dtype
        return
    children = []
    if hasattr(obj, "steps"):                      # Pipeline
        children += [step for _, step in obj.steps]
    for attr in ("calibrated_classifiers_", "estimators_"):
        val = getattr(obj, attr, None)
        if isinstance(val, list):
            children += val
    for attr in ("estimator", "base_estimator"):
        val = getattr(obj, attr, None)
        if val is not None and not isinstance(val, str):
            children.append(val)
    for child in children:
        _patch(child, seen)


def load_m3_model(path):
    """joblib.load + scikit-learn version compatibility patch for SimpleImputer."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        model = joblib.load(Path(path))
    _patch(model, set())
    return model
