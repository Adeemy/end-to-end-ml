"""Config-driven model factory.

Resolves an estimator class from its importable dotted path and instantiates it,
so the set of trainable models is defined entirely in the training-config
``models:`` list (no per-model code). ``build_estimator`` also substitutes
``"${name}"`` placeholders in fixed params with values computed at runtime from
the data (e.g. XGBoost's ``scale_pos_weight``).
"""

import importlib
from typing import Any, Dict, Optional, Type


def resolve_estimator_class(estimator_path: str) -> Type:
    """Imports and returns the estimator class named by a dotted path.

    Args:
        estimator_path: Full importable class path, e.g.
            ``"sklearn.linear_model.LogisticRegression"`` or
            ``"lightgbm.LGBMClassifier"``.

    Returns:
        The estimator class object.

    Raises:
        ValueError: If ``estimator_path`` is not a dotted path, its module cannot
            be imported, or the class is absent from the module.
    """
    module_path, _, class_name = estimator_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Estimator '{estimator_path}' must be a full dotted class path "
            "(e.g. 'lightgbm.LGBMClassifier')."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Could not import module '{module_path}' for estimator "
            f"'{estimator_path}'. Is the package installed?"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module_path}'."
        ) from exc


def _resolve_runtime_params(
    params: Dict[str, Any], runtime_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Substitutes ``"${name}"`` placeholder param values with runtime values.

    A fixed-param value of the exact form ``"${scale_pos_weight}"`` is replaced
    with ``runtime_params["scale_pos_weight"]``; every other value passes through
    unchanged. This lets a config reference a data-derived value (computed once
    per run) without hardcoding it per model.

    Args:
        params: Fixed estimator kwargs from config.
        runtime_params: Available runtime values keyed by name.

    Returns:
        A new dict with placeholders resolved.

    Raises:
        ValueError: If a placeholder names a runtime value that was not provided.
    """
    resolved: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            name = value[2:-1]
            if name not in runtime_params:
                raise ValueError(
                    f"Param '{key}' references unknown runtime value '{name}'. "
                    f"Available runtime values: {sorted(runtime_params)}."
                )
            resolved[key] = runtime_params[name]
        else:
            resolved[key] = value
    return resolved


def build_estimator(
    estimator_path: str,
    params: Optional[Dict[str, Any]] = None,
    runtime_params: Optional[Dict[str, Any]] = None,
):
    """Builds an estimator instance from its dotted class path and fixed params.

    Args:
        estimator_path: Importable estimator class path (see
            ``resolve_estimator_class``).
        params: Fixed estimator kwargs; ``"${name}"`` values are resolved from
            ``runtime_params``.
        runtime_params: Runtime values available to ``"${name}"`` placeholders.

    Returns:
        An instantiated, unfitted estimator.
    """
    estimator_class = resolve_estimator_class(estimator_path)
    resolved_params = _resolve_runtime_params(params or {}, runtime_params or {})
    return estimator_class(**resolved_params)
