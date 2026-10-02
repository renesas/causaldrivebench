"""
evaluation/common/utils/errors.py

Custom errors for the evaluation pipeline.

Errors represent *programming mistakes or unrecoverable system failures* —
conditions the pipeline does NOT expect during normal operation and cannot
meaningfully recover from (e.g. a tensor with the wrong shape passed to the
model, a required environment variable that was never set, an input type that
is fundamentally wrong).

Hierarchy
---------
EvaluationError                  ← base; all pipeline errors inherit from here
├── ConfigurationError           ← setup / environment issues
│   └── MissingEnvVarError       ← a required environment variable is not set
├── TensorError                  ← tensor shape, dtype, or device mismatches
│   └── TensorShapeError         ← tensor has wrong number of dims or axis size
└── InvalidInputError            ← caller passed an argument of the wrong type
                                    or with an invalid value

When to add more errors
-----------------------
Add a new error when:
  1. There is a distinct *non-recoverable* failure mode that should crash the
     pipeline immediately and clearly (no silent fallback).
  2. You want to carry structured context to the developer (attach the bad
     object via `obj=` so inspect_object prints it at raise time).

Prefer subclassing the most-specific existing error over adding a new
top-level one. Keep the hierarchy shallow (at most two levels below
EvaluationError).

Usage
-----
    from evaluation.common.utils.errors import TensorShapeError

    # Raise with a plain message
    raise TensorShapeError("Expected (N, C, H, W), got (C, H, W)")

    # Raise and auto-inspect a context object (printed to stdout)
    raise TensorShapeError(
        f"Bad image tensor shape: {images.shape}",
        obj=images,       # inspect_object(images) is called immediately
    )

    # Distinguish errors from exceptions at the call site
    try:
        validate_tensor(images)
    except TensorShapeError:
        raise   # always re-raise errors — they are not recoverable
"""

from evaluation.common.utils.debugging_tools import inspect_object


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class EvaluationError(Exception):
    """Base class for all evaluation-pipeline errors.

    Parameters
    ----------
    message : str, optional
        Human-readable description of the problem.
    obj : object, optional
        Any Python object relevant to the failure.  When provided,
        inspect_object(obj) is called immediately so its structure is
        visible in the terminal alongside the traceback.
    """

    def __init__(self, message: str = "", obj=None):
        if obj is not None:
            # Print the object's internals right away so the developer can
            # see exactly what went wrong without attaching a debugger.
            print(f"\n[{type(self).__name__}] Inspecting attached object:")
            inspect_object(obj)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(EvaluationError):
    """Raised when the pipeline environment or config is incorrectly set up."""


class MissingEnvVarError(ConfigurationError):
    """Raised when a required environment variable is absent.

    Use for external-service credentials (API keys, tokens) and similar
    secrets that cannot live in eval_config.yaml.  All path configuration
    goes through CDBConfig / eval_config.yaml instead.

    Example
    -------
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise MissingEnvVarError("ANTHROPIC_API_KEY environment variable is not set")
    """


# ---------------------------------------------------------------------------
# Tensor errors
# ---------------------------------------------------------------------------

class TensorError(EvaluationError):
    """Raised for any tensor shape, dtype, or device mismatch."""


class TensorShapeError(TensorError):
    """Raised when a tensor has the wrong number of dimensions or axis size.

    The Alpamayo helper requires image tensors of shape (N, C, H, W) with
    dtype uint8.  Use this error whenever that contract is violated.

    Example
    -------
        if images.ndim != 4:
            raise TensorShapeError(
                f"Expected 4-D tensor (N,C,H,W), got shape {tuple(images.shape)}",
                obj=images,
            )
    """


# ---------------------------------------------------------------------------
# Input errors
# ---------------------------------------------------------------------------

class InvalidInputError(EvaluationError):
    """Raised when a function receives an argument of the wrong type or value.

    Example
    -------
        if not isinstance(scene_id, str):
            raise InvalidInputError(
                f"scene_id must be str, got {type(scene_id).__name__}",
                obj=scene_id,
            )
    """
