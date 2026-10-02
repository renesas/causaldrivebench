"""
evaluation/common/utils/exceptions.py

Custom exceptions for the evaluation pipeline.

Exceptions represent *expected, recoverable* conditions — situations the
pipeline anticipates and can potentially handle with a fallback or a retry
(e.g. a missing image file that gets replaced by a placeholder, a model
that isn't loaded yet, a QA entry with no valid response).

Hierarchy
---------
EvaluationException              ← base; all pipeline exceptions inherit from here
├── DataException                ← data loading / parsing issues
│   ├── MissingFrameException    ← one or more camera frames could not be found
│   └── InvalidManifestException ← manifest.json is missing required keys
├── InferenceException           ← model inference issues
│   └── ModelNotReadyException   ← model / processor not yet loaded
└── PostprocessException         ← response parsing / metric computation issues
    └── EmptyResponseException   ← model returned an empty or null response

When to add more exceptions
---------------------------
Add a new exception when:
  1. There is a distinct *recoverable* failure mode that callers need to
     distinguish from other failures (e.g. catch MissingFrameException
     specifically to insert a placeholder and continue).
  2. The exception carries meaningful context beyond a plain message
     (attach the offending object via `obj=` so inspect_object prints it).

Prefer subclassing the most-specific existing exception over adding a new
top-level one. Keep the hierarchy shallow (at most two levels below
EvaluationException) to avoid unnecessary complexity.

Usage
-----
    from evaluation.common.utils.exceptions import MissingFrameException

    # Raise with a plain message
    raise MissingFrameException("Frame 5 not found for scene 0001")

    # Raise and auto-inspect a context object (printed to stdout)
    raise MissingFrameException(
        "Unexpected frame structure",
        obj=frame_dict,          # inspect_object(frame_dict) is called
    )

    # Catch a specific exception
    try:
        load_frame(path)
    except MissingFrameException as exc:
        logger.warning(exc)
        return placeholder_tensor()
"""

from evaluation.common.utils.debugging_tools import inspect_object


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class EvaluationException(Exception):
    """Base class for all evaluation-pipeline exceptions.

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
            # see exactly what triggered the exception.
            print(f"\n[{type(self).__name__}] Inspecting attached object:")
            inspect_object(obj)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Data exceptions
# ---------------------------------------------------------------------------

class DataException(EvaluationException):
    """Raised for failures during data loading or parsing."""


class MissingFrameException(DataException):
    """Raised when one or more camera frames cannot be located on disk.

    Typical recovery: replace the missing frame with a grey placeholder
    tensor and log a warning before continuing.

    Example
    -------
        raise MissingFrameException(
            f"Image not found: {img_path}",
            obj=frame_meta,   # attach the frame metadata dict for inspection
        )
    """


class InvalidManifestException(DataException):
    """Raised when manifest.json is missing required keys or is malformed.

    Example
    -------
        raise InvalidManifestException(
            "Key 'scenes' missing from manifest",
            obj=manifest_dict,
        )
    """


# ---------------------------------------------------------------------------
# Inference exceptions
# ---------------------------------------------------------------------------

class InferenceException(EvaluationException):
    """Raised for failures that occur during model inference."""


class ModelNotReadyException(InferenceException):
    """Raised when inference is attempted before the model/processor is loaded.

    Example
    -------
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_inference()")
    """


# ---------------------------------------------------------------------------
# Postprocess exceptions
# ---------------------------------------------------------------------------

class PostprocessException(EvaluationException):
    """Raised for failures during response parsing or metric computation."""


class EmptyResponseException(PostprocessException):
    """Raised when the model returns an empty or null CoC/answer string.

    Typical recovery: skip this sample and record a zero score.

    Example
    -------
        if not coc_text.strip():
            raise EmptyResponseException(
                f"Empty response for QA id={qa_id}",
                obj=raw_output,
            )
    """
