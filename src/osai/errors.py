"""Public exception hierarchy."""


class OsAiError(RuntimeError):
    """Base error with a concise, user-actionable message."""


class ConfigurationError(OsAiError):
    """The requested run is invalid or unsafe."""


class DependencyError(OsAiError):
    """A required optional runtime dependency is unavailable."""


class ModelFormatError(OsAiError):
    """A model is unsupported, incomplete, or mislabeled."""


class TrainingError(OsAiError):
    """A training process failed."""


class VerificationError(OsAiError):
    """An output failed an integrity or runtime check."""


class ModelDownloadError(OsAiError):
    """An official model could not be downloaded or activated safely."""
