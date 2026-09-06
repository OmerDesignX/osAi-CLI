"""osAi framework: quantization-preserving training for MLX and GGUF."""

from .config import ModelFormat, TrainingConfig
from .errors import OsAiError
from .hardware import Accelerator, Engine

__all__ = ["Accelerator", "Engine", "ModelFormat", "OsAiError", "TrainingConfig"]
__version__ = "0.1.0"
