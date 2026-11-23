from importlib.metadata import PackageNotFoundError, version

from .core.gpt import GPTModel

try:
    __version__ = version("myllm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["GPTModel", "__version__"]
