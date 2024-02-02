try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._sample_data import open_lung_samples
from ._widget import DiffApfWidget


__all__ = (
    "open_lung_samples",
    "DiffApfWidget",
)
