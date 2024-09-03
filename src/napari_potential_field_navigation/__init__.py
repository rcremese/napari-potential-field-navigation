try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._sample_data import liver_sample, lung_sample
from ._widget import DiffApfWidget

__all__ = (
    "liver_sample",
    "lung_sample",
    "DiffApfWidget",
)
