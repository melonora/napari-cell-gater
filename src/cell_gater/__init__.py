from importlib.metadata import version  # Python = 3.9

__version__ = version("napari-cell-gater")

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse

from cell_gater.widgets.sample_widget import SampleWidget  # noqa: E402
