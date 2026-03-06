from astropy.coordinates import EarthLocation
import astropy.units as u

BASE_FREQ = 1.398e9
BANDWIDTH = 62.5e6
NCHAN = 1024

LOCATION = EarthLocation(
    lat=43.933*u.deg,
    lon=5.7153*u.deg,
    height=654.8*u.m
)

# convenience imports
from .pipeline import GradPipeline  # noqa: E402, F401
from . import io, spectrum, tpi, plotting, utils  # re-export submodules
