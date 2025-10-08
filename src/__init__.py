from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "constants",
    "data_loader",
    "data_processing",
    "viz",
]

try:
    __version__ = version("soil_health_explorer")
except PackageNotFoundError:
    __version__ = "0.0.0"
