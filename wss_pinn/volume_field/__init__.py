"""Active peak-volume ``(u, v, w, p)`` physics-informed route.

The historical WSS-target PINN remains under :mod:`wss_pinn` for provenance.
New development lives in this isolated subpackage so old checkpoints and
configuration files keep their original import surface.
"""

from .config import VolumeExperimentConfig

__all__ = ["VolumeExperimentConfig"]
