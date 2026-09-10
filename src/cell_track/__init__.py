"""Shared analysis code for the cell-track notebooks and Nextflow pipeline.

Stage-specific dependencies are imported lazily by their respective modules so
that BaSiCPy and Cellpose/Ultrack can remain in separate environments.
"""

__version__ = "0.1.0"
