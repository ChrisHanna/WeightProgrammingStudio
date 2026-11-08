"""Generic weight editing with evidence-based compatibility."""

from .loader import LoadedModel, UnsupportedModel, load_model

__all__ = ["LoadedModel", "UnsupportedModel", "load_model"]