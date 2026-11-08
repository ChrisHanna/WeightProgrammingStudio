"""Generic weight editing with evidence-based compatibility."""

from .loader import LoadedModel, UnsupportedModel, WeightEditInputError, load_model

__all__ = ["LoadedModel", "UnsupportedModel", "WeightEditInputError", "load_model"]