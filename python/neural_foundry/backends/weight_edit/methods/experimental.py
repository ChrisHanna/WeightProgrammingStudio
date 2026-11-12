"""Experimental, single-site tensor solver; not a live-job qualification policy."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from ..loader import WeightEditInputError
from .base import Delta, _matrix

METHOD_ID = "value_scaled_retained_edit_v1"
SETTING_NAMES = {
    "strength", "ridge_alpha", "retention_beta", "protection_epsilon",
    "protection_zero_absolute", "rank_rtol", "rank_atol", "solver_atol", "solver_max_iterations",
}


def validate_settings(settings: Mapping) -> None:
    if not isinstance(settings, Mapping) or set(settings) != SETTING_NAMES:
        raise WeightEditInputError("experimental solver requires all explicit calibration settings")
    for name in SETTING_NAMES - {"solver_max_iterations"}:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise WeightEditInputError(f"invalid experimental {name}")
    if settings["strength"] <= 0 or settings["ridge_alpha"] <= 0 or settings["solver_atol"] <= 0:
        raise WeightEditInputError("strength, ridge_alpha and solver_atol must be positive")
    if not 0 <= settings["rank_rtol"] < 1:
        raise WeightEditInputError("rank_rtol must be below one")
    if type(settings["solver_max_iterations"]) is not int or not 1 <= settings["solver_max_iterations"] <= 4096:
        raise WeightEditInputError("solver iteration budget must be explicit and bounded")


class ValueScaledRetainedEdit:
    method_id = METHOD_ID
    supported_roles = ("residual_write_mlp",)

    def __init__(self, retention, protected_keys: torch.Tensor, write_outputs: torch.Tensor):
        self.retention = retention
        self.protected = _matrix(protected_keys, "protected keys").double()
        self.outputs = _matrix(write_outputs, "write outputs").double()
        self.diagnostics: dict = {}

    def compute(self, site, W, keys, targets, settings) -> Delta:
        """Here targets are raw value-loss gradients, not absolute writer targets."""
        try:
            return self._compute(site, W, keys, targets, settings)
        except torch.linalg.LinAlgError as exc:
            raise WeightEditInputError("experimental protected solve failed numerically; no fallback is permitted") from exc

    def _compute(self, site, W, keys, targets, settings) -> Delta:
        validate_settings(settings)
        W, rows, G = (_matrix(x, name).double() for x, name in
                      ((W, "weights"), (keys, "keys"), (targets, "raw gradients")))
        P = self.protected.T
        if (site.role not in self.supported_roles or W.shape != (site.output_size, site.input_size)
            or rows.shape[1] != site.input_size or not rows.shape[0]
            or G.shape != (rows.shape[0], site.output_size) or self.outputs.shape != G.shape
            or P.shape[0] != site.input_size or not P.shape[1]):
            raise WeightEditInputError("experimental matrices require nonempty canonical site-aligned keys")
        stat = self.retention
        if type(stat.count) is not int or stat.count <= 0:
            raise WeightEditInputError("all-token retention statistics must be nonempty")
        C = _matrix(stat.input_gram, "retention Gram").double() / stat.count
        if C.shape != (site.input_size, site.input_size) or not torch.allclose(C, C.T):
            raise WeightEditInputError("retention second moment must be symmetric and site-aligned")
        C = (C + C.T) / 2
        tolerance = settings["solver_atol"]
        self._tolerance = tolerance
        if float(torch.linalg.eigvalsh(C).min()) < -tolerance:
            raise WeightEditInputError("retention second moment must be positive semidefinite")
        K = rows.T
        gram = K @ K.T
        diagonal_scale = float(gram.diag().mean())
        h_rms, g_rms = float(self.outputs.square().mean().sqrt()), float(G.square().mean().sqrt())
        if any(not math.isfinite(x) or x <= 0 for x in (diagonal_scale, h_rms, g_rms)):
            raise WeightEditInputError("Gram, activation and gradient scales must be finite and positive")
        target_scale = settings["strength"] * h_rms / g_rms
        Y = -target_scale * G.T
        ridge = settings["ridge_alpha"] * diagonal_scale
        if not math.isfinite(target_scale) or target_scale <= 0 or not math.isfinite(ridge) or ridge <= 0:
            raise WeightEditInputError("invalid scaled target or ridge")
        A = gram + settings["retention_beta"] * C + ridge * torch.eye(site.input_size, dtype=torch.float64)
        B = Y @ K.T
        reference = float(torch.linalg.vector_norm(W @ P))
        bound = (settings["protection_epsilon"] * reference if reference > 0
                 else settings["protection_zero_absolute"])
        if not math.isfinite(bound):
            raise WeightEditInputError("nonfinite protection bound")
        U, singular, _ = torch.linalg.svd(P, full_matrices=True)
        cutoff = max(settings["rank_atol"], settings["rank_rtol"] * float(singular.max()))
        rank = int((singular > cutoff).sum())
        if bound == 0:
            N = U[:, rank:]
            D = (torch.linalg.solve(N.T @ A @ N, (B @ N).T).T @ N.T
                 if N.shape[1] else torch.zeros_like(W))
        else:
            D = torch.linalg.solve(A, B.T).T
            if float(torch.linalg.vector_norm(D @ P)) > bound:
                # The zero update is feasible. A bounded multiplier search must find
                # and verify a feasible point; numerical failure never drops P.
                PP = P @ P.T
                low, high = 0.0, 1.0
                for _ in range(settings["solver_max_iterations"]):
                    candidate = torch.linalg.solve(A + high * PP, B.T).T
                    if float(torch.linalg.vector_norm(candidate @ P)) <= bound:
                        D = candidate
                        break
                    high *= 2
                else:
                    raise WeightEditInputError("protected constraint multiplier search failed")
                for _ in range(settings["solver_max_iterations"]):
                    middle = (low + high) / 2
                    candidate = torch.linalg.solve(A + middle * PP, B.T).T
                    if float(torch.linalg.vector_norm(candidate @ P)) <= bound:
                        high, D = middle, candidate
                    else:
                        low = middle
                if bound - float(torch.linalg.vector_norm(D @ P)) > tolerance:
                    raise WeightEditInputError("protected constraint solve did not converge within its explicit budget")
        if not bool(torch.isfinite(D).all()) or float(torch.linalg.vector_norm(D @ P)) > bound + tolerance:
            raise WeightEditInputError("protected constraint infeasible at declared rank/tolerance")
        effective = (W + D).to(torch.bfloat16).double() - W.to(torch.bfloat16).double()
        spectrum = torch.linalg.eigvalsh(A)
        if float(spectrum.min()) <= 0 or not math.isfinite(float(spectrum.max() / spectrum.min())):
            raise WeightEditInputError("experimental system is not numerically positive definite")
        norms = G.norm(dim=1)
        activation_norms = self.outputs.norm(dim=1)
        positive = activation_norms > 0
        fractions = Y.T.norm(dim=1)[positive] / activation_norms[positive]
        self.diagnostics = {
            "status": "experimental_cpu_correctness_only", "key_layout": "K[input_dimension,key_count]",
            "zero_effective_bf16_update": not bool(torch.count_nonzero(effective)),
            "no_task_gain_or_preservation_claim": True,
            "loss_reduction": "global_mean_over_selected_value_tokens",
            "ridge_alpha": settings["ridge_alpha"], "mean_gram_diagonal": diagonal_scale, "ridge": ridge,
            "system_min_eigenvalue": float(spectrum.min()), "system_condition": float(spectrum.max() / spectrum.min()),
            "activation_rms": h_rms, "gradient_rms": g_rms, "global_target_scale": target_scale,
            "target_rms": float(Y.square().mean().sqrt()), "zero_gradient_keys": int((norms == 0).sum()),
            "raw_gradient_norm_quantiles": torch.quantile(norms, torch.tensor([0., .5, 1.], dtype=torch.float64)).tolist(),
            "per_key_target_write_fraction_quantiles": torch.quantile(
                fractions, torch.tensor([0., .5, 1.], dtype=torch.float64)).tolist(),
            "zero_per_key_activation_norms": int((~positive).sum()),
            "activation_scope": "writer_output_at_value_prediction_positions_not_residual_stream",
            "edit_gram_reduction": "sum_over_edit_keys",
            "retention_token_count": stat.count, "retention_reduction": "token_weighted_uncentered_second_moment",
            "protected_rank": rank, "protected_key_count": P.shape[1], "protected_reference_norm": reference,
            "protected_bound": bound, "protected_zero_reference_policy": "explicit_absolute_bound",
            "protected_covariance_weighting": "prompt tokens once in C; response value positions constraint-only; no extra covariance weight",
        }
        self.diagnostics.update(self.verify_delta(D, W, K, Y, C, P, "tensor"))
        self.diagnostics.update(self.verify_delta(effective, W, K, Y, C, P, "bf16"))
        if not all(math.isfinite(value) for value in self.diagnostics.values() if isinstance(value, float)):
            raise WeightEditInputError("nonfinite experimental diagnostics")
        return Delta("dense", D=D)

    def verify_delta(self, D, W, K, Y, C, P, prefix):
        residual = float(torch.linalg.vector_norm(D @ P))
        # Cast/reload is an independent feasibility gate, not evidence that local
        # response protection implies exact-output preservation.
        if not bool(torch.isfinite(D).all()) or residual > self.diagnostics["protected_bound"] + self._tolerance:
            raise WeightEditInputError(f"{prefix} protected-response residual {residual} violates "
                                       f"bound {self.diagnostics['protected_bound']} plus solver_atol {self._tolerance}")
        return {
            f"{prefix}_target_fit_residual": float(torch.linalg.vector_norm(D @ K - Y)),
            f"{prefix}_retention_penalty": float(torch.trace(D @ C @ D.T)),
            f"{prefix}_protected_residual": residual,
            f"{prefix}_delta_norm": float(torch.linalg.vector_norm(D)),
            f"{prefix}_realized_write_activation_ratio": float((D @ K).square().mean().sqrt()) / self.diagnostics["activation_rms"],
        }
