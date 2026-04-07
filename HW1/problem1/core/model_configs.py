"""Per-mode parameter specs and validation rules for DataReuploadingRegressor.

Each entry in PARAM_SPECS maps a parameter name to a factory ``lambda L``
that returns the initial tensor for that parameter given ``num_layers=L``.
Scalar parameters ignore ``L``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

_NEG_PI_2 = -math.pi / 2.0

# ── validation ────────────────────────────────────────────────────────────────

# Modes that require exactly 2 qubits (all others require 1).
MODES_REQUIRING_2_QUBITS: frozenset[str] = frozenset({
    "twoqubit_no_reupload",
    "twoqubit_raw_no_reupload",
    "same_axis_twoqubit",
})

# Modes that accept num_layers > 1.
MULTI_LAYER_MODES: frozenset[str] = frozenset({
    "same_axis_reupload",
    "same_axis_raw",
    "same_axis_twoqubit",
    "same_axis_poly",
    "same_axis_rot",
})

# ── parameter specs ───────────────────────────────────────────────────────────

# Shared base for all same-axis modes (per-layer scale / bias / phase).
_SAME_AXIS_BASE: dict[str, Callable[[int], torch.Tensor]] = {
    "phase_shifts": lambda L: torch.full((L,), _NEG_PI_2),
    "exp_scales":   lambda L: torch.ones(L),
    "exp_biases":   lambda L: torch.zeros(L),
    "x2_scales":    lambda L: torch.ones(L),
    "x2_biases":    lambda L: torch.zeros(L),
}

# Additional residual params used only by same_axis_twoqubit (q1 branch).
_TWOQUBIT_RESIDUAL: dict[str, Callable[[int], torch.Tensor]] = {
    "residual_phase_shifts": lambda L: torch.zeros(L),
    "residual_exp_scales":   lambda L: torch.ones(L),
    "residual_exp_biases":   lambda L: torch.zeros(L),
    "residual_x2_scales":    lambda L: torch.ones(L),
    "residual_x2_biases":    lambda L: torch.zeros(L),
    "readout_weights":       lambda _: torch.tensor([1.0, 0.0]),
    "readout_bias":          lambda _: torch.tensor(0.0),
}

PARAM_SPECS: dict[str, dict[str, Callable[[int], torch.Tensor]]] = {
    "quantum_exact": {},

    "twoqubit_no_reupload": {},

    "phase_learnable": {
        "phase_shift": lambda _: torch.tensor(_NEG_PI_2),
    },

    "scaled_exact": {
        "phase_shift": lambda _: torch.tensor(_NEG_PI_2),
        "exp_scale":   lambda _: torch.tensor(1.0),
        "exp_bias":    lambda _: torch.tensor(0.0),
        "x2_scale":    lambda _: torch.tensor(1.0),
        "x2_bias":     lambda _: torch.tensor(0.0),
    },

    "same_axis_reupload": {**_SAME_AXIS_BASE},

    "same_axis_raw": {**_SAME_AXIS_BASE},

    "same_axis_poly": {
        **_SAME_AXIS_BASE,
        # Each layer gets a degree-3 polynomial [c0, c1, c2, c3]; default = identity (x).
        "poly_coefficients": lambda L: torch.tensor([[0.0, 1.0, 0.0, 0.0]] * L),
    },

    "same_axis_rot": {
        **_SAME_AXIS_BASE,
        "block_rotations": lambda L: torch.zeros(L, 3),
    },

    "same_axis_twoqubit": {
        **_SAME_AXIS_BASE,
        **_TWOQUBIT_RESIDUAL,
    },

    "twoqubit_raw_no_reupload": {
        "twoqubit_rotations":      lambda _: torch.zeros(2, 3),
        "twoqubit_readout_weights": lambda _: torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "twoqubit_readout_bias":   lambda _: torch.tensor(0.0),
    },
}
