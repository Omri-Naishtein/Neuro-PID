"""
NeuroPID — Adaptive PID controller with embedded neural gain tuning.

A single-file, zero-dependency library that wraps a classic PID loop with a
tiny MLP that adjusts Kp/Ki/Kd in real time based on recent error history.
The network learns online via SGD so the controller improves as it runs.

Works everywhere Python runs: CPython, MicroPython, Jetson, ESP32, Arduino.
NumPy is used automatically when available but is never required.

Fixes applied
─────────────
  1. Architecture mismatch on load now raises a clear ValueError instead of
     silently returning False.
  2. Online learning is skipped on the very first step so the one-step-
     delayed reward signal never punishes a cold-start.
  3. Gain-smoothing alpha is now a constructor parameter (gain_alpha).
  4. dt fallback is now a constructor parameter (default_dt) instead of the
     hard-coded 0.02 s.
  5. Back-calculation anti-windup: when the output is saturated the integral
     is wound back proportionally, preventing runaway windup.
  6. time.monotonic() is wrapped in a safe fallback for MicroPython builds
     that only expose time.ticks_ms(), with proper ticks_diff for wrap
     safety.
  7. A small gradient floor (EPS_GRAD) prevents gradient death when MLP
     outputs are near zero at initialisation.
  8. The frozen flag is persisted in the save file (new magic "NPD2"); old
     "NMLP" files are still readable via _TinyMLP.load() directly.
  9. max_output parameter (0.0-1.0) caps the final output to a fraction
     of out_min/out_max so motors don't run at full power from a cold
     start (default 0.6 = 60%). Applied AFTER anti-windup so integral
     correction stays accurate against the real hardware limits.
 10. Dead input features removed — the two trailing zeros in the feature
     vector are replaced with dt and the current setpoint error ratio.
 11. input_dim is derived from the actual feature list length, with an
     assertion to catch mismatches.
 12. dt is capped at max_dt (default 10× default_dt) to prevent integral /
     derivative spikes from scheduling stalls or GC pauses.
 13. derivative_smoothing renamed to derivative_alpha for clarity (higher
     value = more responsive, matching standard EMA convention).
 14. gains() returns a namedtuple for readability.
 15. Constructor validates out_min < out_max and gain range ordering.
 16. max_output is persisted in the NPD3 save format.
 17. Physics-based zeta (ζ) optimisation: the neural reward now includes a
     damping-ratio convergence term so the MLP steers gains toward a
     target ζ.  Two plant models are supported ("mass" and "dc_motor")
     and the user can set target_zeta, system_type, and plant parameters
     from the constructor.

     ─── What is ζ (zeta)? ───────────────────────────────────────────
     The damping ratio ζ describes how oscillatory a second-order system
     is after a disturbance:

       ζ < 1.0  →  Underdamped : the system overshoots and oscillates
                    before settling.  Lower values mean more ringing.
                    ζ = 0 is an undamped oscillator (perpetual bounce).

       ζ = 1.0  →  Critically damped : fastest settling with zero
                    overshoot.  The "ideal" for many textbook systems.

       ζ > 1.0  →  Overdamped : no overshoot, but the response is
                    sluggish — the system approaches the target slowly.

     For robotics and motion control, ζ ≈ 0.7–1.0 is the sweet spot:
       • ζ = 0.7  gives ~5 % overshoot but fast rise time — good for
         agile platforms that can tolerate a small bump.
       • ζ = 0.9  gives virtually no overshoot with brisk response.
       • ζ = 1.0  is critically damped — fastest settling with zero
                    overshoot. This is the default in this library.

     The neural network now receives a reward penalty proportional to
     |target_ζ − current_ζ|, so over time it learns gain combinations
     that not only reduce tracking error but also keep the closed-loop
     dynamics in the desired damping regime.

Typical usage
─────────────
    from neuropid import PID

    # ── Differential-drive robot turn PID ──────────────────────────
    turn_pid = PID(
        kp=6.0, ki=0.3, kd=2.0,
        out_min=-75, out_max=75,
        gain_limits={
            "kp": (1.0, 15.0),
            "ki": (0.05, 2.0),
            "kd": (0.5, 5.0),
        },
        profile="low",
        gain_alpha=0.03,
        lr=0.002,
        system_type="dc_motor",
        motor_gain=0.3,      # tune to your motor
        time_constant=0.15,
        target_zeta=1.0,
        zeta_weight=0.5,
    )

    # ── Differential-drive robot drive PID ─────────────────────────
    drive_pid = PID(
        kp=8.0, ki=0.5, kd=1.5,
        out_min=-75, out_max=75,
        gain_limits={
            "kp": (2.0, 20.0),
            "ki": (0.1, 3.0),
            "kd": (0.2, 4.0),
        },
        profile="low",
        gain_alpha=0.03,
        lr=0.002,
        system_type="dc_motor",
        motor_gain=0.3,
        time_constant=0.15,
        target_zeta=1.0,
        zeta_weight=0.5,
    )

    while True:
        measurement = read_sensor()
        error = target - measurement
        output = pid.step(error, setpoint=target)
        motor.write(output)

    turn_pid.save("turn_weights.bin")
    drive_pid.save("drive_weights.bin")
"""

from __future__ import annotations

import math
import random
import struct
from collections import namedtuple

# ── Fix 14: named gains tuple ───────────────────────────────────────────────
Gains = namedtuple("Gains", ["kp", "ki", "kd"])


# ── Fix 6: portable monotonic clock with ticks_diff wrap safety ─────────────
try:
    import time as _time

    _monotonic = _time.monotonic
    _monotonic()  # smoke-test

    def _elapsed(prev: float, now: float) -> float:
        """Simple subtraction — monotonic doesn't wrap on CPython."""
        return now - prev

except AttributeError:
    import time as _time  # noqa: F811

    _HAS_TICKS_DIFF = hasattr(_time, "ticks_diff")

    def _monotonic() -> float:          # type: ignore[misc]
        return _time.ticks_ms() / 1000.0

    def _elapsed(prev: float, now: float) -> float:
        """Wrap-safe elapsed time for MicroPython ticks_ms."""
        if _HAS_TICKS_DIFF:
            # ticks_diff handles 32-bit wrap (~24.8 days)
            return _time.ticks_diff(
                int(now * 1000), int(prev * 1000)
            ) / 1000.0
        return now - prev


# ── optional numpy ───────────────────────────────────────────────────────────
try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _np = None
    _HAS_NP = False


# ── Fix 7: minimum gradient magnitude to prevent dead-zone at init ───────────
_EPS_GRAD = 1e-3


# ═════════════════════════════════════════════════════════════════════════════
#  Pure-python linear-algebra helpers
# ═════════════════════════════════════════════════════════════════════════════

def _zeros(n: int):
    return [0.0] * n


def _randn2d(rows: int, cols: int, scale: float, rng):
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def _dot_mv(mat, vec):
    return [sum(m * v for m, v in zip(row, vec)) for row in mat]


def _outer(a, b):
    return [[ai * bj for bj in b] for ai in a]


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _norm2d(m):
    s = 0.0
    for row in m:
        for x in row:
            s += x * x
    return math.sqrt(s)


# ═════════════════════════════════════════════════════════════════════════════
#  TinyMLP
# ═════════════════════════════════════════════════════════════════════════════

class _TinyMLP:
    """Minimal MLP that can run with or without numpy."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden: int = 48,
        out_dim: int = 3,
        seed: int = 0
    ):
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_dim = out_dim

        if _HAS_NP:
            rng = _np.random.RandomState(seed)

            self.w1 = (
                rng.randn(hidden, input_dim).astype(_np.float32) *
                math.sqrt(2.0 / input_dim)
            )
            self.b1 = _np.zeros(hidden, dtype=_np.float32)

            self.w2 = (
                rng.randn(out_dim, hidden).astype(_np.float32) *
                math.sqrt(2.0 / hidden)
            )
            self.b2 = _np.zeros(out_dim, dtype=_np.float32)

        else:
            rng = random.Random(seed)

            self.w1 = _randn2d(
                hidden, input_dim, math.sqrt(2.0 / input_dim), rng
            )
            self.b1 = _zeros(hidden)

            self.w2 = _randn2d(
                out_dim, hidden, math.sqrt(2.0 / hidden), rng
            )
            self.b2 = _zeros(out_dim)

    # ── forward ──────────────────────────────────────────────────────────
    # Activations are clamped to ±1e6 to prevent float32 overflow in the
    # backward pass outer products (a1 * dL or x * dz1).
    _ACT_CLIP = 1e6

    def forward(self, x):
        if _HAS_NP:
            x = _np.asarray(x, dtype=_np.float32)

            z1 = self.w1.dot(x) + self.b1
            a1 = _np.clip(z1, 0.0, self._ACT_CLIP)
            z2 = self.w2.dot(a1) + self.b2

            self._x = x
            self._z1 = z1
            self._a1 = a1

            return z2.tolist()

        z1 = [s + b for s, b in zip(_dot_mv(self.w1, x), self.b1)]
        a1 = [_clamp(v, 0.0, self._ACT_CLIP) for v in z1]
        z2 = [s + b for s, b in zip(_dot_mv(self.w2, a1), self.b2)]

        self._x = list(x)
        self._z1 = z1
        self._a1 = a1

        return z2

    # ── backward ─────────────────────────────────────────────────────────
    def backward(self, dL_dout):
        if _HAS_NP:
            dL_dout = _np.asarray(dL_dout, dtype=_np.float32)

            with _np.errstate(over="ignore", invalid="ignore"):
                self._dw2 = _np.outer(dL_dout, self._a1)
                self._db2 = dL_dout.copy()

                da1 = self.w2.T.dot(dL_dout)
                dz1 = da1 * (self._z1 > 0).astype(_np.float32)

                self._dw1 = _np.outer(dz1, self._x)
                self._db1 = dz1.copy()

            # Sanitise: if any gradient is non-finite, zero it so the
            # SGD step doesn't corrupt weights.
            for g in (self._dw1, self._dw2, self._db1, self._db2):
                if not _np.all(_np.isfinite(g)):
                    _np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0,
                                   neginf=0.0)

            return

        self._dw2 = _outer(dL_dout, self._a1)
        self._db2 = list(dL_dout)

        da1 = [
            sum(self.w2[o][h] * dL_dout[o] for o in range(self.out_dim))
            for h in range(self.hidden)
        ]

        dz1 = [
            da1[h] * (1.0 if self._z1[h] > 0 else 0.0)
            for h in range(self.hidden)
        ]

        self._dw1 = _outer(dz1, self._x)
        self._db1 = list(dz1)

    # ── SGD ──────────────────────────────────────────────────────────────
    def step_sgd(self, lr=0.01, clip=0.03):
        if _HAS_NP:
            with _np.errstate(over="ignore", invalid="ignore"):
                for g in (self._dw1, self._dw2, self._db1, self._db2):
                    if not _np.all(_np.isfinite(g)):
                        _np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0,
                                       neginf=0.0)
                    n = float(_np.linalg.norm(g))
                    if n > clip and n > 0:
                        g *= clip / n

                self.w1 -= lr * self._dw1
                self.b1 -= lr * self._db1
                self.w2 -= lr * self._dw2
                self.b2 -= lr * self._db2

            return

        grads = [
            (self._dw1, True),
            (self._dw2, True),
            (self._db1, False),
            (self._db2, False),
        ]

        for g, is2d in grads:
            n = _norm2d(g) if is2d else _norm(g)
            if n > clip and n > 0:
                s = clip / n
                if is2d:
                    for row in g:
                        for j in range(len(row)):
                            row[j] *= s
                else:
                    for j in range(len(g)):
                        g[j] *= s

        for i in range(self.hidden):
            for j in range(self.input_dim):
                self.w1[i][j] -= lr * self._dw1[i][j]
            self.b1[i] -= lr * self._db1[i]

        for i in range(self.out_dim):
            for j in range(self.hidden):
                self.w2[i][j] -= lr * self._dw2[i][j]
            self.b2[i] -= lr * self._db2[i]

    # ── save / load ──────────────────────────────────────────────────────
    def _flatten(self):
        flat = []

        if _HAS_NP:
            flat.extend(self.w1.ravel().tolist())
            flat.extend(self.b1.tolist())
            flat.extend(self.w2.ravel().tolist())
            flat.extend(self.b2.tolist())
            return flat

        for row in self.w1:
            flat.extend(row)
        flat.extend(self.b1)
        for row in self.w2:
            flat.extend(row)
        flat.extend(self.b2)

        return flat

    def _unflatten(self, flat):
        idx = 0
        h, inp, out = self.hidden, self.input_dim, self.out_dim
        w1_size = h * inp

        if _HAS_NP:
            self.w1 = _np.array(
                flat[idx:idx + w1_size], dtype=_np.float32
            ).reshape(h, inp)
            idx += w1_size

            self.b1 = _np.array(flat[idx:idx + h], dtype=_np.float32)
            idx += h

            w2_size = out * h
            self.w2 = _np.array(
                flat[idx:idx + w2_size], dtype=_np.float32
            ).reshape(out, h)
            idx += w2_size

            self.b2 = _np.array(flat[idx:idx + out], dtype=_np.float32)
            return

        self.w1 = [
            flat[idx + i * inp: idx + (i + 1) * inp] for i in range(h)
        ]
        idx += w1_size

        self.b1 = flat[idx:idx + h]
        idx += h

        w2_size = out * h
        self.w2 = [
            flat[idx + i * h: idx + (i + 1) * h] for i in range(out)
        ]
        idx += w2_size

        self.b2 = flat[idx:idx + out]

    def save(self, path: str) -> None:
        flat = self._flatten()
        header = struct.pack(
            "<4sIII", b"NMLP",
            self.input_dim, self.hidden, self.out_dim,
        )
        body = struct.pack(f"<{len(flat)}f", *flat)
        with open(path, "wb") as f:
            f.write(header + body)

    def load(self, path: str) -> bool:
        """Load weights from an NMLP file. Returns True on success."""
        try:
            with open(path, "rb") as f:
                data = f.read()

            magic, inp, hid, out = struct.unpack("<4sIII", data[:16])

            if magic != b"NMLP":
                return False

            if (inp, hid, out) != (self.input_dim, self.hidden, self.out_dim):
                raise ValueError(
                    f"Weight file architecture ({inp}, {hid}, {out}) does not "
                    f"match this network ({self.input_dim}, {self.hidden}, "
                    f"{self.out_dim}). Reconstruct PID with the same "
                    f"error_history / output_history / profile that was used "
                    f"when saving."
                )

            n = inp * hid + hid + out * hid + out
            flat = list(struct.unpack(f"<{n}f", data[16:16 + n * 4]))
            self._unflatten(flat)

            return True

        except (OSError, struct.error) as exc:
            raise OSError(
                f"Could not read weight file '{path}': {exc}"
            ) from exc


# ═════════════════════════════════════════════════════════════════════════════
#  System models for zeta (ζ) estimation
# ═════════════════════════════════════════════════════════════════════════════
#
# These functions compute the effective damping ratio of the closed-loop
# system given the current adaptive gains (Kp, Kd) and the known plant
# parameters.  Each model corresponds to a standard transfer-function form
# from control theory.
#
# ── "mass" model ────────────────────────────────────────────────────────────
#   Plant: G(s) = 1 / (m·s² + b·s)
#   This is a rigid body (mass m) with viscous friction (damping b) driven
#   by a force.  With PD control the closed-loop characteristic equation is:
#
#       m·s² + (b + Kd)·s + Kp = 0
#
#   Comparing to the standard form  s² + 2ζω₀s + ω₀² = 0  gives:
#
#       ω₀ = sqrt(Kp / m)
#       ζ  = (b + Kd) / (2 · sqrt(m · Kp))
#
# ── "dc_motor" model ───────────────────────────────────────────────────────
#   Plant: G(s) = K / (τ·s + 1)
#   A first-order motor with gain K and time constant τ.  With PD control
#   the closed-loop becomes second-order:
#
#       τ·s² + (1 + K·Kd)·s + K·Kp = 0
#
#   Dividing by τ and matching to the standard form:
#
#       ω₀ = sqrt(K · Kp / τ)
#       ζ  = (1 + K · Kd) / (2 · sqrt(τ · K · Kp))
#
# Both functions return a clamped float in [0.0, 10.0] and gracefully
# handle degenerate inputs (Kp ≤ 0, mass ≤ 0, etc.) by returning 0.0.
# ═════════════════════════════════════════════════════════════════════════════

# Upper clamp for zeta — values above this are meaninglessly overdamped and
# would create outsized gradient signals if left uncapped.
_ZETA_MAX = 2

# Minimum denominator to avoid division-by-zero in the zeta formulas.
_ZETA_EPS = 1e-12


def _zeta_mass(kp, kd, mass, damping):
    """Compute ζ for a mass-spring-damper plant: G(s) = 1/(ms²+bs).

    Parameters
    ----------
    kp : float      Current proportional gain.
    kd : float      Current derivative gain.
    mass : float    System mass (kg).
    damping : float Viscous damping coefficient (N·s/m).

    Returns
    -------
    float   Effective damping ratio, clamped to [0.0, _ZETA_MAX].
    """
    if kp <= 0.0 or mass <= 0.0:
        return 0.0

    denom = 2.0 * math.sqrt(mass * kp)

    if denom < _ZETA_EPS:
        return 0.0

    zeta = (damping + kd) / denom
    return _clamp(zeta, 0.0, _ZETA_MAX)


def _zeta_dc_motor(kp, kd, motor_gain, time_constant):
    """Compute ζ for a DC-motor plant: G(s) = K/(τs+1).

    Parameters
    ----------
    kp : float            Current proportional gain.
    kd : float            Current derivative gain.
    motor_gain : float    DC gain K of the motor.
    time_constant : float Electrical/mechanical time constant τ (seconds).

    Returns
    -------
    float   Effective damping ratio, clamped to [0.0, _ZETA_MAX].
    """
    if kp <= 0.0 or motor_gain <= 0.0 or time_constant <= 0.0:
        return 0.0

    denom = 2.0 * math.sqrt(time_constant * motor_gain * kp)

    if denom < _ZETA_EPS:
        return 0.0

    zeta = (1.0 + motor_gain * kd) / denom
    return _clamp(zeta, 0.0, _ZETA_MAX)


# Registry mapping system_type strings to their zeta functions and the
# plant-parameter names each one expects.
_SYSTEM_MODELS = {
    "mass": {
        "fn": _zeta_mass,
        "params": ("mass", "damping"),
    },
    "dc_motor": {
        "fn": _zeta_dc_motor,
        "params": ("motor_gain", "time_constant"),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  NeuroPID
# ═════════════════════════════════════════════════════════════════════════════

# Fix 17: new magic NPD4 — adds system model + zeta fields to the header.
_PID_MAGIC = b"NPD4"
_PID_MAGIC_V3 = b"NPD3"  # still readable for backward compat
_PID_MAGIC_V2 = b"NPD2"  # still readable for backward compat


class PID:
    """Adaptive PID controller with neural gain tuning.

    Fix 17 additions
    ─────────────────
    The controller now estimates the effective damping ratio (ζ) of the
    closed-loop system at every control step and includes the deviation
    from a user-specified target_zeta in the neural reward signal.

    Two plant models are supported out of the box:

      "mass"     — rigid body with viscous friction
                   G(s) = 1 / (m·s² + b·s)
                   ζ = (b + Kd) / (2·√(m·Kp))

      "dc_motor" — first-order motor with gain and time constant
                   G(s) = K / (τ·s + 1)
                   ζ = (1 + K·Kd) / (2·√(τ·K·Kp))

    If system_type is None (the default for backward compatibility) the
    zeta calculation is skipped and the reward function behaves exactly
    as before Fix 17.

    The default target_zeta is 1.0 (critically damped): fastest settling
    with zero overshoot.
    """

    PROFILES = {
        "low": 16,
        "medium": 32,
        "high": 64,
    }

    def __init__(
        self,
        kp: float = 10.0,
        ki: float = 1.0,
        kd: float = 0.1,
        out_min=None,
        out_max=None,
        gain_limits=None,
        hidden=None,
        profile: str = "medium",
        error_history: int = 3,
        output_history: int = 2,
        lr: float = 0.005,
        clip: float = 0.03,
        seed: int = 0,
        integral_limit: float = 50.0,
        # Fix 13: renamed from derivative_smoothing for clarity
        derivative_alpha: float = 0.15,
        frozen: bool = False,
        gain_alpha: float = 0.08,
        default_dt: float = 0.02,
        # Fix 12: maximum dt to prevent spikes from scheduling stalls
        max_dt: float | None = None,
        # Fix 9: power limit — now applied upstream of anti-windup
        max_output: float = 0.6,
        # ── Fix 17: zeta-aware tuning parameters ────────────────────────
        #
        # target_zeta — the damping ratio the neural network should steer
        #   the gains toward.  Default is 1.0 (critically damped):
        #   fastest settling with zero overshoot.
        #
        # system_type — selects the plant model used to compute ζ from the
        #   current adaptive gains.  None disables zeta optimisation
        #   entirely (full backward compatibility).
        #
        # mass / damping — plant parameters for system_type="mass".
        # motor_gain / time_constant — plant parameters for "dc_motor".
        #
        # zeta_weight — how strongly the zeta error term influences the
        #   neural reward relative to the tracking-error and control-effort
        #   terms.  Higher values make the MLP prioritise damping-ratio
        #   convergence over raw error reduction.
        target_zeta: float = 1.0,
        system_type: str | None = None,
        mass: float = 1.0,
        damping: float = 0.0,
        motor_gain: float = 1.0,
        time_constant: float = 1.0,
        zeta_weight: float = 0.2,
    ):
        # ── Fix 15: validate constructor arguments ──────────────────────
        if out_min is not None and out_max is not None and out_min >= out_max:
            raise ValueError(
                f"out_min ({out_min}) must be less than out_max ({out_max})"
            )

        if not 0.0 <= max_output <= 1.0:
            raise ValueError(
                f"max_output must be in [0.0, 1.0], got {max_output}"
            )

        _gl = gain_limits or {}
        kp_range = _gl.get("kp", (1.0, 200.0))
        ki_range = _gl.get("ki", (0.0, 50.0))
        kd_range = _gl.get("kd", (0.0, 50.0))

        for name, (lo, hi) in [
            ("kp", kp_range), ("ki", ki_range), ("kd", kd_range),
        ]:
            if lo > hi:
                raise ValueError(
                    f"gain_limits['{name}'] lower bound ({lo}) exceeds "
                    f"upper bound ({hi})"
                )

        # ── Fix 17: validate system model parameters ────────────────────
        if system_type is not None and system_type not in _SYSTEM_MODELS:
            raise ValueError(
                f"Unknown system_type '{system_type}'. "
                f"Supported: {list(_SYSTEM_MODELS.keys())} or None."
            )

        if target_zeta < 0.0:
            raise ValueError(
                f"target_zeta must be non-negative, got {target_zeta}"
            )

        if zeta_weight < 0.0:
            raise ValueError(
                f"zeta_weight must be non-negative, got {zeta_weight}"
            )

        if system_type == "mass":
            if mass <= 0.0:
                raise ValueError(f"mass must be positive, got {mass}")
        elif system_type == "dc_motor":
            if motor_gain <= 0.0:
                raise ValueError(
                    f"motor_gain must be positive, got {motor_gain}"
                )
            if time_constant <= 0.0:
                raise ValueError(
                    f"time_constant must be positive, got {time_constant}"
                )

        # base gains
        self.base_kp = kp
        self.base_ki = ki
        self.base_kd = kd

        self.kp = kp
        self.ki = ki
        self.kd = kd

        # output clamp
        self.out_min = out_min
        self.out_max = out_max

        # gain safety clamps
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        self.integral_limit = integral_limit

        # derivative filtering (Fix 13: renamed)
        self.derivative = 0.0
        self.derivative_alpha = derivative_alpha

        # history buffers
        self._n_err = error_history
        self._n_out = output_history

        self.e_hist = [0.0] * error_history
        self.u_hist = [0.0] * output_history

        # network profile
        if hidden is None:
            hidden = self.PROFILES.get(profile, 32)
        self.profile = profile

        # ── Fix 10 & 11: build feature list and derive input_dim ────────
        # Feature layout:
        #   e_hist (error_history)
        #   integral (1)
        #   derivative (1)
        #   u_hist (output_history)
        #   setpoint (1)
        #   dt (1)                     ← was 0.0 placeholder
        #   setpoint_error_ratio (1)   ← was 0.0 placeholder
        self._feature_len = error_history + 1 + 1 + output_history + 1 + 1 + 1
        input_dim = self._feature_len

        self._mlp = _TinyMLP(
            input_dim=input_dim,
            hidden=hidden,
            out_dim=3,
            seed=seed,
        )

        # learning hyper-parameters
        self.lr = lr
        self.clip = clip
        self.frozen = frozen

        self.gain_alpha = gain_alpha
        self.default_dt = default_dt

        # Fix 12: dt cap — defaults to 10× default_dt
        self.max_dt = max_dt if max_dt is not None else default_dt * 10.0

        # Fix 9: power limit
        self.max_output = max_output

        # ── Fix 17: zeta-aware tuning state ─────────────────────────────
        self.target_zeta = target_zeta
        self.system_type = system_type
        self.zeta_weight = zeta_weight

        # Plant parameters — stored generically so both models can coexist
        # in the same object without conditional attribute creation.
        self.mass = mass
        self.damping = damping
        self.motor_gain = motor_gain
        self.time_constant = time_constant

        # Live zeta estimate — updated every step() when system_type is set.
        # Exposed publicly so the caller can log / display it.
        self.current_zeta = 0.0
        self.zeta_error = 0.0

        # bookkeeping
        self.setpoint = 0.0
        self.step_count = 0

        # temporal reward
        self.prev_abs_error = 0.0

    # ── Fix 17: zeta computation helper ──────────────────────────────────
    def _compute_zeta(self) -> float:
        """Compute the effective damping ratio from current gains and plant.

        Returns 0.0 if system_type is None or the calculation is invalid
        (e.g. Kp ≤ 0).  The result is clamped to [0, _ZETA_MAX].
        """
        if self.system_type is None:
            return 0.0

        model = _SYSTEM_MODELS[self.system_type]

        if self.system_type == "mass":
            return model["fn"](self.kp, self.kd, self.mass, self.damping)

        if self.system_type == "dc_motor":
            return model["fn"](
                self.kp, self.kd, self.motor_gain, self.time_constant,
            )

        # Unreachable if constructor validation is correct, but safe.
        return 0.0

    # ── main step ────────────────────────────────────────────────────────
    def step(self, error: float, setpoint: float = 0.0) -> float:
        now = _monotonic()

        if self.prev_time is None:
            dt = self.default_dt
        else:
            # Fix 6: wrap-safe elapsed time
            raw_dt = _elapsed(self.prev_time, now)
            # Fix 12: clamp dt to [1ms, max_dt]
            dt = _clamp(raw_dt, 1e-3, self.max_dt)

        self.prev_time = now
        self.setpoint = setpoint

        # ── filtered derivative (Fix 13: uses derivative_alpha) ─────────
        raw_derivative = (error - self.prev_error) / dt

        self.derivative = (
            (1.0 - self.derivative_alpha) * self.derivative +
            self.derivative_alpha * raw_derivative
        )

        # ── Fix 10: setpoint-error ratio feature ────────────────────────
        # Bounded to [-1, 1] to keep the feature well-scaled.
        if abs(setpoint) > 1e-9:
            sp_err_ratio = _clamp(error / setpoint, -1.0, 1.0)
        else:
            sp_err_ratio = _clamp(error, -1.0, 1.0)

        # ── Fix 10 & 11: feature vector with no dead inputs ─────────────
        feat = (
            self.e_hist +
            [self.integral, self.derivative] +
            self.u_hist +
            [setpoint, dt, sp_err_ratio]
        )

        # Fix 11: assert feature length matches input_dim
        assert len(feat) == self._feature_len, (
            f"Feature length {len(feat)} != expected {self._feature_len}"
        )

        # ── MLP inference ───────────────────────────────────────────────
        out = self._mlp.forward(feat)

        # smooth adaptive scaling
        scale = min(2.0, 0.2 + abs(error))
        adj = [o * scale for o in out]

        target_kp = _clamp(self.base_kp + adj[0], *self.kp_range)
        target_ki = _clamp(self.base_ki + adj[1], *self.ki_range)
        target_kd = _clamp(self.base_kd + adj[2], *self.kd_range)

        self.kp += self.gain_alpha * (target_kp - self.kp)
        self.ki += self.gain_alpha * (target_ki - self.ki)
        self.kd += self.gain_alpha * (target_kd - self.kd)

        # ── Fix 17: compute effective zeta from current adaptive gains ──
        # This must happen AFTER the gains are updated but BEFORE the
        # reward is computed, so the reward reflects the gains that
        # actually produced this step's output.
        self.current_zeta = self._compute_zeta()

        if self.system_type is not None:
            self.zeta_error = abs(self.target_zeta - self.current_zeta)
        else:
            self.zeta_error = 0.0

        # ── PID computation ─────────────────────────────────────────────
        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = _clamp(
                self.integral, -self.integral_limit, self.integral_limit
            )

        output_raw = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * self.derivative
        )

        # ── Fix 5: back-calculation anti-windup against hardware limits ──
        output = output_raw

        if self.out_min is not None and output < self.out_min:
            output = self.out_min
            if self.ki != 0.0:
                self.integral -= (output_raw - output) / self.ki

        elif self.out_max is not None and output > self.out_max:
            output = self.out_max
            if self.ki != 0.0:
                self.integral -= (output_raw - output) / self.ki

        # ── Fix 9: max_output cap ───────────────────────────────────────
        # Clamp the magnitude to max_output fraction of the saturation
        # range.  This is a soft safety ceiling — the anti-windup above
        # already corrected the integral against the real hardware limits,
        # so this extra cap doesn't distort the integrator.
        if self.out_max is not None:
            cap_hi = self.out_max * self.max_output
            if output > cap_hi:
                output = cap_hi

        if self.out_min is not None:
            cap_lo = self.out_min * self.max_output
            if output < cap_lo:
                output = cap_lo

        # ── online learning ─────────────────────────────────────────────
        # Fix 2: skip learning on step 0
        if not self.frozen and self.step_count > 0:

            # ── Fix 17: multi-objective reward ──────────────────────────
            #
            # The reward signal now balances three objectives:
            #
            #   1. error_improvement — positive when |error| decreased
            #      since the last step, encouraging the MLP to reduce
            #      tracking error.
            #
            #   2. control_penalty — a small cost proportional to the
            #      control magnitude, discouraging unnecessarily large
            #      actuator commands.
            #
            #   3. zeta_penalty — proportional to |target_ζ − current_ζ|,
            #      encouraging the MLP to steer the gains toward the
            #      desired damping regime.  Only active when a system
            #      model is configured (system_type is not None).
            #
            # The overall loss_signal is:
            #
            #   loss = error_improvement
            #        − control_penalty
            #        − zeta_penalty × zeta_weight
            #
            error_improvement = self.prev_abs_error - abs(error)

            # Use the actual (post-clamp) output for control penalty so
            # the learning signal is consistent with what goes into u_hist
            control_penalty = abs(output) * 0.001

            # Zeta penalty — only contributes when a plant model is set.
            if self.system_type is not None:
                zeta_penalty = self.zeta_error
            else:
                zeta_penalty = 0.0

            loss_signal = (
                error_improvement
                - control_penalty
                - zeta_penalty * self.zeta_weight
            )

            # Fix 7: gradient floor prevents dead-zone when outputs ≈ 0
            dL = []
            for o in out:
                g = -loss_signal * o
                if abs(g) < _EPS_GRAD and loss_signal != 0.0:
                    g = -math.copysign(_EPS_GRAD, loss_signal)
                dL.append(g)

            self._mlp.backward(dL)
            self._mlp.step_sgd(lr=self.lr, clip=self.clip)

        # ── update history ──────────────────────────────────────────────
        self.e_hist = [error] + self.e_hist[:self._n_err - 1]
        self.u_hist = [output] + self.u_hist[:self._n_out - 1]

        self.prev_error = error
        self.prev_abs_error = abs(error)
        self.step_count += 1

        return output

    # ── reset ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset PID state. MLP weights are preserved."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.derivative = 0.0

        self.e_hist = [0.0] * self._n_err
        self.u_hist = [0.0] * self._n_out

        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd

        self.step_count = 0
        self.prev_abs_error = 0.0

        # Fix 17: reset live zeta state
        self.current_zeta = 0.0
        self.zeta_error = 0.0

    # ── freeze learning ─────────────────────────────────────────────────
    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    # ── Fix 17: zeta helper methods ─────────────────────────────────────

    def get_zeta(self) -> float:
        """Return the most recently computed effective damping ratio.

        This value is updated every step() call.  If no system model is
        configured (system_type is None) this always returns 0.0.
        """
        return self.current_zeta

    def set_target_zeta(self, target_zeta: float) -> None:
        """Change the target damping ratio at runtime.

        Parameters
        ----------
        target_zeta : float
            Desired damping ratio.  Must be non-negative.
            Values in [0.7, 1.0] are recommended for robotics.
        """
        if target_zeta < 0.0:
            raise ValueError(
                f"target_zeta must be non-negative, got {target_zeta}"
            )
        self.target_zeta = target_zeta

    def set_system_model(
        self,
        system_type: str | None,
        mass: float | None = None,
        damping: float | None = None,
        motor_gain: float | None = None,
        time_constant: float | None = None,
    ) -> None:
        """Change the plant model and/or parameters at runtime.

        Pass system_type=None to disable zeta optimisation entirely.

        Only parameters relevant to the chosen model need to be supplied;
        others are ignored.  Omitted parameters retain their current value.

        Parameters
        ----------
        system_type : str or None
            "mass", "dc_motor", or None.
        mass : float, optional
            System mass for the "mass" model (kg).  Must be positive.
        damping : float, optional
            Viscous damping for the "mass" model (N·s/m).
        motor_gain : float, optional
            DC gain K for the "dc_motor" model.  Must be positive.
        time_constant : float, optional
            Time constant τ for the "dc_motor" model (s).  Must be positive.
        """
        if system_type is not None and system_type not in _SYSTEM_MODELS:
            raise ValueError(
                f"Unknown system_type '{system_type}'. "
                f"Supported: {list(_SYSTEM_MODELS.keys())} or None."
            )

        self.system_type = system_type

        if mass is not None:
            if mass <= 0.0:
                raise ValueError(f"mass must be positive, got {mass}")
            self.mass = mass

        if damping is not None:
            self.damping = damping

        if motor_gain is not None:
            if motor_gain <= 0.0:
                raise ValueError(
                    f"motor_gain must be positive, got {motor_gain}"
                )
            self.motor_gain = motor_gain

        if time_constant is not None:
            if time_constant <= 0.0:
                raise ValueError(
                    f"time_constant must be positive, got {time_constant}"
                )
            self.time_constant = time_constant

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """
        Save MLP weights, frozen flag, max_output, and system model
        parameters to a binary file (magic NPD4).

        Fix 17: NPD4 extends NPD3 with zeta-aware fields:
          - system_type (encoded as uint8: 0=None, 1=mass, 2=dc_motor)
          - target_zeta (float)
          - zeta_weight (float)
          - mass (float)
          - damping (float)
          - motor_gain (float)
          - time_constant (float)
        """
        flat = self._mlp._flatten()

        # Encode system_type as uint8
        _sys_map = {None: 0, "mass": 1, "dc_motor": 2}
        sys_code = _sys_map.get(self.system_type, 0)

        # Header: magic(4) + input_dim(4) + hidden(4) + out_dim(4)
        #       + frozen(4) + max_output(4f)
        #       + sys_code(1B) + pad(3B)
        #       + target_zeta(4f) + zeta_weight(4f)
        #       + mass(4f) + damping(4f) + motor_gain(4f) + time_constant(4f)
        #
        # Total header: 4(magic)+16(IIII)+4(f)+1(B)+3(pad)+24(ffffff) = 52 bytes
        header = struct.pack(
            "<4sIIIIf B3x ffffff",
            _PID_MAGIC,
            self._mlp.input_dim,
            self._mlp.hidden,
            self._mlp.out_dim,
            int(self.frozen),
            self.max_output,
            sys_code,
            # 3x padding bytes (implicit from struct)
            self.target_zeta,
            self.zeta_weight,
            self.mass,
            self.damping,
            self.motor_gain,
            self.time_constant,
        )

        body = struct.pack(f"<{len(flat)}f", *flat)

        with open(path, "wb") as f:
            f.write(header + body)

    def load(self, path: str) -> bool:
        """
        Load from an NPD4 file (current), NPD3, NPD2, or legacy NMLP file.

        Architecture mismatches raise ValueError.
        Returns True on success.
        """
        with open(path, "rb") as f:
            data = f.read()

        magic = data[:4]

        # ── NPD4 format (current — Fix 17) ──────────────────────────────
        if magic == _PID_MAGIC:
            # Header is 52 bytes total (see save() layout)
            (
                inp, hid, out, frozen_flag, max_output_val,
                sys_code,
                target_zeta_val, zeta_weight_val,
                mass_val, damping_val, motor_gain_val, time_constant_val,
            ) = struct.unpack("<IIIIf B3x ffffff", data[4:52])

            self._check_arch(inp, hid, out)

            n = inp * hid + hid + out * hid + out
            flat = list(struct.unpack(f"<{n}f", data[52:52 + n * 4]))
            self._mlp._unflatten(flat)

            self.frozen = bool(frozen_flag)
            self.max_output = _clamp(max_output_val, 0.0, 1.0)

            # Decode system_type
            _sys_decode = {0: None, 1: "mass", 2: "dc_motor"}
            self.system_type = _sys_decode.get(sys_code, None)

            self.target_zeta = max(0.0, target_zeta_val)
            self.zeta_weight = max(0.0, zeta_weight_val)
            self.mass = mass_val
            self.damping = damping_val
            self.motor_gain = motor_gain_val
            self.time_constant = time_constant_val

            return True

        # ── NPD3 format (backward compat — pre-zeta) ────────────────────
        if magic == _PID_MAGIC_V3:
            inp, hid, out, frozen_flag = struct.unpack("<IIII", data[4:20])
            max_output_val = struct.unpack("<f", data[20:24])[0]

            self._check_arch(inp, hid, out)

            n = inp * hid + hid + out * hid + out
            flat = list(struct.unpack(f"<{n}f", data[24:24 + n * 4]))
            self._mlp._unflatten(flat)

            self.frozen = bool(frozen_flag)
            self.max_output = _clamp(max_output_val, 0.0, 1.0)
            # system_type / zeta params not in NPD3 — keep constructor defaults

            return True

        # ── NPD2 format (backward compat) ───────────────────────────────
        if magic == _PID_MAGIC_V2:
            inp, hid, out, frozen_flag = struct.unpack("<IIII", data[4:20])

            self._check_arch(inp, hid, out)

            n = inp * hid + hid + out * hid + out
            flat = list(struct.unpack(f"<{n}f", data[20:20 + n * 4]))
            self._mlp._unflatten(flat)

            self.frozen = bool(frozen_flag)
            # max_output not in NPD2 — keep constructor default

            return True

        # ── legacy NMLP format ──────────────────────────────────────────
        if magic == b"NMLP":
            return self._mlp.load(path)

        raise ValueError(
            f"Unrecognised file format (magic={magic!r}). "
            "Expected NPD4, NPD3, NPD2, or NMLP."
        )

    def _check_arch(self, inp, hid, out):
        if (inp, hid, out) != (
            self._mlp.input_dim, self._mlp.hidden, self._mlp.out_dim,
        ):
            raise ValueError(
                f"Weight file architecture ({inp}, {hid}, {out}) does not "
                f"match this PID's network ({self._mlp.input_dim}, "
                f"{self._mlp.hidden}, {self._mlp.out_dim}). Reconstruct "
                f"PID with the same error_history / output_history / "
                f"profile that was used when saving."
            )

    # ── info ────────────────────────────────────────────────────────────
    def gains(self) -> Gains:
        """Return current gains as a named tuple."""
        return Gains(self.kp, self.ki, self.kd)

    def __repr__(self) -> str:
        parts = [
            f"NeuroPID(",
            f"kp={self.kp:.3f}, ",
            f"ki={self.ki:.3f}, ",
            f"kd={self.kd:.3f}, ",
            f"profile='{self.profile}', ",
            f"frozen={self.frozen}, ",
            f"max_output={self.max_output:.0%}, ",
        ]

        # Fix 17: include zeta info when a system model is configured
        if self.system_type is not None:
            parts.append(
                f"ζ={self.current_zeta:.3f}→{self.target_zeta:.2f}, "
            )
            parts.append(f"system='{self.system_type}', ")

        parts.append(f"steps={self.step_count})")

        return "".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
#  Example: Differential-Drive Robot Tuning
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    # ─────────────────────────────────────────────────────────────────────
    #  Turn PID
    #  ========
    #  For a differential-drive robot turning in place.
    #  Fixed motor parameters: motor_gain=0.3, time_constant=0.15
    #
    #  These should be tuned to match your actual hardware:
    #    motor_gain = (max_speed_at_100pct) / 100
    #    time_constant ≈ time to reach 63% of final speed (seconds)
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Differential-Drive Robot: Turn PID")
    print("=" * 70)

    turn_pid = PID(
        kp=6.0,
        ki=0.3,
        kd=2.0,
        out_min=-75,
        out_max=75,
        gain_limits={
            "kp": (1.0, 15.0),
            "ki": (0.05, 2.0),
            "kd": (0.5, 5.0),
        },
        profile="low",
        gain_alpha=0.03,
        lr=0.002,
        integral_limit=30.0,
        derivative_alpha=0.08,
        system_type="dc_motor",
        motor_gain=0.3,      # tune to your motor: output_speed / input_pwm
        time_constant=0.15,  # ~150ms typical for small geared DC motor
        target_zeta=1.0,
        zeta_weight=0.5,
        max_output=0.75,
        default_dt=0.02,
    )

    # Simulate a 90-degree turn
    target_angle = 90.0
    current_angle = 0.0

    print("\nTurning toward 90°...")
    for i in range(200):
        error = target_angle - current_angle
        control = turn_pid.step(error, setpoint=target_angle)

        # Simple angular plant: angle += (control_speed * dt)
        # Assume 75% output = 75 deg/sec max
        current_angle += (control * 0.75 / 75.0) * 0.02

        if i % 20 == 0:
            g = turn_pid.gains()
            print(
                f"  Step {i:03d} | "
                f"err={error:7.2f}° | "
                f"ctrl={control:6.1f}% | "
                f"kp={g.kp:5.2f} ki={g.ki:5.3f} kd={g.kd:5.3f} | "
                f"ζ={turn_pid.current_zeta:.3f}"
            )

        time.sleep(0.02)
        if abs(error) < 1.0:
            print(f"  → Converged at step {i}")
            break

    turn_pid.save("turn_weights.bin")

    # ─────────────────────────────────────────────────────────────────────
    #  Drive PID
    #  =========
    #  For a differential-drive robot driving straight.
    #  Same motor parameters as turn PID.
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Differential-Drive Robot: Drive PID")
    print("=" * 70)

    drive_pid = PID(
        kp=8.0,
        ki=0.5,
        kd=1.5,
        out_min=-75, out_max=75,
        gain_limits={
            "kp": (2.0, 20.0),
            "ki": (0.1, 3.0),
            "kd": (0.2, 4.0),
        },
        profile="low",
        gain_alpha=0.03,
        lr=0.002,
        integral_limit=40.0,
        derivative_alpha=0.08,
        system_type="dc_motor",
        motor_gain=0.3,
        time_constant=0.15,
        target_zeta=1.0,
        zeta_weight=0.5,
        max_output=0.75,
        default_dt=0.02,
    )

    # Simulate driving 100 units and stopping 30 cm away from wall
    target_distance = 100.0
    current_distance = 0.0
    stop_distance = 30.0

    print("\nDriving toward 100 units, target stop at 70 units...")
    for i in range(300):
        error = (target_distance - stop_distance) - current_distance
        control = drive_pid.step(error, setpoint=target_distance - stop_distance)

        # Simple linear plant
        current_distance += (control * 0.75 / 75.0) * 0.02

        if i % 30 == 0:
            g = drive_pid.gains()
            remaining = target_distance - current_distance
            print(
                f"  Step {i:03d} | "
                f"err={error:7.2f} | "
                f"pos={current_distance:6.1f} | "
                f"ctrl={control:6.1f}% | "
                f"kp={g.kp:5.2f} ki={g.ki:5.3f} kd={g.kd:5.3f} | "
                f"ζ={drive_pid.current_zeta:.3f}"
            )

        time.sleep(0.02)
        if abs(error) < 2.0:
            print(f"  → Converged at step {i}, distance={current_distance:.1f}")
            break

    drive_pid.save("drive_weights.bin")

    # ─────────────────────────────────────────────────────────────────────
    #  Load and verify persistence
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Verifying save/load persistence")
    print("=" * 70)

    turn_pid_loaded = PID(
        kp=6.0, ki=0.3, kd=2.0,
        out_min=-75, out_max=75,
        profile="low",
        system_type="dc_motor",
        motor_gain=0.3,
        time_constant=0.15,
    )
    turn_pid_loaded.load("turn_weights.bin")
    print(f"Loaded turn PID: {turn_pid_loaded}")

    drive_pid_loaded = PID(
        kp=8.0, ki=0.5, kd=1.5,
        out_min=-75, out_max=75,
        profile="low",
        system_type="dc_motor",
        motor_gain=0.3,
        time_constant=0.15,
    )
    drive_pid_loaded.load("drive_weights.bin")
    print(f"Loaded drive PID: {drive_pid_loaded}")
