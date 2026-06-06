"""
Neuro.py  —  Adaptive (neural) PID gain tuner for the wall-following maze robot.
================================================================================

ARCHITECTURE (unchanged in spirit)
----------------------------------
A small MLP (5 -> 16 tanh -> 3 scaled-sigmoid) maps the *current driving
situation* to PID gains (Kp, Ki, Kd).  The classic PID controller still does the
real-time control; the network only schedules its gains online.  Gains are
HARD-BOUNDED by a scaled sigmoid, and the output layer is initialized to a known
safe controller so the robot is safe at step zero and only learns *deviations*.

WHAT CHANGED (engineering-report summary — every item is a requested fix)
------------------------------------------------------------------------
1. LEARN FROM REAL CLOSED-LOOP PERFORMANCE  (see `PerformanceMonitor` + the
   policy-gradient update in `OnlineTuner._learn_from_window`).
   Previously the loss only saw the gain values and the *instantaneous* error,
   so the network never found out whether the gains it picked actually produced
   good control.  Now a rolling buffer measures the *consequences* of the gains
   (cumulative error, overshoot, oscillation, control effort, settling) and an
   advantage-weighted policy-gradient update reinforces gain choices that led to
   measurably better windows and discourages the ones that did not.

2. PERSIST LEARNED WEIGHTS  (see `save_checkpoint` / `_load`).
   Atomic writes (tmp + os.replace), versioned checkpoints, auto-load on start,
   periodic saves every N updates, a final save on shutdown, and graceful
   handling of missing / corrupt / version-mismatched files.

3. ROBUST DERIVATIVE  — implemented on the controller side in
   maze-bot-1.py `NeuralPID` (first-order low-pass + clamp).  Documented there.

4. CONSISTENT FEATURE SEMANTICS  (see `FeatureBuilder` + the three subclasses
   `DistanceFeatureBuilder` / `HeadingFeatureBuilder` / `TurnFeatureBuilder`).
   The 5 network inputs now always carry the same physical meaning, and each
   controller gets a builder that normalizes *its* error/actuation correctly
   (including the sign of the actuation, which differs between loops).

5. ANTI-DRIFT  (see `_update_best_and_guards`).
   Learning-rate decay, automatic freezing once performance stops improving,
   a continuously-maintained best checkpoint, and rollback to that best
   checkpoint if performance degrades — so the robot cannot slowly *unlearn*.

6. ZETA LOGIC / DOCUMENTATION  (see `zeta_est` and the notes below).
   `zeta = Kd / (2*sqrt(Kp + 0.1*Ki))` is the damping estimate.  zeta = 1 is
   critical damping (no overshoot).  This file no longer claims the loops are
   critically damped: the wall-approach loop runs slightly *underdamped*
   (target zeta < 1) as a deliberate speed/overshoot trade-off — see the
   decision note in maze-bot-1.py where the targets are set.

Plus: NaN/inf guards, gradient clipping, advantage normalization, a save lock
for thread-safety, bounded deques to avoid unbounded memory growth, and a
no-exploration safe default.

Tested target: Jetson Orin Nano with PyTorch (JetPack).  CPU is fine too.
"""

import math
import os
import threading
from collections import deque

import torch
import torch.nn as nn

# Checkpoint format version — bump when the on-disk schema changes so old/new
# files are never silently mixed (fix #2: version information in checkpoints).
CHECKPOINT_FORMAT = 2

# ---------------------------------------------------------------------------
# Safe gain ranges (the scaled-sigmoid output can never leave these)
# ---------------------------------------------------------------------------
KP_LO, KP_HI = 20.0, 120.0
KI_LO, KI_HI = 0.0, 10.0
KD_LO, KD_HI = 2.0, 20.0

# Safe critically-damped starting point (used for init AND as the fallback).
# NOTE: this *baseline* is genuinely critically damped (zeta == 1).  Individual
# control loops may target a different zeta (see OnlineTuner(target_zeta=...));
# that is their steady-state aim, not the init point.
KP0, KI0 = 80.0, 0.5
KD0 = 2.0 * math.sqrt(KP0 + 0.1 * KI0)          # ~= 17.89 -> zeta_est == 1.0
SAFE_GAINS = torch.tensor([KP0, KI0, KD0], dtype=torch.float32)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# The network  (unchanged structurally; kept backward compatible)
# ---------------------------------------------------------------------------
class GainMLP(nn.Module):
    """5 inputs -> 16 tanh -> 3 scaled-sigmoid outputs (Kp, Ki, Kd)."""

    def __init__(self, n_in: int = 5, n_hidden: int = 16,
                 kp_range=(KP_LO, KP_HI),
                 ki_range=(KI_LO, KI_HI),
                 kd_range=(KD_LO, KD_HI),
                 init_gains=None):
        super().__init__()
        kp_lo, kp_hi = kp_range
        ki_lo, ki_hi = ki_range
        kd_lo, kd_hi = kd_range

        # Default safe init = the critically-damped distance controller.
        if init_gains is None:
            init_gains = (KP0, KI0, KD0)
        kp0, ki0, kd0 = init_gains

        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 3)

        # Hidden: Xavier (correct for tanh), bias 0.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # Output weights 0 -> network starts as a CONSTANT controller and
        # learns deviations from it. Output biases set so that constant is
        # the supplied safe (kp0, ki0, kd0) point.
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias[0] = _logit((kp0 - kp_lo) / (kp_hi - kp_lo))
            self.fc2.bias[1] = _logit((ki0 - ki_lo) / (ki_hi - ki_lo))
            self.fc2.bias[2] = _logit((kd0 - kd_lo) / (kd_hi - kd_lo))

        self.register_buffer("lo", torch.tensor([kp_lo, ki_lo, kd_lo]))
        self.register_buffer("hi", torch.tensor([kp_hi, ki_hi, kd_hi]))
        self.register_buffer("safe", torch.tensor([kp0, ki0, kd0]))

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        z = torch.sigmoid(self.fc2(h))
        return self.lo + (self.hi - self.lo) * z      # (..., 3) = Kp, Ki, Kd


# ---------------------------------------------------------------------------
# zeta and the analytic regularizer (kept as a *prior*, not the only signal)
# ---------------------------------------------------------------------------
def zeta_est(gains, mass=1.0):
    """Damping-ratio estimate from the PID gains (Kp, Ki, Kd):

        zeta = Kd / (2 * sqrt(mass * (Kp + 0.1 * Ki))).

    zeta == 1 is critical damping; zeta < 1 is underdamped (faster, with
    overshoot); zeta > 1 is overdamped (slow, no overshoot).

    WHY `mass` MATTERS (fix #6 — corrected from a previous version that dropped
    it): `mass` is a per-loop stiffness scale that lets the SAME network design
    and the SAME target_zeta serve loops whose gain magnitudes differ by orders
    of magnitude.  The distance loop runs Kp~80 with mass=1; the heading loop
    runs Kp~2 with mass~3e-4.  Without the mass term, identical target_zeta /
    zeta-band settings would put the small-gain loops permanently outside their
    band, firing the safety fallback every step and pinning them to fixed gains
    (i.e. the heading/turn loops would silently stop being neural).  Dropping
    `mass` was therefore a real bug, not a simplification — it is restored here.
    """
    kp, ki, kd = gains[..., 0], gains[..., 1], gains[..., 2]
    return kd / (2.0 * torch.sqrt(mass * (kp + 0.1 * ki) + 1e-6))


def gain_loss(gains, track_err=None, prev_gains=None, mass=1.0, target_zeta=1.0,
              w_zeta=1.0, w_track=0.1, w_effort=1e-4, w_smooth=1e-3):
    """Analytic regularizer.  This is NOT the performance objective any more —
    it is a *prior* that keeps the network well-posed and safe between the
    sparser performance updates (drives zeta toward target, discourages huge Kp,
    smooths gain changes).  The actual closed-loop reward comes from
    `PerformanceMonitor` via the policy-gradient term in `OnlineTuner`.
    """
    z = zeta_est(gains, mass)
    loss = w_zeta * (z - target_zeta) ** 2               # drive zeta -> target
    if track_err is not None:                            # makes input matter
        loss = loss + w_track * track_err ** 2
    loss = loss + w_effort * gains[..., 0] ** 2          # discourage huge Kp
    if prev_gains is not None:                           # smooth gain changes
        loss = loss + w_smooth * ((gains - prev_gains) ** 2).sum(-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# FIX #4 — Feature builders with CONSISTENT, controller-specific semantics.
# ---------------------------------------------------------------------------
class FeatureBuilder:
    """Maps raw controller state -> a normalized 5-vector.

    The five network inputs ALWAYS carry these fixed physical meanings — this is
    the heart of fix #4.  Previously slot [3] was labelled "speed" but the
    heading loop fed it a (signed) steering trim and the turn loop fed it a turn
    rate, so identical input numbers meant different things to the same network.

        x[0] = tracking error / error_scale      (error = measurement - setpoint)
        x[1] = error rate / error_scale          (d(error)/dt)
        x[2] = clamped integral of error         (unitless, in [-1, 1])
        x[3] = actuation / actuation_scale       (this loop's control output)
        x[4] = oscillation proxy in [0, 1]       (fraction of error-rate flips)

    `setpoint_offset` is subtracted from the raw measurement to form the error.
    For the distance loop it is the stop distance; the heading/turn loops already
    pass an error so their offset is 0.  `signed_actuation` keeps the sign of
    x[3] when the actuation is bidirectional (heading steering trim) and takes
    the magnitude when it is not (forward speed, turn rate).

    Attribute names `stop` / `scale` / `pwm_max` are retained for backward
    compatibility with existing call sites.
    """

    def __init__(self, stop_dist=0.20, max_range=2.0, pwm_max=75.0, win=20,
                 signed_actuation=False, kind="generic"):
        self.stop = float(stop_dist)              # setpoint offset
        self.scale = max(float(max_range), 1e-6)  # error normalizer
        self.pwm_max = max(float(pwm_max), 1e-6)  # actuation normalizer
        self.win = int(win)
        self.signed = bool(signed_actuation)
        self.kind = kind
        self.reset()

    def reset(self):
        self.prev_e = None
        self.prev_t = None
        self.eint = 0.0
        self.sign_hist = deque(maxlen=self.win)   # bounded -> no memory growth

    def build(self, measurement, actuation, t):
        # --- input hardening (fix: safety / numerical edge cases) -----------
        # A single non-finite LiDAR/IMU sample must never poison the network or
        # the integral.  Fall back to the last good error / zero actuation.
        if not math.isfinite(measurement):
            e = self.prev_e if self.prev_e is not None else 0.0
        else:
            e = measurement - self.stop
        if not math.isfinite(actuation):
            actuation = 0.0

        if self.prev_e is None or self.prev_t is None:
            edot, dt = 0.0, 0.02
        else:
            dt = max(1e-3, t - self.prev_t)
            edot = (e - self.prev_e) / dt
        self.prev_e, self.prev_t = e, t

        self.eint = max(-1.0, min(1.0, self.eint + e * dt))   # clamped integral

        # oscillation proxy: fraction of sign flips of edot in a recent window
        self.sign_hist.append(1 if edot >= 0 else -1)
        h = list(self.sign_hist)
        flips = sum(a != b for a, b in zip(h, h[1:]))
        osc = flips / max(1, self.win - 1)

        act = actuation if self.signed else abs(actuation)

        return torch.tensor([
            e / self.scale,
            edot / self.scale,
            self.eint,
            act / self.pwm_max,
            osc,
        ], dtype=torch.float32)


class DistanceFeatureBuilder(FeatureBuilder):
    """Wall-approach loop: error = (front distance - stop), actuation = forward
    speed (PWM %, non-negative)."""
    def __init__(self, stop_dist=0.20, max_range=2.0, pwm_max=75.0, win=20):
        super().__init__(stop_dist, max_range, pwm_max, win,
                         signed_actuation=False, kind="distance")


class HeadingFeatureBuilder(FeatureBuilder):
    """Heading-hold loop: error = heading error (deg), actuation = steering trim
    (PWM %, SIGNED — left/right correction is bidirectional)."""
    def __init__(self, err_scale_deg=90.0, corr_scale=20.0, win=20):
        super().__init__(0.0, err_scale_deg, corr_scale, win,
                         signed_actuation=True, kind="heading")


class TurnFeatureBuilder(FeatureBuilder):
    """In-place turn loop: error = angle error (deg), actuation = turn rate
    (PWM %, magnitude — direction is handled by the controller, not the gains)."""
    def __init__(self, angle_scale_deg=180.0, turn_pwm=75.0, win=20):
        super().__init__(0.0, angle_scale_deg, turn_pwm, win,
                         signed_actuation=False, kind="turn")


def make_feature_builder(kind, feat_stop, feat_scale, feat_pwm, win=20):
    if kind == "distance":
        return DistanceFeatureBuilder(feat_stop, feat_scale, feat_pwm, win)
    if kind == "heading":
        return HeadingFeatureBuilder(feat_scale, feat_pwm, win)
    if kind == "turn":
        return TurnFeatureBuilder(feat_scale, feat_pwm, win)
    return FeatureBuilder(feat_stop, feat_scale, feat_pwm, win)


# ---------------------------------------------------------------------------
# FIX #1 — closed-loop performance estimator.
# ---------------------------------------------------------------------------
class PerformanceMonitor:
    """Rolling estimator of *closed-loop* control quality.

    Fed `(error, control_output)` every control step.  On demand it reduces the
    recent window to a single scalar cost `J` (lower == better) plus the named
    components, which are exactly the *consequences* of the gains the tuner chose:

        iae       mean |e|                         cumulative tracking error
        overshoot worst excursion to the wrong side of the setpoint
        osc       fraction of error-rate sign flips (ringing)
        effort    mean |u|                          actuator usage
        chatter   mean |du|                         actuator wear / noise
        settle    |e| at end of window              settling quality

    All deques are bounded -> constant memory.
    """

    def __init__(self, window=40, err_scale=1.0, eff_scale=1.0,
                 w_iae=1.0, w_over=2.0, w_osc=1.0, w_eff=0.2, w_settle=1.0):
        self.window = int(window)
        self.err_scale = max(float(err_scale), 1e-6)
        self.eff_scale = max(float(eff_scale), 1e-6)
        self.w = (w_iae, w_over, w_osc, w_eff, w_settle)
        self.reset()

    def reset(self):
        self.e = deque(maxlen=self.window)
        self.u = deque(maxlen=self.window)
        self.desc = deque(maxlen=self.window)   # sign of d(error)
        self._prev_e = None

    def push(self, e, u):
        if not (math.isfinite(e) and math.isfinite(u)):
            return
        self.e.append(e)
        self.u.append(u)
        if self._prev_e is not None:
            self.desc.append(1 if (e - self._prev_e) >= 0 else -1)
        self._prev_e = e

    def ready(self):
        return len(self.e) >= max(4, self.window // 2)

    def metrics(self):
        es = list(self.e)
        us = list(self.u)
        n = len(es)
        if n == 0:
            return float("inf"), {}
        iae = (sum(abs(v) for v in es) / n) / self.err_scale
        # Overshoot: how far the error travelled to the *opposite* side of where
        # it started (i.e. past the setpoint).  For wall approach this is the
        # robot crossing inside the stop distance; for heading/turn it is ringing
        # past the target heading.
        sign0 = 1.0 if es[0] >= 0 else -1.0
        over = max(0.0, max(-sign0 * v for v in es)) / self.err_scale
        d = list(self.desc)
        flips = sum(a != b for a, b in zip(d, d[1:]))
        osc = flips / max(1, len(d) - 1)
        eff = (sum(abs(v) for v in us) / n) / self.eff_scale
        chatter = (sum(abs(us[i] - us[i - 1]) for i in range(1, n))
                   / max(1, n - 1)) / self.eff_scale
        settle = abs(es[-1]) / self.err_scale
        w = self.w
        J = (w[0] * iae + w[1] * over + w[2] * osc
             + w[3] * (eff + chatter) + w[4] * settle)
        return J, dict(iae=iae, overshoot=over, osc=osc, effort=eff,
                       chatter=chatter, settle=settle, J=J)

    def cost(self):
        return self.metrics()[0]


# ---------------------------------------------------------------------------
# Runtime tuner: throttled updates, performance learning, persistence,
# anti-drift, rate-limiting and safety fallback.
# ---------------------------------------------------------------------------
class OnlineTuner:
    def __init__(self, lr=1e-3, update_hz=8.0, max_rate=20.0,
                 train=False, weights_path=None,
                 kp_range=(KP_LO, KP_HI), ki_range=(KI_LO, KI_HI),
                 kd_range=(KD_LO, KD_HI), init_gains=None,
                 mass=1.0, target_zeta=0.8, zeta_band=0.4,
                 feat_stop=0.20, feat_scale=2.0, feat_pwm=75.0,
                 clamp_near_stop=True,
                 # ---------- new (all optional, backward compatible) ----------
                 name="tuner", feat_kind="distance",
                 checkpoint_dir=None, checkpoint_path=None, save_every=200,
                 explore_std=0.0, perf_window=40, perf_weights=None,
                 lr_decay=0.9995, min_lr=1e-5, grad_clip=1.0,
                 w_pg=1.0, w_reg=1.0,
                 auto_freeze=True, freeze_patience=40, freeze_tol=0.02,
                 rollback=True, rollback_factor=1.5, rollback_patience=8,
                 seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.name = name
        self.net = GainMLP(kp_range=kp_range, ki_range=ki_range,
                           kd_range=kd_range, init_gains=init_gains)
        self.net.eval()

        self.feat = make_feature_builder(feat_kind, feat_stop, feat_scale,
                                         feat_pwm, win=20)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.train_online = train
        self.period = 1.0 / update_hz
        self.max_rate = max_rate                # max gain change per second
        self.last_update = 0.0
        self.mass = mass                        # accepted, ignored by zeta_est
        self.target_zeta = target_zeta          # damping target this loop aims for
        self.zeta_band = zeta_band              # +/- band before safety fallback
        self.clamp_near_stop = clamp_near_stop  # distance loop: snap to safe near wall

        self.safe = self.net.safe.clone()       # per-loop safe fallback gains
        self.gains = self.safe.clone()
        self.prev_out = self.safe.clone()
        self._last_meas = None

        # ---- FIX #1: performance learning ---------------------------------
        # Exploration std (fraction of each gain's range).  Policy-gradient
        # learning needs a little action variance to discover better gains.
        # Default 0.0 == pure deterministic inference (safest); the robot config
        # enables a small value.  Exploration is applied ONLY while training and
        # is still bounded, rate-limited and zeta-checked downstream.
        self.explore_std = float(explore_std)
        self.sigma = (self.explore_std * (self.net.hi - self.net.lo)).detach()
        pw = perf_weights or {}
        self.perf = PerformanceMonitor(window=perf_window,
                                       err_scale=feat_scale, eff_scale=feat_pwm,
                                       **pw)
        self.perf_window = int(perf_window)
        self.batch = deque(maxlen=self.perf_window)   # (x, applied_gains) per step
        self.w_pg = float(w_pg)
        self.w_reg = float(w_reg)
        self.grad_clip = float(grad_clip)
        self.baseline = None                    # EMA of window cost (reward ref)
        self._adv_sq = 1.0                      # running advantage variance
        self.last_metrics = None

        # ---- FIX #5: anti-drift -------------------------------------------
        self.lr_decay = float(lr_decay)
        self.min_lr = float(min_lr)
        self.auto_freeze = bool(auto_freeze)
        self.freeze_patience = int(freeze_patience)
        self.freeze_tol = float(freeze_tol)
        self.rollback = bool(rollback)
        self.rollback_factor = float(rollback_factor)
        self.rollback_patience = int(rollback_patience)
        self.best_cost = float("inf")
        self.best_state = None                  # best-performing weights (in RAM)
        self.ema_cost = None
        self._since_improve = 0
        self._rollback_count = 0
        self.update_count = 0

        # ---- FIX #2: persistence ------------------------------------------
        self._lock = threading.Lock()           # guards on-disk writes
        self.save_every = int(save_every)
        self.ckpt_path, self.best_path = self._resolve_paths(
            checkpoint_dir, checkpoint_path, weights_path)
        # Auto-load if a checkpoint exists; falls back silently to safe init.
        if not self._load(self.ckpt_path):
            self._load(self.best_path)

    # ------------------------------------------------------------------ utils
    def _resolve_paths(self, checkpoint_dir, checkpoint_path, weights_path):
        if checkpoint_path:
            root, ext = os.path.splitext(checkpoint_path)
            return checkpoint_path, root + "_best" + (ext or ".pt")
        if checkpoint_dir:
            return (os.path.join(checkpoint_dir, f"{self.name}.pt"),
                    os.path.join(checkpoint_dir, f"{self.name}_best.pt"))
        if weights_path:        # legacy single-file argument
            root, ext = os.path.splitext(weights_path)
            return weights_path, root + "_best" + (ext or ".pt")
        return None, None

    @staticmethod
    def _as_tuple(g):
        return float(g[0]), float(g[1]), float(g[2])

    def metrics(self):
        """Last computed performance metrics (for logging); may be None."""
        return self.last_metrics

    # --------------------------------------------------------------- main API
    def maybe_update(self, meas, actuation, t):
        """Call every control loop; returns the current (kp, ki, kd) floats.

        `meas` is the raw measurement (front distance, heading error, or angle
        error depending on the loop) and `actuation` is this loop's last control
        output (forward speed / steering trim / turn rate).
        """
        # Input hardening — never let a bad sample propagate.
        if not math.isfinite(meas):
            meas = self._last_meas if self._last_meas is not None else self.feat.stop
        if not math.isfinite(actuation):
            actuation = 0.0
        self._last_meas = meas

        # Throttle: only do real work at `update_hz`.
        if self.last_update and (t - self.last_update) < self.period:
            return self._as_tuple(self.gains)
        dt = (t - self.last_update) if self.last_update else self.period
        self.last_update = t

        with torch.no_grad():
            x = self.feat.build(meas, actuation, t)
            mu = self.net(x)                          # deterministic mean gains

            # Exploration (training only) so the learner can discover whether
            # slightly different gains perform better.
            if self.train_online and self.explore_std > 0.0:
                a = mu + self.sigma * torch.randn_like(mu)
            else:
                a = mu
            a = torch.maximum(self.net.lo, torch.minimum(self.net.hi, a))

            # Rate-limit each gain so hardware never sees a sudden jump.
            step = self.max_rate * dt
            a = torch.maximum(self.prev_out - step,
                              torch.minimum(self.prev_out + step, a))

            # NaN/inf guard on the gains themselves.
            if torch.isnan(a).any() or torch.isinf(a).any():
                a = self.safe.clone()

            # SAFETY FALLBACK (preserved): revert to safe gains if zeta leaves
            # the band around its target, or — distance loop only — if we are
            # at/inside the stop distance.
            z = float(zeta_est(a, self.mass))
            lo_z = self.target_zeta - self.zeta_band
            hi_z = self.target_zeta + self.zeta_band
            near_limit = self.clamp_near_stop and (meas <= self.feat.stop)
            if near_limit or not (lo_z <= z <= hi_z):
                a = self.safe.clone()

            self.prev_out = a.clone()
            self.gains = a

        # ---- performance bookkeeping + learning ----------------------------
        e = meas - self.feat.stop
        self.perf.push(e, actuation)
        if self.train_online:
            self.batch.append((x.detach(), a.detach()))   # what we actually did
            if len(self.batch) >= self.perf_window and self.perf.ready():
                self._learn_from_window()

        # Periodic checkpoint.
        self.update_count += 1
        if (self.save_every and self.train_online
                and self.update_count % self.save_every == 0):
            self.save_checkpoint()

        return self._as_tuple(self.gains)

    # -------------------------------------------------- FIX #1: the learner
    def _learn_from_window(self):
        """Advantage-weighted policy-gradient step from REAL performance.

        The window's measured cost J is compared against a running baseline; a
        window that beat the baseline gets a positive advantage and the network
        is nudged to *reproduce* the gains it used there, while worse-than-average
        windows are pushed away.  This is the mechanism by which the tuner learns
        from the consequences of previously selected gains, not just the current
        error.  The analytic `gain_loss` is kept as a small regularizer/prior.
        """
        J, comp = self.perf.metrics()
        self.last_metrics = comp
        if not math.isfinite(J):
            self.batch.clear()
            return

        X = torch.stack([b[0] for b in self.batch])       # (N, 5)
        A_act = torch.stack([b[1] for b in self.batch])   # (N, 3) gains applied

        self.net.train()
        mu = self.net(X)                                  # (N, 3) policy mean

        # Prior / regularizer (keeps zeta sane + smooth between perf updates).
        loss = self.w_reg * gain_loss(
            mu, track_err=None, prev_gains=self.prev_out.detach(),
            mass=self.mass, target_zeta=self.target_zeta,
            w_track=0.0, w_effort=1e-4, w_smooth=1e-3)

        # Policy-gradient term (needs exploration variance + a baseline).
        if self.explore_std > 0.0 and self.baseline is not None:
            advantage = self.baseline - J                 # >0 == better than avg
            self._adv_sq = 0.9 * self._adv_sq + 0.1 * (advantage * advantage)
            A = advantage / (math.sqrt(self._adv_sq) + 1e-6)
            A = max(-3.0, min(3.0, A))                     # clip for stability
            sigma = self.sigma.clamp_min(1e-6)
            logp = -0.5 * (((A_act - mu) / sigma) ** 2).sum(-1)   # (N,)
            pg = -float(A) * logp.mean()                   # minimize -A*logp
            loss = loss + self.w_pg * pg

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
        self.net.eval()

        # Update the reward baseline AFTER it has been used this step.
        self.baseline = J if self.baseline is None else 0.9 * self.baseline + 0.1 * J

        # FIX #5: learning-rate decay (bounded below).
        for g in self.opt.param_groups:
            g["lr"] = max(self.min_lr, g["lr"] * self.lr_decay)

        self._update_best_and_guards(J)
        self.batch.clear()

    # ------------------------------------------- FIX #5: best / freeze / rollback
    def _update_best_and_guards(self, J):
        self.ema_cost = J if self.ema_cost is None else 0.8 * self.ema_cost + 0.2 * J

        improved = (self.best_cost - J) > self.freeze_tol * max(1e-6, self.best_cost)
        if J < self.best_cost:
            self.best_cost = J
            # Snapshot the best-performing weights (RAM + disk).
            self.best_state = {k: v.detach().clone()
                               for k, v in self.net.state_dict().items()}
            self.save_checkpoint(best=True)

        # Auto-freeze: once we stop meaningfully improving, stop training so the
        # robot cannot slowly unlearn good behavior over long runs.
        self._since_improve = 0 if improved else self._since_improve + 1
        if self.auto_freeze and self._since_improve >= self.freeze_patience:
            if self.train_online:
                self.train_online = False
                self.save_checkpoint()
                print(f"[{self.name}] performance stabilized -> training frozen "
                      f"(best J={self.best_cost:.3f})")

        # Rollback: if recent performance is much worse than our best for a
        # sustained period, restore the best weights and clear the optimizer's
        # momentum (which is now pointing the wrong way).
        if (self.rollback and self.best_state is not None
                and self.ema_cost > self.rollback_factor * self.best_cost):
            self._rollback_count += 1
            if self._rollback_count >= self.rollback_patience:
                self.net.load_state_dict(self.best_state)
                self._reset_optimizer()
                self.ema_cost = self.best_cost
                self._rollback_count = 0
                print(f"[{self.name}] performance degraded -> rolled back to best "
                      f"(J={self.best_cost:.3f})")
        else:
            self._rollback_count = 0

    def _reset_optimizer(self):
        lr = self.opt.param_groups[0]["lr"]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def freeze(self):
        """Switch to inference-only mode (no further weight changes)."""
        self.train_online = False

    def unfreeze(self):
        self.train_online = True

    # --------------------------------------------------- FIX #2: persistence
    def _ckpt_dict(self):
        lo, hi = self.net.lo, self.net.hi
        return {
            "format_version": CHECKPOINT_FORMAT,
            "name": self.name,
            "net_state": self.net.state_dict(),
            "opt_state": self.opt.state_dict(),
            "best_state": self.best_state,
            "best_cost": self.best_cost,
            "baseline": self.baseline,
            "ema_cost": self.ema_cost,
            "update_count": self.update_count,
            "lr": self.opt.param_groups[0]["lr"],
            "kp_range": (float(lo[0]), float(hi[0])),
            "ki_range": (float(lo[1]), float(hi[1])),
            "kd_range": (float(lo[2]), float(hi[2])),
            "target_zeta": self.target_zeta,
        }

    def save_checkpoint(self, best=False):
        """Atomic checkpoint write (tmp file + os.replace) so a crash mid-write
        can never corrupt the live checkpoint."""
        path = self.best_path if best else self.ckpt_path
        if not path:
            return
        with self._lock:
            try:
                d = os.path.dirname(path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = path + ".tmp"
                torch.save(self._ckpt_dict(), tmp)
                os.replace(tmp, path)            # atomic on POSIX
            except Exception as e:               # never crash the control loop
                print(f"[{self.name}] checkpoint save failed (ignored): {e}")

    def save_final(self):
        """Call on shutdown to persist the latest learning."""
        self.save_checkpoint()

    def _ranges_match(self, ck):
        lo, hi = self.net.lo, self.net.hi
        want = {
            "kp_range": (float(lo[0]), float(hi[0])),
            "ki_range": (float(lo[1]), float(hi[1])),
            "kd_range": (float(lo[2]), float(hi[2])),
        }
        for k, v in want.items():
            saved = tuple(ck.get(k, ()))
            if saved and not all(abs(a - b) < 1e-6 for a, b in zip(saved, v)):
                return False
        return True

    def _load(self, path):
        """Load a checkpoint if present.  Tolerates missing/corrupt/old files:
        on any problem it warns and leaves the safe initialization in place."""
        if not path or not os.path.exists(path):
            return False
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[{self.name}] checkpoint unreadable (ignored): {e}")
            return False
        try:
            # Backward compat: a raw state_dict (old format, no version key).
            if not isinstance(ck, dict) or "format_version" not in ck:
                self.net.load_state_dict(ck)
                print(f"[{self.name}] loaded legacy weights from {path}")
                return True
            if ck.get("format_version") != CHECKPOINT_FORMAT:
                print(f"[{self.name}] checkpoint version "
                      f"{ck.get('format_version')} != {CHECKPOINT_FORMAT} -> ignoring")
                return False
            if not self._ranges_match(ck):
                print(f"[{self.name}] checkpoint gain ranges differ -> ignoring")
                return False
            self.net.load_state_dict(ck["net_state"])
            try:
                self.opt.load_state_dict(ck["opt_state"])
            except Exception:
                pass                             # optimizer state is non-critical
            self.best_state = ck.get("best_state")
            self.best_cost = ck.get("best_cost", float("inf"))
            self.baseline = ck.get("baseline")
            self.ema_cost = ck.get("ema_cost")
            self.update_count = ck.get("update_count", 0)
            lr = ck.get("lr")
            if lr:
                for g in self.opt.param_groups:
                    g["lr"] = lr
            print(f"[{self.name}] loaded checkpoint from {path} "
                  f"(updates={self.update_count}, best J={self.best_cost:.3f})")
            return True
        except Exception as e:
            print(f"[{self.name}] checkpoint apply failed (using safe init): {e}")
            return False


# ---------------------------------------------------------------------------
# Optional: offline pretraining so it deploys already-good (safer than
# learning on the live robot). Feed it real logged states if you have them.
# ---------------------------------------------------------------------------
def pretrain(net, steps=3000, batch=256, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for i in range(steps):
        # random plausible situations: [e, edot, eint, v, osc]
        x = torch.randn(batch, 5) * 0.3
        x[:, 4] = torch.rand(batch)                     # osc in [0,1]
        g = net(x)
        # track_err proxy = |e|; this teaches "far -> stronger, near -> gentler"
        loss = gain_loss(g, track_err=x[:, 0].abs(), prev_gains=None)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 500 == 0:
            print(f"step {i:5d}  loss {loss.item():.4f}  "
                  f"mean zeta {float(zeta_est(g).mean()):.3f}")
    net.eval()
    return net


if __name__ == "__main__":
    # ----- 1) network sanity ------------------------------------------------
    net = GainMLP()
    g0 = net(torch.zeros(5))
    print("Init gains:", [round(float(v), 2) for v in g0],
          " zeta:", round(float(zeta_est(g0)), 3))      # expect zeta ~= 1.0

    # ----- 2) performance-driven online tuner (fix #1 + #2 + #5) ------------
    import tempfile
    tmpdir = tempfile.mkdtemp()
    tuner = OnlineTuner(train=True, explore_std=0.05, update_hz=1000.0,
                        perf_window=20, save_every=50, name="selftest",
                        checkpoint_dir=tmpdir, feat_kind="distance",
                        target_zeta=0.9, zeta_band=0.6, seed=0)

    # Simulate a 1-D wall approach so the perf monitor sees real consequences.
    dist, t = 1.5, 0.0
    speed = 0.0
    for i in range(1500):
        kp, ki, kd = tuner.maybe_update(dist, speed, t)
        speed = max(0.0, (dist - 0.2) * 40.0)            # crude proportional plant
        dist = max(0.05, dist - 0.0008 * (speed / 75.0) + 0.0006 * math.sin(i * 0.3))
        if dist <= 0.25:
            dist = 1.5                                   # start a new leg
        t += 0.001
    print("After training:", [round(v, 2) for v in (kp, ki, kd)],
          "metrics:", {k: round(v, 3) for k, v in (tuner.metrics() or {}).items()})
    tuner.save_final()

    # ----- 3) persistence round-trip (fix #2) -------------------------------
    tuner2 = OnlineTuner(train=False, name="selftest", checkpoint_dir=tmpdir,
                         feat_kind="distance", target_zeta=0.9, zeta_band=0.6)
    print("Reloaded update_count:", tuner2.update_count,
          " best J:", round(tuner2.best_cost, 3))

    # ----- 4) corrupt-file tolerance (fix #2) -------------------------------
    with open(os.path.join(tmpdir, "selftest.pt"), "wb") as f:
        f.write(b"not a checkpoint")
    tuner3 = OnlineTuner(train=False, name="selftest", checkpoint_dir=tmpdir,
                         feat_kind="distance")
    print("Corrupt-file handled, safe gains:",
          [round(v, 2) for v in tuner3._as_tuple(tuner3.gains)])

    # ----- 5) original far/near demonstration -------------------------------
    pretrain(net)
    g1 = net(torch.tensor([0.8, 0.0, 0.0, 0.5, 0.1]))   # far from wall
    g2 = net(torch.tensor([0.05, 0.0, 0.0, 0.3, 0.7]))  # close + ringing
    print("Far :", [round(float(v), 2) for v in g1], "zeta", round(float(zeta_est(g1)), 3))
    print("Near:", [round(float(v), 2) for v in g2], "zeta", round(float(zeta_est(g2)), 3))
