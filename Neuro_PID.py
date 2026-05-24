"""
neuro_pid.py
============
Physics-tuned PID controller library — no external PID dependency.

Design method
-------------
Gains are derived from a second-order closed-loop pole-placement formula
with an additional derivative pre-filter zero (γ), so the tuning knobs are
human-readable physical quantities instead of raw Kp/Ki/Kd numbers:

    ζ  – damping ratio         (0 = undamped, 1 = critically damped)
    ω₀ – natural frequency     (rad/s  — higher = faster response)
    γ  – derivative zero       (rad/s  — default ω₀/10, mild phase lead)
    m  – effective mass/inertia scaling

Resulting gains
---------------
    Kd = m · (2ζω₀ + γ)
    Kp = m · (ω₀² + 2ζω₀γ)
    Ki = m · (ω₀²γ)

Default operating point: ζ = 0.7, ω₀ = 3 rad/s
    → γ  = 0.30
    → Kp ≈ 10.26
    → Ki ≈ 2.70
    → Kd ≈ 4.50

Implementation notes
--------------------
- The derivative filter is applied only to the derivative term, not to the
  full error signal.  This keeps the proportional and integral paths clean
  and avoids adding lag to large-step responses.
- Anti-windup is a hard clamp on the integrator accumulator — simple and
  transparent, not back-calculation.
- The controller owns its own timing; just call step() as fast as you like
  and it will self-throttle to pid_hz.

Typical usage
-------------
    from neuro_pid import PID

    ctrl = PID(setpoint=0, feedback=True, graph=True, save_path="turn.png")

    # inside your sensor loop:
    output = ctrl(error)          # or ctrl.step(error)
    if ctrl.done:
        ctrl.finish()
        break

    # to reuse the same controller for the next move:
    ctrl.reset()

    # at the end (interrupt, timeout, etc.):
    ctrl.finish()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_gains(
    zeta: float,
    omega_n: float,
    mass: float = 1.0,
    gamma: Optional[float] = None,
) -> tuple[float, float, float, float]:
    """
    Compute PID gains from physical parameters.

    Returns
    -------
    (Kp, Ki, Kd, gamma_used)
    """
    if gamma is None:
        gamma = omega_n / 10.0
    Kd = mass * (2.0 * zeta * omega_n + gamma)
    Kp = mass * (omega_n ** 2 + 2.0 * zeta * omega_n * gamma)
    Ki = mass * (omega_n ** 2 * gamma)
    return Kp, Ki, Kd, gamma


def angle_diff(target: float, current: float) -> float:
    """
    Shortest signed angle from *current* to *target*, wrapped to (−180, +180].
    Works for any angular sensor that reports in degrees.
    """
    return (target - current + 180.0) % 360.0 - 180.0


# ─────────────────────────────────────────────────────────────────────────────
# PID controller class
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    """
    Physics-tuned PID controller with optional terminal feedback and plot.

    The caller is responsible for computing the error before each call:
        - Linear:  error = setpoint − measurement
        - Angular: error = angle_diff(target, yaw)

    The `setpoint` parameter is informational only (shown in the banner).
    The controller always operates on the pre-computed error you pass in.

    Parameters
    ----------
    setpoint        : float  – displayed in banner; not used in math
    zeta            : float  – damping ratio                   (default 0.7)
    omega_n         : float  – natural frequency  [rad/s]      (default 3.0)
    mass            : float  – inertia / gain scaling          (default 1.0)
    gamma           : float  – derivative pre-filter zero      (default ω₀/10)
    d_alpha         : float  – derivative low-pass coefficient (default 0.15)
                               0 → pure derivative, 1 → derivative off
    output_max      : float  – maximum output magnitude        (default 60)
    output_min      : float  – minimum non-zero output         (default 9)
    imax            : float  – integrator anti-windup clamp    (default 20)
    pid_hz          : float  – controller update-rate cap [Hz] (default 50)
    slow_zone       : float  – |error| below which output_max is reduced
                               (default 12, same units as error)
    slow_output_max : float  – output cap inside slow zone     (default 28)
    done_threshold  : float  – |error| below which done=True   (default 1.5)
    max_slew        : float  – max output change per step      (default 16)
    max_samples     : int    – telemetry ring-buffer size       (default 10 000)
    feedback        : bool   – stream telemetry to terminal    (default True)
    graph           : bool   – save a telemetry PNG on finish() (default False)
    save_path       : str    – destination path for the PNG    (default "pid_plot.png")
    label           : str    – free-text label added to plot title
    """

    # ── construction ──────────────────────────────────────────────────────────
    def __init__(
        self,
        setpoint: float = 0.0,
        zeta: float = 0.7,
        omega_n: float = 3.0,
        mass: float = 1.0,
        gamma: Optional[float] = None,
        d_alpha: float = 0.15,
        output_max: float = 60.0,
        output_min: float = 9.0,
        imax: float = 20.0,
        pid_hz: float = 50.0,
        slow_zone: float = 12.0,
        slow_output_max: float = 28.0,
        done_threshold: float = 1.5,
        max_slew: float = 16.0,
        max_samples: int = 10_000,
        feedback: bool = True,
        graph: bool = False,
        save_path: str = "pid_plot.png",
        label: str = "",
    ):
        self.setpoint        = setpoint
        self.zeta            = zeta
        self.omega_n         = omega_n
        self.mass            = mass
        self.d_alpha         = d_alpha
        self.output_max      = output_max
        self.output_min      = output_min
        self.imax            = imax
        self.pid_hz          = pid_hz
        self.slow_zone       = slow_zone
        self.slow_output_max = slow_output_max
        self.done_threshold  = done_threshold
        self.max_slew        = max_slew
        self.max_samples     = max_samples
        self.feedback        = feedback
        self.graph           = graph
        self.save_path       = save_path
        self.label           = label

        self._min_dt = 1.0 / pid_hz

        # ── physics-based gains ──
        Kp, Ki, Kd, gamma_used = compute_gains(zeta, omega_n, mass, gamma)
        self.Kp    = Kp
        self.Ki    = Ki
        self.Kd    = Kd
        self.gamma = gamma_used

        # ── mutable PID state (reset() re-zeroes these) ──
        self._integral        : float          = 0.0
        self._prev_error      : Optional[float] = None
        self._prev_deriv_filt : float          = 0.0
        self._last_output     : float          = 0.0
        self._last_step_time  : Optional[float] = None
        self._t0              : Optional[float] = None
        self.done             : bool            = False

        # ── telemetry ring-buffers (bounded memory) ──
        self._times    : deque[float] = deque(maxlen=max_samples)
        self._errors   : deque[float] = deque(maxlen=max_samples)
        self._p_terms  : deque[float] = deque(maxlen=max_samples)
        self._i_terms  : deque[float] = deque(maxlen=max_samples)
        self._d_terms  : deque[float] = deque(maxlen=max_samples)
        self._outputs  : deque[float] = deque(maxlen=max_samples)
        self._speeds   : deque[float] = deque(maxlen=max_samples)

        if feedback:
            self._print_banner()

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, clear_telemetry: bool = True) -> None:
        """
        Reset controller state so it can be reused for a new move without
        reinstantiating.  Gains and configuration are preserved.

        Parameters
        ----------
        clear_telemetry : if True (default) the history buffers are also
                          cleared; pass False to keep the data for plotting
                          across multiple moves.
        """
        self._integral        = 0.0
        self._prev_error      = None
        self._prev_deriv_filt = 0.0
        self._last_output     = 0.0
        self._last_step_time  = None
        self._t0              = None
        self.done             = False

        if clear_telemetry:
            for buf in (self._times, self._errors, self._p_terms,
                        self._i_terms, self._d_terms, self._outputs, self._speeds):
                buf.clear()

    # ── step ─────────────────────────────────────────────────────────────────
    def step(
        self,
        raw_error: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Feed one error sample and return the signed PID output.

        Parameters
        ----------
        raw_error  : error = setpoint − measurement (or angle_diff for angular)
        timestamp  : monotonic time of this sample; auto-filled if None

        Returns
        -------
        float  – signed output.
                 |output| ≥ output_min when active, 0.0 when done.
                 Positive → one direction, negative → the other.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # ── initialise on first call ──────────────────────────────────────────
        if self._t0 is None:
            self._t0 = timestamp

        t = timestamp - self._t0

        # ── rate-limit: skip step if called too fast ──────────────────────────
        if self._last_step_time is not None:
            elapsed = timestamp - self._last_step_time
            if elapsed < self._min_dt:
                return self._last_output

        dt = (
            (timestamp - self._last_step_time)
            if self._last_step_time is not None
            else self._min_dt
        )
        dt = max(dt, 1e-4)          # guard against zero / negative dt
        self._last_step_time = timestamp

        # ── done check on raw error (no filter lag on the stop decision) ──────
        if abs(raw_error) < self.done_threshold:
            self.done = True
            self._record(t, raw_error, 0.0, 0.0, 0.0, 0.0, 0.0)
            if self.feedback:
                print(f"  [DONE] t={t:.3f}s  err={raw_error:+.2f}")
            return 0.0

        # ── P term  (raw error — no filter lag) ───────────────────────────────
        p = self.Kp * raw_error

        # ── I term  (with hard anti-windup clamp) ─────────────────────────────
        self._integral += raw_error * dt
        self._integral  = max(-self.imax, min(self.imax, self._integral))
        i = self.Ki * self._integral

        # ── D term  (low-pass filter on derivative only) ──────────────────────
        if self._prev_error is None:
            deriv_raw = 0.0
        else:
            deriv_raw = (raw_error - self._prev_error) / dt

        deriv_filt = (
            self.d_alpha * deriv_raw
            + (1.0 - self.d_alpha) * self._prev_deriv_filt
        )
        self._prev_deriv_filt = deriv_filt
        self._prev_error      = raw_error

        d = self.Kd * deriv_filt

        # ── sum and apply adaptive output cap ─────────────────────────────────
        raw_output = p + i + d
        limit      = self.slow_output_max if abs(raw_error) < self.slow_zone else self.output_max
        raw_output = max(-limit, min(limit, raw_output))

        # ── slew-rate limiter ─────────────────────────────────────────────────
        output = max(
            self._last_output - self.max_slew,
            min(self._last_output + self.max_slew, raw_output),
        )
        self._last_output = output

        # ── motor speed: unsigned, floored so motors don't stall ──────────────
        speed = max(abs(output), self.output_min)

        self._record(t, raw_error, p, i, d, output, speed)

        if self.feedback:
            direction = "→" if output >= 0 else "←"
            print(
                f"  {direction} t={t:.3f}s"
                f"  err={raw_error:+7.2f}"
                f"  P={p:+6.1f}  I={i:+6.1f}  D={d:+6.1f}"
                f"  out={output:+6.1f}"
                f"  spd={speed:5.1f}"
            )

        return output

    # ── convenience: make the object callable ─────────────────────────────────
    def __call__(
        self,
        raw_error: float,
        timestamp: Optional[float] = None,
    ) -> float:
        return self.step(raw_error, timestamp)

    # ── telemetry ─────────────────────────────────────────────────────────────
    def _record(
        self,
        t: float,
        error: float,
        p: float,
        i: float,
        d: float,
        output: float,
        speed: float,
    ) -> None:
        self._times.append(t)
        self._errors.append(error)
        self._p_terms.append(p)
        self._i_terms.append(i)
        self._d_terms.append(d)
        self._outputs.append(output)
        self._speeds.append(speed)

    # ── public graph API ──────────────────────────────────────────────────────
    def get_graph(self) -> plt.Figure:
        """
        Build and return a matplotlib Figure with four telemetry subplots.
        No side-effects: the figure is NOT saved to disk here.
        """
        return self._build_figure()

    def finish(self) -> None:
        """
        Finalise the session: print a summary and (optionally) save the plot.
        Call once when your control loop exits, normally or via interrupt.
        """
        if self.feedback and self._times:
            duration = self._times[-1] - self._times[0]
            print(
                f"  Session: {duration:.2f}s"
                f"  samples={len(self._times)}"
                f"  final_err={self._errors[-1]:+.2f}"
            )

        if self.graph:
            if not self._times:
                if self.feedback:
                    print("  [neuro_pid] No data — skipping plot.")
                return
            fig = self._build_figure()
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if self.feedback:
                print(f"  Plot saved → {self.save_path}")

    # ── figure builder ────────────────────────────────────────────────────────
    def _build_figure(self) -> plt.Figure:
        ts = list(self._times)

        subtitle = (
            f"ζ={self.zeta}  ω₀={self.omega_n} rad/s"
            f"  γ={self.gamma:.3f}  m={self.mass}  α={self.d_alpha}"
            f"   →   Kp={self.Kp:.3f}  Ki={self.Ki:.3f}  Kd={self.Kd:.3f}"
        )
        title = "neuro_pid — Telemetry"
        if self.label:
            title += f"  [{self.label}]"

        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        fig.suptitle(title + "\n" + subtitle, fontsize=12, fontweight="bold")

        # ── subplot 1: error over time ────────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(ts, list(self._errors), color="#2196F3", lw=1.8, label="Error")
        ax1.axhline(0.0, color="#F44336", lw=1.2, ls="--", label="Zero-error target")
        ax1.axhline( self.done_threshold, color="#4CAF50", lw=0.8, ls=":",
                     label=f"±done threshold ({self.done_threshold})")
        ax1.axhline(-self.done_threshold, color="#4CAF50", lw=0.8, ls=":")
        ax1.set_ylabel("Error  (sensor units)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1.grid(True, which="major", alpha=0.4)
        ax1.grid(True, which="minor", alpha=0.15)

        # ── subplot 2: P / I / D term breakdown ───────────────────────────────
        ax2 = axes[1]
        ax2.plot(ts, list(self._p_terms), color="#F44336", lw=1.5, label="P")
        ax2.plot(ts, list(self._i_terms), color="#FF9800", lw=1.5, label="I")
        ax2.plot(ts, list(self._d_terms), color="#9C27B0", lw=1.5, label="D")
        ax2.axhline(0, color="grey", lw=0.8, ls=":")
        ax2.set_ylabel("PID terms")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.grid(True, which="major", alpha=0.4)
        ax2.grid(True, which="minor", alpha=0.15)

        # ── subplot 3: PID output (signed) ────────────────────────────────────
        ax3 = axes[2]
        ax3.plot(ts, list(self._outputs), color="#FF9800", lw=1.8, label="PID output  (signed)")
        ax3.axhline(0, color="grey", lw=0.8, ls=":")
        ax3.axhline( self.slow_output_max, color="#795548", lw=0.8, ls="--",
                     label=f"±slow zone cap ({self.slow_output_max})")
        ax3.axhline(-self.slow_output_max, color="#795548", lw=0.8, ls="--")
        ax3.set_ylabel("PID output")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax3.grid(True, which="major", alpha=0.4)
        ax3.grid(True, which="minor", alpha=0.15)

        # ── subplot 4: motor speed (unsigned PWM) ─────────────────────────────
        ax4 = axes[3]
        ax4.plot(ts, list(self._speeds), color="#4CAF50", lw=1.8, label="Motor speed  (PWM %)")
        ax4.axhline(self.output_min, color="#A5D6A7", lw=0.8, ls="--",
                    label=f"output_min ({self.output_min})")
        ax4.axhline(self.output_max, color="#1B5E20", lw=0.8, ls="--",
                    label=f"output_max ({self.output_max})")
        ax4.set_ylabel("Speed  (PWM %)")
        ax4.set_xlabel("Time  (s)")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax4.grid(True, which="major", alpha=0.4)
        ax4.grid(True, which="minor", alpha=0.15)

        plt.tight_layout()
        return fig

    # ── banner ────────────────────────────────────────────────────────────────
    def _print_banner(self) -> None:
        w   = 60
        sep = "─" * (w - 4)
        print("=" * w)
        print("  neuro_pid  —  Physics-tuned PID controller")
        print("=" * w)
        print(f"  ζ  (zeta)            = {self.zeta}")
        print(f"  ω₀ (omega_n)         = {self.omega_n} rad/s")
        print(f"  γ  (gamma)           = {self.gamma:.4f} rad/s")
        print(f"  m  (mass)            = {self.mass}")
        print(f"  α  (d_alpha)         = {self.d_alpha}  (derivative filter)")
        print(f"  {sep}")
        print(f"  Kp                   = {self.Kp:.4f}")
        print(f"  Ki                   = {self.Ki:.4f}")
        print(f"  Kd                   = {self.Kd:.4f}")
        print(f"  {sep}")
        print(f"  Integrator clamp     = ±{self.imax}")
        print(f"  Output range         = [{self.output_min}, {self.output_max}]")
        print(f"  Slow zone  < {self.slow_zone}  → max = {self.slow_output_max}")
        print(f"  Done threshold       = ±{self.done_threshold}")
        print(f"  Max slew / step      = {self.max_slew}")
        print(f"  Update cap           = {self.pid_hz} Hz")
        print(f"  Setpoint (display)   = {self.setpoint}")
        print(f"  Telemetry cap        = {self.max_samples} samples")
        if self.label:
            print(f"  Label                = {self.label}")
        print(f"  Graph on finish      = {self.graph}")
        if self.graph:
            print(f"  Save path            = {self.save_path}")
        print("=" * w)
