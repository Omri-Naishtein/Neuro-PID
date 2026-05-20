"""
neuro_pid.py
============
Physics-tuned PID controller library.

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

Typical usage
-------------
    from neuro_pid import PID

    ctrl = PID(setpoint=0, feedback=True, graph=True, save_path="turn.png")

    # inside your sensor loop:
    output = ctrl(error)          # or ctrl.step(error)
    if ctrl.done:
        ctrl.finish()
        break

    # at the end (interrupt, timeout, etc.):
    ctrl.finish()
"""

from __future__ import annotations

import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from simple_pid import PID as _SimplePID


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

    Parameters
    ----------
    setpoint        : float  – target value that the *error* is measured against
                               (pass error = measurement - setpoint, or pre-compute
                               error = angle_diff(target, yaw) and use setpoint=0)
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
    feedback        : bool   – stream telemetry to terminal    (default True)
    graph           : bool   – save a telemetry PNG on finish()(default False)
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
        feedback: bool = True,
        graph: bool = False,
        save_path: str = "pid_plot.png",
        label: str = "",
    ):
        # ── store config ──
        self.setpoint       = setpoint
        self.zeta           = zeta
        self.omega_n        = omega_n
        self.mass           = mass
        self.d_alpha        = d_alpha
        self.output_max     = output_max
        self.output_min     = output_min
        self.imax           = imax
        self.pid_hz         = pid_hz
        self.slow_zone      = slow_zone
        self.slow_output_max = slow_output_max
        self.done_threshold = done_threshold
        self.max_slew       = max_slew
        self.feedback       = feedback
        self.graph          = graph
        self.save_path      = save_path
        self.label          = label

        # ── physics-based gains ──
        Kp, Ki, Kd, gamma_used = compute_gains(zeta, omega_n, mass, gamma)
        self.Kp    = Kp
        self.Ki    = Ki
        self.Kd    = Kd
        self.gamma = gamma_used

        # ── inner PID (error is fed directly → inner setpoint stays 0) ──
        self._pid = _SimplePID(Kp, Ki, Kd, setpoint=0.0)
        self._pid.output_limits = (-output_max, output_max)
        self._pid.sample_time   = 1.0 / pid_hz
        try:
            self._pid.integral_limits = (-imax, imax)
        except AttributeError:
            pass  # older simple-pid versions expose no integral_limits

        # ── state ──
        self._last_output   = 0.0
        self._error_filt    = None     # running low-pass state
        self._t0            = None
        self.done           = False    # set True when |error| < done_threshold

        # ── telemetry buffers ──
        self._times         : list[float] = []
        self._raw_errors    : list[float] = []
        self._filt_errors   : list[float] = []
        self._outputs       : list[float] = []
        self._speeds        : list[float] = []

        if feedback:
            self._print_banner()

    # ── banner ────────────────────────────────────────────────────────────────
    def _print_banner(self) -> None:
        w = 60
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
        print(f"  Setpoint             = {self.setpoint}")
        if self.label:
            print(f"  Label                = {self.label}")
        print(f"  Graph on finish      = {self.graph}")
        if self.graph:
            print(f"  Save path            = {self.save_path}")
        print("=" * w)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(
        self,
        raw_error: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Feed one error sample and return the signed PID output.

        The caller computes error as  (setpoint − measurement)  for linear
        sensors, or  angle_diff(target, yaw)  for angular ones.

        Parameters
        ----------
        raw_error  : latest error reading from your sensor
        timestamp  : monotonic time of this sample; auto-filled if None

        Returns
        -------
        float  – signed output value.
                 Magnitude ≥ output_min when active, 0.0 when done.
                 Positive → one direction, negative → the other.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # initialise timer on first call
        if self._t0 is None:
            self._t0 = timestamp

        t = timestamp - self._t0

        # ── derivative low-pass filter on error ──────────────────────────────
        if self._error_filt is None:
            self._error_filt = raw_error
        else:
            self._error_filt = (
                self.d_alpha * raw_error
                + (1.0 - self.d_alpha) * self._error_filt
            )
        filt_error = self._error_filt

        # ── done check (use raw so we don't lag the stop) ────────────────────
        if abs(raw_error) < self.done_threshold:
            self.done = True
            self._record(t, raw_error, filt_error, 0.0, 0.0)
            if self.feedback:
                print(
                    f"  [DONE] t={t:.3f}s"
                    f"  raw_err={raw_error:+.2f}"
                    f"  filt_err={filt_error:+.2f}"
                )
            return 0.0

        # ── adaptive output limits (slow zone) ───────────────────────────────
        if abs(filt_error) < self.slow_zone:
            self._pid.output_limits = (-self.slow_output_max, self.slow_output_max)
        else:
            self._pid.output_limits = (-self.output_max, self.output_max)

        # ── PID compute ──────────────────────────────────────────────────────
        # simple_pid expects (measurement); with setpoint=0 we pass filt_error
        raw_output = self._pid(filt_error)

        # ── slew-rate limiter ────────────────────────────────────────────────
        output = float(
            max(
                self._last_output - self.max_slew,
                min(self._last_output + self.max_slew, raw_output),
            )
        )
        self._last_output = output

        # speed = |output|, floored at output_min so motors don't stall
        speed = max(abs(output), self.output_min)

        self._record(t, raw_error, filt_error, speed, output)

        if self.feedback:
            direction = "→" if output > 0 else "←"
            print(
                f"  {direction} t={t:.3f}s"
                f"  raw={raw_error:+7.2f}"
                f"  filt={filt_error:+7.2f}"
                f"  out={output:+6.1f}"
                f"  spd={speed:5.1f}"
            )

        return output

    # convenience: make the object callable
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
        raw_err: float,
        filt_err: float,
        speed: float,
        output: float,
    ) -> None:
        self._times.append(t)
        self._raw_errors.append(raw_err)
        self._filt_errors.append(filt_err)
        self._speeds.append(speed)
        self._outputs.append(output)

    # ── graph ─────────────────────────────────────────────────────────────────
    def get_graph(self) -> plt.Figure:
        """
        Build and return a matplotlib Figure with three telemetry subplots.
        The caller can show, save, or embed the figure as needed.
        No side-effects: the figure is NOT saved to disk here.
        """
        return self._build_figure()

    def finish(self) -> None:
        """
        Finalise the session.
        - Prints a summary line if feedback=True.
        - Saves the telemetry PNG if graph=True.
        Call once when your control loop exits (normally or via interrupt).
        """
        if self.feedback and self._times:
            duration = self._times[-1] - self._times[0]
            print(
                f"  Session: {duration:.2f}s"
                f"  samples={len(self._times)}"
                f"  final_err={self._raw_errors[-1]:+.2f}"
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
        ts = self._times

        subtitle = (
            f"ζ={self.zeta}  ω₀={self.omega_n} rad/s"
            f"  γ={self.gamma:.3f}  m={self.mass}  α={self.d_alpha}"
            f"   →   Kp={self.Kp:.3f}  Ki={self.Ki:.3f}  Kd={self.Kd:.3f}"
        )
        title = "neuro_pid — Telemetry"
        if self.label:
            title += f"  [{self.label}]"

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig.suptitle(title + "\n" + subtitle, fontsize=12, fontweight="bold")

        # ── subplot 1: raw vs filtered error ──────────────────────────────────
        ax1 = axes[0]
        ax1.plot(
            ts, self._raw_errors,
            color="#90CAF9", lw=1.0, alpha=0.65,
            label="Raw error (sensor)",
        )
        ax1.plot(
            ts, self._filt_errors,
            color="#2196F3", lw=1.8,
            label="Filtered error  (PID input)",
        )
        ax1.axhline(0.0,  color="#F44336", lw=1.2, ls="--", label="Zero-error target")
        ax1.axhline( self.done_threshold, color="#4CAF50", lw=0.8, ls=":",
                     label=f"±done threshold ({self.done_threshold})")
        ax1.axhline(-self.done_threshold, color="#4CAF50", lw=0.8, ls=":")
        ax1.set_ylabel("Error  (sensor units)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1.grid(True, which="major", alpha=0.4)
        ax1.grid(True, which="minor", alpha=0.15)

        # ── subplot 2: PID output (signed) ────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(
            ts, self._outputs,
            color="#FF9800", lw=1.8,
            label="PID output  (signed)",
        )
        ax2.axhline(0, color="grey", lw=0.8, ls=":")
        ax2.axhline( self.slow_output_max, color="#795548",
                     lw=0.8, ls="--", label=f"±slow zone cap ({self.slow_output_max})")
        ax2.axhline(-self.slow_output_max, color="#795548", lw=0.8, ls="--")
        ax2.set_ylabel("PID output")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.grid(True, which="major", alpha=0.4)
        ax2.grid(True, which="minor", alpha=0.15)

        # ── subplot 3: motor speed (unsigned PWM) ─────────────────────────────
        ax3 = axes[2]
        ax3.plot(
            ts, self._speeds,
            color="#9C27B0", lw=1.8,
            label="Motor speed  (PWM %)",
        )
        ax3.axhline(self.output_min, color="#CE93D8",
                    lw=0.8, ls="--", label=f"output_min ({self.output_min})")
        ax3.axhline(self.output_max, color="#6A1B9A",
                    lw=0.8, ls="--", label=f"output_max ({self.output_max})")
        ax3.set_ylabel("Speed  (PWM %)")
        ax3.set_xlabel("Time  (s)")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax3.grid(True, which="major", alpha=0.4)
        ax3.grid(True, which="minor", alpha=0.15)

        plt.tight_layout()
        return fig