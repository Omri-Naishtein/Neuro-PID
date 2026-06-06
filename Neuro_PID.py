"""
pid_gain_mlp.py
----------------
Adaptive PID auto-tuner for the wall-following robot.

A small MLP (5 -> 16 tanh -> 3 scaled-sigmoid) maps the current driving
situation to PID gains (Kp, Ki, Kd). The gains are shaped so that the
estimated damping ratio

        zeta_est = Kd / (2 * sqrt(Kp + 0.1*Ki))

is driven toward 1 (critical damping = fastest approach with no overshoot).

Design notes
------------
* Output is a SCALED SIGMOID -> gains are HARD-BOUNDED to safe ranges.
* Output layer is initialized to a CONSTANT, critically-damped controller
  (Kp=80, Ki=0.5, Kd=17.9 -> zeta=1) so the robot is safe at step zero and
  only learns DEVIATIONS from that baseline.
* zeta=1 alone is underdetermined, so the loss also includes a tracking term
  (so the input actually matters), an effort penalty, and a smoothness term.
* Runtime is inference-only by default. Online training is optional and
  guarded by rate-limiting + a hard safety fallback.

Tested target: Jetson Orin Nano with PyTorch (JetPack). CPU is fine too.
"""

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Safe gain ranges (the scaled-sigmoid output can never leave these)
# ---------------------------------------------------------------------------
KP_LO, KP_HI = 20.0, 120.0
KI_LO, KI_HI = 0.0, 10.0
KD_LO, KD_HI = 2.0, 20.0

# Safe critically-damped starting point (used for init AND as the fallback)
KP0, KI0 = 80.0, 0.5
KD0 = 2.0 * math.sqrt(KP0 + 0.1 * KI0)          # ~= 17.89 -> zeta_est == 1.0
SAFE_GAINS = torch.tensor([KP0, KI0, KD0], dtype=torch.float32)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# The network
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

    # --- RECOMMENDED ALTERNATIVE -------------------------------------------
    # To make zeta == 1 EXACT and unconditional, output only Kp, Ki and
    # derive Kd. Replace fc2 with nn.Linear(n_hidden, 2), drop the Kd bias,
    # and use:
    #   kp = KP_LO + (KP_HI-KP_LO)*sigmoid(z[...,0])
    #   ki = KI_LO + (KI_HI-KI_LO)*sigmoid(z[...,1])
    #   kd = 2.0 * torch.sqrt(kp + 0.1*ki)
    # Then the (zeta-1)^2 loss term is unnecessary.
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# zeta and loss
# ---------------------------------------------------------------------------
def zeta_est(gains, mass=1.0):
    # Damping estimate computed only from the PID gains (Kp, Ki, Kd):
    #   ζ = Kd / (2 * sqrt(Kp + 0.1 * Ki)).
    # `mass` is accepted for API compatibility but ignored — the zeta
    # calculation now depends solely on the returned gains.
    kp, ki, kd = gains[..., 0], gains[..., 1], gains[..., 2]
    return kd / (2.0 * torch.sqrt(kp + 0.1 * ki + 1e-6))


def gain_loss(gains, track_err=None, prev_gains=None, mass=1.0, target_zeta=1.0,
              w_zeta=1.0, w_track=0.1, w_effort=1e-4, w_smooth=1e-3):
    z = zeta_est(gains, mass)
    loss = w_zeta * (z - target_zeta) ** 2               # drive ζ → target
    if track_err is not None:                            # makes input matter
        loss = loss + w_track * track_err ** 2
    loss = loss + w_effort * gains[..., 0] ** 2          # discourage huge Kp
    if prev_gains is not None:                           # smooth gain changes
        loss = loss + w_smooth * ((gains - prev_gains) ** 2).sum(-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Feature builder: robot state -> normalized 5-vector
# ---------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, stop_dist=0.20, max_range=2.0, pwm_max=75.0, win=20):
        self.stop, self.scale, self.pwm_max, self.win = stop_dist, max_range, pwm_max, win
        self.reset()

    def reset(self):
        self.prev_e = None
        self.prev_t = None
        self.eint = 0.0
        self.sign_hist = []

    def build(self, dist_front, speed, t):
        e = dist_front - self.stop
        if self.prev_e is None:
            edot, dt = 0.0, 0.02
        else:
            dt = max(1e-3, t - self.prev_t)
            edot = (e - self.prev_e) / dt
        self.prev_e, self.prev_t = e, t

        self.eint = max(-1.0, min(1.0, self.eint + e * dt))   # clamped integral

        # oscillation proxy: fraction of sign flips of edot in recent window
        self.sign_hist.append(1 if edot >= 0 else -1)
        if len(self.sign_hist) > self.win:
            self.sign_hist.pop(0)
        flips = sum(a != b for a, b in zip(self.sign_hist, self.sign_hist[1:]))
        osc = flips / max(1, self.win - 1)

        return torch.tensor([
            e / self.scale,
            edot / self.scale,
            self.eint,
            speed / self.pwm_max,
            osc,
        ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Runtime tuner: throttled updates, rate-limiting, safety fallback
# ---------------------------------------------------------------------------
class OnlineTuner:
    def __init__(self, lr=1e-3, update_hz=8.0, max_rate=20.0,
                 train=False, weights_path=None,
                 kp_range=(KP_LO, KP_HI), ki_range=(KI_LO, KI_HI),
                 kd_range=(KD_LO, KD_HI), init_gains=None,
                 mass=1.0, target_zeta=0.8, zeta_band=0.4,
                 feat_stop=0.20, feat_scale=2.0, feat_pwm=75.0,
                 clamp_near_stop=True):
        self.net = GainMLP(kp_range=kp_range, ki_range=ki_range,
                           kd_range=kd_range, init_gains=init_gains)
        if weights_path:
            self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()
        self.feat = FeatureBuilder(stop_dist=feat_stop, max_range=feat_scale,
                                   pwm_max=feat_pwm)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.train_online = train
        self.period = 1.0 / update_hz
        self.max_rate = max_rate                # max gain change per second
        self.last_update = 0.0
        self.mass = mass                        # ζ-model scale for this loop
        self.target_zeta = target_zeta          # damping target this loop aims for
        self.zeta_band = zeta_band              # ± band before safety fallback
        self.clamp_near_stop = clamp_near_stop  # distance loop: snap to safe near wall
        self.safe = self.net.safe.clone()       # per-loop safe fallback gains
        self.gains = self.safe.clone()
        self.prev_out = self.safe.clone()

    def maybe_update(self, dist_front, speed, t):
        """Call every loop; returns current (kp, ki, kd) as floats."""
        if self.last_update and (t - self.last_update) < self.period:
            return self._as_tuple(self.gains)
        dt = (t - self.last_update) if self.last_update else self.period
        self.last_update = t

        x = self.feat.build(dist_front, speed, t)

        if self.train_online:
            self.net.train()
            out = self.net(x)
            loss = gain_loss(out,
                             track_err=(dist_front - self.feat.stop),
                             prev_gains=self.prev_out.detach(),
                             mass=self.mass, target_zeta=self.target_zeta)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.net.eval()
            with torch.no_grad():
                out = self.net(x)
        else:
            with torch.no_grad():
                out = self.net(x)

        # rate-limit each gain so hardware never sees a sudden jump
        step = self.max_rate * dt
        out = torch.maximum(self.prev_out - step,
                            torch.minimum(self.prev_out + step, out))
        self.prev_out = out.clone()

        # safety fallback: revert to this loop's safe gains if ζ leaves the
        # band around its target, or (distance loop only) if we are at/inside
        # the stop distance.
        z = float(zeta_est(out, self.mass))
        lo_z = self.target_zeta - self.zeta_band
        hi_z = self.target_zeta + self.zeta_band
        near_limit = self.clamp_near_stop and (dist_front <= self.feat.stop)
        if near_limit or not (lo_z <= z <= hi_z):
            out = self.safe.clone()
            self.prev_out = out.clone()

        self.gains = out
        return self._as_tuple(out)

    @staticmethod
    def _as_tuple(g):
        return float(g[0]), float(g[1]), float(g[2])


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
    # quick self-test
    net = GainMLP()
    g0 = net(torch.zeros(5))
    print("Init gains:", [round(float(v), 2) for v in g0],
          " zeta:", round(float(zeta_est(g0)), 3))   # expect zeta ~= 1.0
    pretrain(net)
    g1 = net(torch.tensor([0.8, 0.0, 0.0, 0.5, 0.1]))   # far from wall
    g2 = net(torch.tensor([0.05, 0.0, 0.0, 0.3, 0.7]))  # close + ringing
    print("Far :", [round(float(v), 2) for v in g1], "zeta", round(float(zeta_est(g1)), 3))
    print("Near:", [round(float(v), 2) for v in g2], "zeta", round(float(zeta_est(g2)), 3))
