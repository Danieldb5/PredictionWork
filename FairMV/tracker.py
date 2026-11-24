"""
tracker.py - smooth fair value over time with a local-level Kalman filter.

In a thin market the instantaneous estimate flickers: someone posts or cancels
a dust order, a 5-share trade prints, a sibling quote moves. Without smoothing,
a market maker re-quotes on every flicker and is easy to pick off.

State: x_t = log-odds fair value, a random walk with variance sigma^2 per hour.
Measurement: the fused estimate z_t with variance R_t (from combine.py).

    predict   P <- P + sigma^2 * dt_h
    update    K = P / (P + R),  x <- x + K (z - x),  P <- (1 - K) P

Two guards:
  * Variance floor P >= floor * R. Successive measurements are NOT independent
    (the same stale book is seen on every poll), so the filter must not grow
    overconfident from seeing the same information many times.
  * Jump reset. If |z - x| / sqrt(P + R) > jump_z, something real happened
    (news, a big trade, a new maker). Lagging it would be expensive, so we
    jump straight to the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _State:
    x: float
    P: float
    t: float


class Tracker:
    def __init__(self, floor: float = 0.35, jump_z: float = 4.0):
        self.floor, self.jump_z = floor, jump_z
        self.states: dict = {}

    def reset(self, key=None) -> None:
        if key is None:
            self.states.clear()
        else:
            self.states.pop(key, None)

    def update(self, key, z: float, R: float, t: float, sigma2: float) -> tuple[float, float, dict]:
        """Returns (x, P, info). t in epoch seconds, sigma2 in log-odds var per hour."""
        st = self.states.get(key)
        if st is None:
            self.states[key] = _State(z, R, t)
            return z, R, {"tracker": "init"}
        dt_h = max(t - st.t, 0.0) / 3600.0
        P = st.P + sigma2 * dt_h
        innov = z - st.x
        S = P + R
        if innov * innov > self.jump_z ** 2 * S:
            self.states[key] = _State(z, R, t)
            return z, R, {"tracker": "jump-reset", "innov_z": innov / S ** 0.5}
        K = P / S
        x = st.x + K * innov
        P = max((1 - K) * P, self.floor * R)
        self.states[key] = _State(x, P, t)
        return x, P, {"tracker": "update", "gain": K, "innov_z": innov / S ** 0.5}
