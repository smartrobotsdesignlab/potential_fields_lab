"""Synthetic LDS-01 scan: 360 beams, 1 deg, ray-cast against circles + walls.
Noise model (assumption, labelled): spec 'precision +/-10 mm (<0.5 m), +/-3.5 % (>=0.5 m)'
taken as ~2 sigma -> sigma = 5 mm below 0.5 m, 1.75 % of range above.
Optional systematic scale error models the 'accuracy +/-5 %' spec line."""
import numpy as np

N = 360
ANGLE_MIN = 0.0
ANGLE_INC = 2 * np.pi / N


def raycast(circles, walls_box, rng_max=3.5):
    """circles: list of (cx, cy, r) in laser frame; walls_box: (xmin,xmax,ymin,ymax) or None."""
    a = ANGLE_MIN + ANGLE_INC * np.arange(N)
    d = np.stack([np.cos(a), np.sin(a)], 1)
    out = np.full(N, np.inf)
    for cx, cy, r in circles:
        c = np.array([cx, cy])
        b = d @ c                      # projection
        disc = b * b - (c @ c - r * r)
        hit = disc >= 0
        t = b - np.sqrt(np.where(hit, disc, 0))
        t = np.where(hit & (t > 0), t, np.inf)
        out = np.minimum(out, t)
    if walls_box is not None:
        xmin, xmax, ymin, ymax = walls_box
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(d[:, 0] > 0, xmax / d[:, 0], np.where(d[:, 0] < 0, xmin / d[:, 0], np.inf))
            ty = np.where(d[:, 1] > 0, ymax / d[:, 1], np.where(d[:, 1] < 0, ymin / d[:, 1], np.inf))
        out = np.minimum(out, np.minimum(tx, ty))
    out[out > rng_max] = np.inf
    return out


def noisy(ranges, rng, scale_bias=0.0):
    r = ranges.copy()
    f = np.isfinite(r)
    sig = np.where(r < 0.5, 0.005, 0.0175 * r)
    r[f] = r[f] * (1 + scale_bias) + rng.normal(0, sig[f])
    r[(r < 0.12)] = np.inf          # LDS-01 lower limit -> invalid
    return r
