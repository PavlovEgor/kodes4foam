#!/usr/bin/env python3
"""
Laminar flame speed of tutorials/1D_premixed_flame, and Cantera's for the same
mixture, mechanism and transport model.

The flame speed of the case is the inlet velocity less the velocity of the
front, the front being where the temperature crosses the mean of its lowest
and highest value and its velocity a straight-line fit over the written times
after --from. The cells are taken as uniform along x.

The reference is a freely propagating flame computed by Cantera with what
constant/canteraProperties and 0/ say: the mechanism, the diffusion model,
Soret on or off, and the temperature, pressure and composition of the inlet.
Cantera has thermal diffusion with multicomponent transport only, so a
mixture-averaged case with Soret is compared with that.

Usage:
    flameSpeed.py <case> [--from TIME] [--no-reference]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from foamlib import FoamCase, FoamFile


def front_positions(case):
    """(times, front positions) of the written times after 0."""
    points = np.asarray(FoamFile(case.path / "constant" / "polyMesh" / "points")[None])
    x0, x1 = points[:, 0].min(), points[:, 0].max()

    times, fronts = [], []
    for time in case:
        if time.time <= 0:
            continue
        T = np.asarray(time["T"].internal_field, dtype=float)
        if T.ndim == 0:
            continue
        x = x0 + (np.arange(T.size) + 0.5) * (x1 - x0) / T.size
        level = 0.5 * (T.min() + T.max())
        i = int(np.argmax(T > level))
        if i == 0:
            continue
        fronts.append(x[i - 1] + (level - T[i - 1]) / (T[i] - T[i - 1]) * (x[i] - x[i - 1]))
        times.append(time.time)

    return np.array(times), np.array(fronts)


def inlet_state(case):
    """(velocity, T, p, mass fractions) of the inlet in 0/."""
    zero = case[0]

    def value(field):
        v = zero[field].boundary_field["inlet"]["value"]
        return np.atleast_1d(np.asarray(v, dtype=float))

    Y = {}
    for path in sorted(zero.path.iterdir()):
        name = path.name
        if name in ("U", "T", "p", "p_rgh", "Ydefault") or not path.is_file():
            continue
        y = float(value(name)[0])
        if y > 0:
            Y[name] = y

    p = float(np.atleast_1d(np.asarray(zero["p"].internal_field, dtype=float))[0])
    return float(value("U")[0]), float(value("T")[0]), p, Y


def reference(case, T, p, Y):
    """Cantera's flame speed and what it was computed with."""
    import cantera as ct

    properties = FoamFile(case.path / "constant" / "canteraProperties")
    mechanism = str(properties["mechanism"]).strip('"')
    mechanism = mechanism.replace("<constant>", str(case.path / "constant"))

    soret = str(properties.get("Soret", "off")) in ("on", "yes", "true", "True")
    diffusion = properties["diffusion"]
    model = "multicomponent" if soret or diffusion == "multicomponent" else "mixture-averaged"
    if diffusion == "unityLewis":
        model = "unity-Lewis-number"

    gas = ct.Solution(mechanism)
    gas.TPY = T, p, Y

    flame = ct.FreeFlame(gas, width=0.03)
    flame.transport_model = model
    flame.soret_enabled = soret
    flame.set_refine_criteria(ratio=3, slope=0.05, curve=0.1)
    flame.solve(loglevel=0, auto=True)

    return float(flame.velocity[0]), f"{model}, Soret {'on' if soret else 'off'}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case")
    parser.add_argument("--from", dest="start", type=float, default=None,
                        help="fit the front from this time on (default: the second half of the run)")
    parser.add_argument("--no-reference", action="store_true", help="skip the Cantera flame")
    args = parser.parse_args()

    case = FoamCase(args.case)
    times, fronts = front_positions(case)
    if times.size < 3:
        sys.exit(f"{case.name}: fewer than three written times with a flame in them")

    start = args.start if args.start is not None else 0.5 * (times[0] + times[-1])
    fit = times >= start
    if fit.sum() < 3:
        sys.exit(f"fewer than three written times from {start:g} s on")

    u, T, p, Y = inlet_state(case)
    front_velocity = np.polyfit(times[fit], fronts[fit], 1)[0]
    speed = u - front_velocity

    print(f"{case.name}: T = {T:g} K, p = {p:g} Pa, Y = {Y}")
    print(f"  front at {fronts[-1] * 1e3:.3f} mm, moving at {front_velocity:+.4f} m/s"
          f" (fit over {times[fit][0]:g} .. {times[-1]:g} s, {fit.sum()} times)")
    print(f"  flame speed        {speed:.4f} m/s")

    if not args.no_reference:
        ref, what = reference(case, T, p, Y)
        print(f"  Cantera            {ref:.4f} m/s  ({what})")
        print(f"  difference         {100 * (speed / ref - 1):+.1f} %")


if __name__ == "__main__":
    main()
