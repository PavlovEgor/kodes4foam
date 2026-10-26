#!/usr/bin/env python3
"""
Convert a detailed CHEMKIN-III mechanism (kinetics + thermo + transport) into
an OpenFOAM reactions/thermo.compressibleGas pair using the real, upstream
`chemkinToFoam` utility instead of a hand-rolled parser.

This exists because two things about the raw CHEMKIN files trip up OpenFOAM's
chemkinReader / chemkinToFoam:

1. PLOG (pressure-dependent Arrhenius) auxiliary lines are not supported by
   OpenFOAM's chemkinReader at all ("unknown third-body specie PLOG"). They
   are stripped here, so pressure-dependence is lost and each such reaction
   falls back to its own leading (high-pressure-limit) Arrhenius parameters.
2. The THERMO/THERMO ALL temperature-range line must be an exact 3x10-column
   fixed-width record (Fortran 3F10.0); many real-world thermo files use
   free-format spacing there instead, so it is reformatted.

`chemkinToFoam` also does not read a CHEMKIN TRANSPORT (.TRAN) file itself --
its <CHEMKINTransport> argument must already be an OpenFOAM dictionary of
per-specie sutherlandTransport coefficients. This script computes those
coefficients for real, using Cantera: for each specie it evaluates the pure
specie viscosity (via kinetic theory, from the same Lennard-Jones parameters
in the .TRAN file) over its NASA-polynomial temperature range, then fits
Sutherland's law mu(T) = As*sqrt(T)/(1 + Ts/T) with a linear least squares
solve (1/mu is linear in T^-0.5 and T^-1.5).

Requires: cantera (for parsing + transport property evaluation) and the
OpenFOAM `chemkinToFoam` binary (available once an OpenFOAM environment is
sourced).

Usage:
    chemkinToFoam.py <kinetics.CKI> <thermo.CKT> <transport.TRAN> \\
        <output_dir> [--chemkinToFoam PATH] [--n-fit-points N]
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import cantera as ct
import cantera.ck2yaml as ck2yaml


def strip_plog(kinetics_file, out_file):
    """Remove PLOG auxiliary rate lines; OpenFOAM's chemkinReader can't
    parse them and misreads "PLOG" as an unknown third-body specie."""
    removed = 0
    with open(kinetics_file) as fin, open(out_file, 'w') as fout:
        for line in fin:
            if re.match(r'^\s*PLOG\b', line):
                removed += 1
                continue
            fout.write(line)
    return removed


def fix_thermo_header(thermo_file, out_file):
    """Normalise the THERMO header of a standalone thermodynamics file into
    the shape OpenFOAM's chemkinReader accepts.

    Two things are fixed:

    - The keyword is forced to ``THERMO ALL``.  A bare ``THERMO`` puts the
      reader into the mode used for a thermo section embedded in the kinetics
      file, where no global temperature-range line is expected, so it reads
      that line as the first specie record and fails on its element columns.
    - The temperature-range line itself is rewritten into the strict 3x10
      fixed-width columns (Fortran 3F10.0); many real-world files, and
      Cantera's yaml2ck output, use free-format spacing there instead.
    """
    with open(thermo_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if re.match(r'^\s*THERMO\b', line, re.IGNORECASE):
            temps = re.findall(r'[-+]?\d+\.?\d*', lines[i + 1])
            if len(temps) < 3:
                raise ValueError(
                    f"Expected 3 temperatures after THERMO line, found "
                    f"{temps!r} in {thermo_file}"
                )
            lines[i] = 'THERMO ALL\n'
            lines[i + 1] = ''.join(f"{float(t):10.3f}" for t in temps[:3]) + '\n'
            break
    else:
        raise ValueError(f"No THERMO/THERMO ALL keyword found in {thermo_file}")

    with open(out_file, 'w') as f:
        f.writelines(lines)


def upper_case_elements(kinetics_file, out_file):
    """Upper-case the element symbols in the ELEMENTS block.

    OpenFOAM's chemkinReader only accepts fully upper-case element names, so
    the mixed-case symbols Cantera's yaml2ck writes ("Ar", "He") are rejected
    with "expected <elementName> ... but found". CHEMKIN itself is
    case-insensitive here, and the species formulae are already upper case.
    """
    with open(kinetics_file) as f:
        lines = f.readlines()

    in_elements = False
    for i, line in enumerate(lines):
        if re.match(r'^\s*ELEM', line, re.IGNORECASE):
            in_elements = True
            # Anything on the keyword line itself counts as an entry too
            head, _, tail = line.partition(' ')
            lines[i] = head + ' ' + tail.upper() if tail else line
            continue
        if not in_elements:
            continue
        if re.match(r'^\s*END\b', line, re.IGNORECASE):
            break
        lines[i] = line.upper()

    with open(out_file, 'w') as f:
        f.writelines(lines)


def fit_sutherland(gas, specie_name, n_points):
    """Fit Sutherland's law to pure-specie viscosity computed from the
    mechanism's own Lennard-Jones transport data via Cantera, over that
    specie's NASA-polynomial validity range. 1/mu = (1/As) T^-0.5
    + (Ts/As) T^-1.5 is linear in the unknowns, so an ordinary least
    squares solve is used instead of a nonlinear fit."""
    thermo = gas.species(specie_name).thermo
    t_low = max(thermo.min_temp, 200.0)
    t_high = thermo.max_temp
    temperatures = np.linspace(t_low, t_high, n_points)

    mu = np.empty_like(temperatures)
    for i, T in enumerate(temperatures):
        gas.TPX = T, ct.one_atm, {specie_name: 1.0}
        mu[i] = gas.viscosity

    basis = np.column_stack([temperatures**-0.5, temperatures**-1.5])
    (a, b), *_ = np.linalg.lstsq(basis, 1.0 / mu, rcond=None)

    As = 1.0 / a
    Ts = b * As
    return As, Ts


def write_transport_dict(species_coeffs, out_file):
    with open(out_file, 'w') as f:
        for name, (As, Ts) in species_coeffs.items():
            f.write(
                f"{name}\n{{\n"
                f"    transport\n    {{\n"
                f"        As              {As:.6e};\n"
                f"        Ts              {Ts:.4f};\n"
                f"    }}\n}}\n\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kinetics", help="CHEMKIN kinetics/reactions file (.CKI)")
    parser.add_argument("thermo", help="CHEMKIN thermodynamics file (.CKT)")
    parser.add_argument(
        "transport",
        help="Source of the Lennard-Jones transport data the Sutherland "
             "coefficients are fitted from: either a CHEMKIN transport file "
             "(.TRAN) to combine with <kinetics>/<thermo>, or a ready Cantera "
             "mechanism (.yaml, e.g. 'gri30.yaml') that already carries it",
    )
    parser.add_argument("output_dir", help="Directory to write reactions / thermo.compressibleGas into")
    parser.add_argument("--chemkinToFoam", default="chemkinToFoam", help="Path to the chemkinToFoam binary (default: rely on $PATH)")
    parser.add_argument("--n-fit-points", type=int, default=20, help="Number of temperature samples used per specie for the Sutherland fit")
    args = parser.parse_args()

    if shutil.which(args.chemkinToFoam) is None:
        sys.exit(
            f"error: '{args.chemkinToFoam}' not found. Source an OpenFOAM "
            f"environment (e.g. 'source /usr/lib/openfoam/openfoam2412/etc/bashrc') "
            f"or pass --chemkinToFoam <path>."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        kinetics_plog = tmp / "kinetics.noPlog"
        n_removed = strip_plog(args.kinetics, kinetics_plog)
        print(f"Removed {n_removed} PLOG line(s) from {args.kinetics}")

        kinetics_clean = tmp / "kinetics.CKI"
        upper_case_elements(kinetics_plog, kinetics_clean)

        thermo_clean = tmp / "thermo.CKT"
        fix_thermo_header(args.thermo, thermo_clean)

        if args.transport.endswith(('.yaml', '.yml')):
            print(f"Reading transport data from Cantera mechanism {args.transport}...")
            gas = ct.Solution(args.transport)
        else:
            print("Parsing mechanism with Cantera (ck2yaml) for transport fitting...")
            yaml_file = tmp / "mech.yaml"
            ck2yaml.convert(
                str(args.kinetics),
                thermo_file=str(args.thermo),
                transport_file=str(args.transport),
                out_name=str(yaml_file),
                permissive=True,
                quiet=True,
            )
            gas = ct.Solution(str(yaml_file))

        print(f"Fitting Sutherland coefficients for {gas.n_species} species...")
        species_coeffs = {
            name: fit_sutherland(gas, name, args.n_fit_points)
            for name in gas.species_names
        }

        transport_dict = tmp / "transport.dict"
        write_transport_dict(species_coeffs, transport_dict)

        reactions_out = out_dir / "reactions"
        thermo_out = out_dir / "thermo.compressibleGas"

        print(f"Running {args.chemkinToFoam}...")
        subprocess.run(
            [
                args.chemkinToFoam,
                str(kinetics_clean),
                str(thermo_clean),
                str(transport_dict),
                str(reactions_out),
                str(thermo_out),
            ],
            check=True,
        )

    print(f"Wrote {reactions_out} and {thermo_out}")


if __name__ == "__main__":
    main()
