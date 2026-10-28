#!/usr/bin/env python3
"""
Set up, run and read a kodes4foam tutorial for one mechanism and one chemistry
variant, with foamlib.

A tutorial case carries every mechanism it can run with,

    0.<mechanism>/          the initial conditions for that mechanism
    constant/<mechanism>    symlink to tutorials/mechanisms/<mechanism>

and setting it up for a run is four edits, always made together so that they
cannot disagree:

    0/                                  a copy of 0.<mechanism>
    constant/thermophysicalProperties   foamChemistry{,Thermo}File -> constant/<mechanism>
    constant/chemistryProperties        chemistryType/method and kodesSetting
    system/controlDict                  libs

The chemistry variants are

    standard            the chemistry model OpenFOAM ships, on the CPU
    kodes               kodesChemistryModel, KODES on the GPU

Every case's ./Allrun is `kodesCase.py run <case> "$@"`. The functions are
also the API utilities/parametricStudy.ipynb drives the cases with.

Usage:
    kodesCase.py list  <case>
    kodesCase.py setup <case> <mechanism> <variant>
    kodesCase.py run   <case> [mechanism] [variant] [--np N]
    kodesCase.py log   <log>...
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from foamlib import CalledProcessError, FoamCase

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"

VARIANTS = ("standard", "kodes")

KODES_LIBRARY = "libkodesChemistryModel.so"


def _case(case):
    return case if isinstance(case, FoamCase) else FoamCase(case)


def mechanisms(case):
    """The mechanisms a case can run with: one per 0.<mechanism>, smallest first."""
    path = _case(case).path

    def size(mechanism):
        reactions = path / "constant" / mechanism / "reactions"
        return reactions.stat().st_size if reactions.is_file() else float("inf")

    names = [p.name[2:] for p in path.glob("0.*") if p.is_dir()]
    return sorted(names, key=size)


def parse_variant(variant):
    """chemistryType/method for a variant name."""
    if variant == "standard":
        return "standard"
    if variant == "kodes":
        return "kodesChemistryModel"
    raise ValueError(
        f"unknown variant {variant!r}; one of: {', '.join(VARIANTS)}"
    )


def current(case):
    """(mechanism, variant) the case is set up for now."""
    chemistry = _case(case).file("constant/chemistryProperties")
    kodes = chemistry["kodesSetting"]

    if chemistry["chemistryType"]["method"] == "standard":
        variant = "standard"
    else:
        variant = "kodes"

    return kodes["mechanism"], variant


def configure(case, mechanism, variant):
    """Set the case up for one mechanism and one variant."""
    case = _case(case)
    method = parse_variant(variant)

    available = mechanisms(case)
    if mechanism not in available:
        raise ValueError(
            f"{case.name} has no 0.{mechanism}; one of: {', '.join(available)}"
        )
    if not (case.path / "constant" / mechanism / "reactions").is_file():
        raise FileNotFoundError(
            f"{case.path / 'constant' / mechanism} does not reach a mechanism"
        )

    zero = case.path / "0"
    shutil.rmtree(zero, ignore_errors=True)
    shutil.copytree(case.path / f"0.{mechanism}", zero, symlinks=True)

    thermo = case.file("constant/thermophysicalProperties")
    with thermo:
        _set(thermo, "foamChemistryFile", f'"<constant>/{mechanism}/reactions"')
        _set(
            thermo,
            "foamChemistryThermoFile",
            f'"<constant>/{mechanism}/thermo.compressibleGas"',
        )

    chemistry = case.file("constant/chemistryProperties")
    with chemistry:
        _set(chemistry["chemistryType"], "method", method)
        # The library a kodes run opens follows the mechanism even for a
        # standard run, so the next switch to kodes cannot pick up a stale one
        kodes = chemistry["kodesSetting"]
        _set(kodes, "mechanism", mechanism)

    _set(
        case.control_dict, "libs", [] if method == "standard" else [KODES_LIBRARY]
    )

    return case


def _set(entries, key, value):
    # Only what changes is written: foamlib rewrites an entry in its own
    # layout, so assigning the value already there would still touch the file
    if entries.get(key) != value:
        entries[key] = value


def clone(case, dst):
    """
    Clone a case with foamlib, mechanisms included.

    foamlib copies the constant/<mechanism> symlinks as they are, relative to
    the tutorial, so a clone anywhere else would reach nothing; they are
    pointed at the mechanisms themselves instead.
    """
    case = _case(case)
    copy = case.clone(dst)

    for link in (copy.path / "constant").iterdir():
        if link.is_symlink():
            target = (case.path / "constant" / link.name).resolve()
            link.unlink()
            link.symlink_to(target, target_is_directory=target.is_dir())

    return copy


def solver_log(case, mechanism, variant):
    """Where run() writes the solver's log."""
    case = _case(case)
    return case.path / f"log.{case.application}.{mechanism}.{variant}"


def reset(case):
    """
    Remove what a previous run left: time directories, 0, processor*, the
    mesh and the logs of the pre-processing steps. The solver logs, one per
    mechanism and variant, stay, so the timings of every run survive.
    """
    case = _case(case)
    keep = f"log.{case.application}."

    for path in case.path.iterdir():
        name = path.name
        try:
            remove = path.is_dir() and float(name) >= 0
        except ValueError:
            remove = (
                (path.is_dir() and name.startswith("processor"))
                or name in ("postProcessing", "dynamicCode")
                or (name.startswith("log.") and not name.startswith(keep))
            )
        if remove:
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    shutil.rmtree(case.path / "constant" / "polyMesh", ignore_errors=True)


def run(case, mechanism=None, variant=None, *, nprocs=1, check=True):
    """
    Set the case up and run it: blockMesh, setFields if the case has a
    setFieldsDict, then the solver, in parallel on nprocs ranks if nprocs > 1. A
    mechanism or variant left out is the one the case is set up for now.

    Returns the path of the solver's log.
    """
    case = _case(case)
    now = current(case)
    mechanism = mechanism or now[0]
    variant = variant or now[1]

    reset(case)
    configure(case, mechanism, variant)

    case.block_mesh(check=check)
    if (case.path / "system" / "setFieldsDict").is_file():
        case.run(["setFields"], parallel=False, check=check)

    log = solver_log(case, mechanism, variant)
    log.unlink(missing_ok=True)

    if nprocs > 1:
        case.decompose_par_dict["numberOfSubdomains"] = nprocs
        case.decompose_par(check=check)
        case.run(
            [case.application], parallel=True, cpus=nprocs, check=check, log=log.name
        )
    else:
        case.run([case.application], parallel=False, check=check, log=log.name)

    return log


_TIME = re.compile(r"^Time = (\S+)$")
_EXECUTION = re.compile(r"^ExecutionTime = (\S+) s\s+ClockTime = (\S+) s")
_FATAL = re.compile(r"FOAM FATAL (IO )?ERROR|CUDA error")
_KODES = re.compile(r"^\s+([A-Z][\w ]*?)\s*[=:]\s+(.+?)\s*$")


def read_log(log):
    """
    What a solver log says about the run, as a dict:

        completed       the solver reached End
        steps           time steps taken
        time            simulated time reached [s]
        executionTime   CPU time at the last step [s]
        clockTime       wall-clock time at the last step [s]
        stepTime        wall-clock per step, the first step left out [s]:
                        start-up -- reading the mechanism, opening the GPU --
                        lands in it and says nothing about the integrator
        error           the first fatal error, if the run stopped on one
        kodes           the kodesChemistryModel banner, if it printed one
    """
    times, clocks = [], []
    error = None
    completed = False
    kodes = {}
    in_banner = False

    lines = Path(log).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if match := _TIME.match(line):
            times.append(float(match.group(1)))
        elif match := _EXECUTION.match(line):
            clocks.append((float(match.group(1)), float(match.group(2))))
        elif line.strip() == "End":
            completed = True
        elif "kodesChemistryModel: CUSTOM MODEL LOADED" in line:
            in_banner = True
        elif in_banner and line.startswith("====="):
            in_banner = False
        elif in_banner and (match := _KODES.match(line)):
            kodes[match.group(1)] = match.group(2)
        elif error is None and _FATAL.search(line):
            message = [l.strip() for l in lines[i:i + 8] if l.strip()]
            error = " ".join(message[:4])

    steps = len(clocks)
    clock = clocks[-1][1] if clocks else None
    step = (
        (clocks[-1][1] - clocks[0][1]) / (steps - 1) if steps > 1 else None
    )

    return {
        "completed": completed,
        "steps": steps,
        "time": times[-1] if times else None,
        "executionTime": clocks[-1][0] if clocks else None,
        "clockTime": clock,
        "stepTime": step,
        "error": error,
        "kodes": kodes,
    }


def compare(case, reference, fields=("T",)):
    """
    How far a run's fields are from a reference run's at the latest time
    both wrote: for each field, the largest absolute difference and the
    L2 norm of the difference relative to that of the reference.
    """
    case, reference = _case(case), _case(reference)
    common = sorted({t.time for t in case} & {t.time for t in reference})
    if not common or common[-1] == 0:
        raise ValueError(f"{case.name} and {reference.name} share no written time")
    time = common[-1]

    result = {"time": time}
    for field in fields:
        x = np.asarray(case[time][field].internal_field, dtype=float)
        y = np.asarray(reference[time][field].internal_field, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        diff = x - y
        result[f"{field}.maxAbs"] = float(np.max(np.abs(diff)))
        result[f"{field}.relL2"] = float(
            np.linalg.norm(diff) / max(np.linalg.norm(y), np.finfo(float).tiny)
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("list", help="mechanisms and variants a case runs with")
    p.add_argument("case")

    p = commands.add_parser("setup", help="set a case up without running it")
    p.add_argument("case")
    p.add_argument("mechanism")
    p.add_argument("variant", choices=VARIANTS)

    p = commands.add_parser("run", help="set a case up and run it")
    p.add_argument("case")
    p.add_argument("mechanism", nargs="?", help="default: the current one")
    p.add_argument("variant", nargs="?", choices=VARIANTS,
                   help="default: the current one")
    p.add_argument("--np", type=int, default=1, help="MPI ranks (default 1)")

    p = commands.add_parser("log", help="summarise solver logs")
    p.add_argument("logs", nargs="+")

    args = parser.parse_args()

    try:
        if args.command == "list":
            mechanism, variant = current(args.case)
            print(f"mechanisms: {' '.join(mechanisms(args.case))}")
            print(f"variants  : {' '.join(VARIANTS)}")
            print(f"current   : {mechanism} {variant}")

        elif args.command == "setup":
            configure(args.case, args.mechanism, args.variant)
            print(f"{FoamCase(args.case).name}: set up for {args.mechanism} {args.variant}")

        elif args.command == "run":
            log = run(args.case, args.mechanism, args.variant, nprocs=args.np)
            summary = read_log(log)
            print(f"{log.name}: {summary['steps']} steps, "
                  f"{summary['clockTime']} s wall clock, "
                  f"{summary['stepTime']} s per step")

        elif args.command == "log":
            for log in args.logs:
                summary = read_log(log)
                kodes = summary.pop("kodes")
                print(log)
                for key, value in summary.items():
                    print(f"    {key:14s}{value}")
                for key, value in kodes.items():
                    print(f"    {key:14s}{value}")

    except (ValueError, FileNotFoundError, CalledProcessError) as e:
        print(f"{parser.prog}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
