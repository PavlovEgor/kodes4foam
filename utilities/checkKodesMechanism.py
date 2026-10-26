#!/usr/bin/env python3
"""
Check that a case's OpenFOAM species table and its pyJac mechanism header
describe the same set of species, i.e. that Foam::kodesSpecieMap will build
without a fatal error.

kodesSpecieMap resolves the pyJac <-> OpenFOAM specie mapping by name at case
start-up and calls FatalError if the two orderings are not the same set of the
same size.  That check needs a running case and therefore libkodes_<mechanism>.so,
built with nvcc, whose specie names KODES read out of this same mechanism.cuh;
this script performs the identical check offline, from the two files alone, so
a case can be validated on a machine without a GPU.

It reproduces, in order, what kodesSpecieMap does:

  1. read the specie names from the "Species Indexes" block of mechanism.cuh
     and cross-check them against the NSP define,
  2. read the species list from the OpenFOAM reactions file,
  3. require both to have the same size,
  4. require every pyJac name to be in the OpenFOAM table (and vice versa),
  5. report the specie pyJac eliminated from the state vector -- the last one
     in its ordering, recovered at run time from sum(Y) = 1.

Optionally (--thermophysical-properties) it also checks the case's inertSpecie
against that eliminated specie.  They are independent settings -- OpenFOAM's
inertSpecie only balances the Y equations -- but both are best pointed at the
same, most abundant, specie so neither subtraction amplifies round-off.

Usage:
    checkKodesMechanism.py <mechanism.cuh> <reactions> \\
        [--thermophysical-properties <thermophysicalProperties>]

Exits non-zero if the case would fail at start-up.
"""

import argparse
import re
import sys
from pathlib import Path


def read_pyjac_species(mechanism_file):
    """Specie names in pyJac order, from the generated mechanism header."""
    text = Path(mechanism_file).read_text()

    block = re.search(r'Species Indexes(.*?)\*/', text, re.S)
    if not block:
        sys.exit(
            f"error: no 'Species Indexes' block in {mechanism_file} -- "
            f"is this a pyJac-generated mechanism header?"
        )

    names = []
    for line in block.group(1).splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        index, name = int(fields[0]), fields[1]
        if index != len(names):
            sys.exit(
                f"error: non-contiguous specie indices in {mechanism_file}: "
                f"expected index {len(names)}, found {index} ({name})"
            )
        names.append(name)

    if not names:
        sys.exit(f"error: empty 'Species Indexes' block in {mechanism_file}")

    nsp = re.search(r'#define\s+NSP\s+(\d+)', text)
    if nsp and int(nsp.group(1)) != len(names):
        sys.exit(
            f"error: inconsistent {mechanism_file}: NSP is {nsp.group(1)} but "
            f"the 'Species Indexes' block lists {len(names)} species"
        )

    return names


def read_foam_species(reactions_file):
    """Specie names in OpenFOAM order, from a foamChemistryFile."""
    text = Path(reactions_file).read_text()

    match = re.search(r'\bspecies\s+(\d+)?\s*\((.*?)\)\s*;', text, re.S)
    if not match:
        sys.exit(f"error: no 'species (...)' list in {reactions_file}")

    names = match.group(2).split()

    if match.group(1) and int(match.group(1)) != len(names):
        sys.exit(
            f"error: inconsistent {reactions_file}: the species list is "
            f"declared as {match.group(1)} entries but holds {len(names)}"
        )

    return names


def read_inert_specie(thermophysical_properties):
    text = Path(thermophysical_properties).read_text()
    match = re.search(r'^\s*inertSpecie\s+(\S+?)\s*;', text, re.M)
    if not match:
        sys.exit(f"error: no 'inertSpecie' entry in {thermophysical_properties}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", help="pyJac-generated mechanism.cuh")
    parser.add_argument("reactions", help="OpenFOAM foamChemistryFile (reactions)")
    parser.add_argument(
        "--thermophysical-properties",
        help="Also check this dictionary's inertSpecie against the specie "
             "pyJac eliminated from the state vector",
    )
    args = parser.parse_args()

    pyjac = read_pyjac_species(args.mechanism)
    foam = read_foam_species(args.reactions)

    print(f"pyJac mechanism : {args.mechanism}")
    print(f"    {len(pyjac)} species, state vector is y[0] = T, y[1..{len(pyjac) - 1}] = Y")
    print(f"OpenFOAM species: {args.reactions}")
    print(f"    {len(foam)} species")

    failed = False

    if len(pyjac) != len(foam):
        print(
            f"FAIL: specie count mismatch -- {len(pyjac)} in the mechanism, "
            f"{len(foam)} in the species table",
            file=sys.stderr,
        )
        failed = True

    missing = [name for name in pyjac if name not in foam]
    if missing:
        print(
            f"FAIL: {len(missing)} pyJac specie(s) missing from the OpenFOAM "
            f"species table: {' '.join(missing)}",
            file=sys.stderr,
        )
        failed = True

    extra = [name for name in foam if name not in pyjac]
    if extra:
        print(
            f"FAIL: {len(extra)} OpenFOAM specie(s) missing from the pyJac "
            f"mechanism: {' '.join(extra)}",
            file=sys.stderr,
        )
        failed = True

    duplicates = sorted({name for name in foam if foam.count(name) > 1})
    if duplicates:
        print(
            f"FAIL: duplicate specie name(s) in the species table: "
            f"{' '.join(duplicates)}",
            file=sys.stderr,
        )
        failed = True

    if failed:
        return 1

    eliminated = pyjac[-1]
    print(f"OK  : both orderings hold the same {len(pyjac)} species")
    print(f"    eliminated from the state vector (from sum(Y) = 1): {eliminated}")

    if args.thermophysical_properties:
        inert = read_inert_specie(args.thermophysical_properties)
        if inert == eliminated:
            print(f"OK  : inertSpecie {inert} is the eliminated specie")
        else:
            print(
                f"WARN: inertSpecie is {inert} but pyJac eliminates "
                f"{eliminated}; both are valid, but pointing them at the same "
                f"(most abundant) specie keeps round-off down"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
