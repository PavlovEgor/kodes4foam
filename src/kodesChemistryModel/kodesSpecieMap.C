/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (c) 2026 Egor Pavlov
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kodesSpecieMap.H"
#include "Ostream.H"

// * * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(kodesSpecieMap, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::kodesSpecieMap::buildMaps(const speciesTable& species)
{
    const label nSpecie = kodesNames_.size();

    if (nSpecie != species.size())
    {
        FatalErrorInFunction
            << "Specie count mismatch between the pyJac mechanism and the"
            << " OpenFOAM species table" << nl
            << "    " << mechanism_ << " : " << nSpecie << nl
            << "    species table : " << species.size() << nl
            << exit(FatalError);
    }

    kodesToFoam_.resize(nSpecie, -1);
    foamToKodes_.resize(nSpecie, -1);

    // y[0] is the temperature, y[1..nSpecie-1] hold pyJac species 0..nSpecie-2
    for (label kodesi = 1; kodesi < nSpecie; ++kodesi)
    {
        const word& name = kodesNames_[kodesi - 1];
        const label foami = species.find(name);

        if (foami < 0)
        {
            FatalErrorInFunction
                << "Specie " << name << " (pyJac index " << kodesi - 1
                << ") is not in the OpenFOAM species table" << nl
                << "    mechanism : " << mechanism_ << nl
                << "    species   : " << species << nl
                << exit(FatalError);
        }

        kodesToFoam_[kodesi] = foami;
        foamToKodes_[foami] = kodesi;
    }

    // The specie pyJac eliminated from the state vector is its last one
    const word& inertName = kodesNames_[nSpecie - 1];

    inertIndex_ = species.find(inertName);

    if (inertIndex_ < 0)
    {
        FatalErrorInFunction
            << "Eliminated specie " << inertName
            << " (pyJac index " << nSpecie - 1
            << ") is not in the OpenFOAM species table" << nl
            << "    mechanism : " << mechanism_ << nl
            << "    species   : " << species << nl
            << exit(FatalError);
    }

    // Both orderings have the same size and every pyJac name was found, so an
    // unmapped OpenFOAM specie can only come from a duplicate name
    forAll(foamToKodes_, foami)
    {
        if (foamToKodes_[foami] < 0 && foami != inertIndex_)
        {
            FatalErrorInFunction
                << "Specie " << species[foami]
                << " is not mapped to any KODES state entry" << nl
                << "    duplicate specie names in " << mechanism_ << '?'
                << nl << exit(FatalError);
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::kodesSpecieMap::kodesSpecieMap
(
    const word& mechanism,
    const wordList& kodesNames,
    const speciesTable& species
)
:
    mechanism_(mechanism),
    kodesNames_(kodesNames),
    kodesToFoam_(),
    foamToKodes_(),
    inertIndex_(-1)
{
    buildMaps(species);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::kodesSpecieMap::write(Ostream& os) const
{
    os  << "kodes/FOAM specie map from " << mechanism_ << nl
        << "    " << nSpecie() << " species, state vector: y[0] = T, "
        << "parameter = p" << nl;

    for (label kodesi = 1; kodesi < kodesToFoam_.size(); ++kodesi)
    {
        os  << "    y[" << kodesi << "] = " << kodesNames_[kodesi - 1]
            << " -> Y[" << kodesToFoam_[kodesi] << ']' << nl;
    }

    os  << "    eliminated: " << kodesNames_[nSpecie() - 1]
        << " -> Y[" << inertIndex_ << ']' << endl;
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

Foam::Ostream& Foam::operator<<(Ostream& os, const kodesSpecieMap& map)
{
    map.write(os);
    return os;
}


// ************************************************************************* //
