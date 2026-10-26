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

Class
    Foam::kodesSpecieMap

Description
    Translation between the specie ordering used by a pyJac-generated
    mechanism (as integrated by KODES) and the ordering of the OpenFOAM
    species table.

    The two orderings are unrelated: pyJac numbers the species in the order
    of the Chemkin input with the eliminated ("last") specie moved to the end,
    while the OpenFOAM ordering is whatever order the species are listed in
    the reactions file.  For GRI-Mech 3.0 as written by chemkinToFoam the two
    differ for the first twenty species, so the mapping cannot be derived
    arithmetically and has to be built from the specie names.

    The pyJac ordering is the one the KODES library reports for its
    mechanism (kodes::Library::specie()). KODES reads it out of the
    "Species Indexes" comment block pyJac writes into mechanism.cuh when the
    library is built, so it is the ordering of the very code that integrates
    the cells and cannot go out of step with it.

    The OpenFOAM ordering is taken from the speciesTable itself
    (thermo.composition().species()): it is a hashedWordList, so it already
    provides the name -> index lookup, is authoritative for the Y_/RR_ ordering
    and needs no separate input file that could go out of sync with the case.

    KODES state vector layout (systemSize == number of species):

    \verbatim
        y[0]                 temperature
        y[1 .. nSpecie()-1]  mass fractions of pyJac species 0 .. nSpecie()-2
        (parameter)          pressure
    \endverbatim

    The last pyJac specie does not appear in the state vector; it is recovered
    from sum(Y) = 1.  Its OpenFOAM index is inertIndex().

Note
    All lookups are resolved once, in the constructor.  A specie present in one
    ordering and missing from the other is a fatal error.

SourceFiles
    kodesSpecieMap.C

\*---------------------------------------------------------------------------*/

#ifndef kodesSpecieMap_H
#define kodesSpecieMap_H

#include "labelList.H"
#include "wordList.H"
#include "speciesTable.H"
#include "className.H"
#include "error.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declarations
class Ostream;

/*---------------------------------------------------------------------------*\
                       Class kodesSpecieMap Declaration
\*---------------------------------------------------------------------------*/

class kodesSpecieMap
{
    // Private Data

        //- The mechanism the pyJac ordering belongs to, for messages
        word mechanism_;

        //- Specie names in pyJac order
        wordList kodesNames_;

        //- kodes state index -> OpenFOAM specie index.
        //  Size nSpecie(); entry [0] is -1 (temperature, not a specie)
        labelList kodesToFoam_;

        //- OpenFOAM specie index -> kodes state index.
        //  Size nSpecie(); the eliminated specie maps to -1
        labelList foamToKodes_;

        //- OpenFOAM index of the specie eliminated by pyJac
        label inertIndex_;


    // Private Member Functions

        //- Resolve both mappings, checking that the two orderings describe
        //- the same set of species
        void buildMaps(const speciesTable& species);


public:

    //- Declare type-name and the "kodesSpecieMap" debug switch, which makes
    //- the resolved mapping be written out at construction
    ClassName("kodesSpecieMap");


    // Constructors

        //- Construct from the specie names of a pyJac mechanism, in its
        //- order, and an OpenFOAM species table
        kodesSpecieMap
        (
            const word& mechanism,
            const wordList& kodesNames,
            const speciesTable& species
        );


    // Member Functions

        //- Number of species; also the size of the KODES state vector
        inline label nSpecie() const noexcept
        {
            return kodesNames_.size();
        }

        //- OpenFOAM index of the specie eliminated from the state vector
        inline label inertIndex() const noexcept
        {
            return inertIndex_;
        }

        //- Specie names in pyJac order
        inline const wordList& kodesNames() const noexcept
        {
            return kodesNames_;
        }

        //- The mechanism the pyJac ordering belongs to
        inline const word& mechanism() const noexcept
        {
            return mechanism_;
        }

        //- KODES state index -> OpenFOAM specie index.
        //  Not defined for kodesi = 0 (temperature)
        inline label kodesToFoam(const label kodesi) const
        {
            #ifdef FULLDEBUG
            if (kodesi < 1 || kodesi >= kodesToFoam_.size())
            {
                FatalErrorInFunction
                    << "kodes state index " << kodesi << " out of range [1,"
                    << kodesToFoam_.size() << ")" << nl
                    << abort(FatalError);
            }
            #endif

            return kodesToFoam_[kodesi];
        }

        //- OpenFOAM specie index -> KODES state index.
        //  Returns -1 for the eliminated specie
        inline label foamToKodes(const label foami) const
        {
            #ifdef FULLDEBUG
            if (foami < 0 || foami >= foamToKodes_.size())
            {
                FatalErrorInFunction
                    << "OpenFOAM specie index " << foami << " out of range [0,"
                    << foamToKodes_.size() << ")" << nl
                    << abort(FatalError);
            }
            #endif

            return foamToKodes_[foami];
        }

        //- Write the resolved mapping
        void write(Ostream& os) const;
};


// * * * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * //

Ostream& operator<<(Ostream& os, const kodesSpecieMap& map);


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
