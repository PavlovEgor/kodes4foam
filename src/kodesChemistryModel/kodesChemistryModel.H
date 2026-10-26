/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020-2023 OpenCFD Ltd.
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
    Foam::kodesChemistryModel

Description
    Chemistry model that hands the cell-wise ODE integration to KODES on the
    GPU, one cell per thread, in batches.

    KODES is not compiled into this library. Each pyJac mechanism is a
    prebuilt libkodes_<mechanism>.so - the whole of KODES and the mechanism
    behind one device link - which this model opens with dlopen() when the
    case starts, by the name given in kodesSetting. Nothing here includes a
    CUDA header or needs a CUDA compiler, and switching a case to another
    mechanism is an edit to chemistryProperties, not a rebuild.

Usage
    In constant/chemistryProperties:
    \verbatim
    kodesSetting
    {
        // Opens libkodes_<mechanism>.so, from LD_LIBRARY_PATH - which has
        // the OpenFOAM library directories on it, Allwmake installs them
        // next to this model
        mechanism       grimech;

        // Which method integrates a cell: seulex | euler
        //   seulex - semi-implicit extrapolation, controls its own step
        //   euler  - explicit, for checking the machinery only
        integrationMethod seulex;

        // How much of the device one run may take: best | half
        deviceShare     best;

        // Size the run by hand instead - both entries are needed
        // concurrentSystems 8192;   // cells integrated at the same time
        // batchSize      1000000;   // cells per host->device transfer

        // threadsPerBlock 256;

        // Group each batch before integrating it, so the lanes of a warp
        // get cells that need a similar number of steps
        balanceBatches  true;

        // What to group by: temperature | rhsNorm | stiffness | none
        //   temperature - component 0 of the state, free to read
        //   rhsNorm     - how fast the cell is moving, one extra
        //                 derivatives() per cell per step
        //   stiffness   - temperature first, rhsNorm inside each band
        // 'none' is the same as balanceBatches false.
        balancerKey     temperature;
    }
    \endverbatim
    Tolerances and the step limit are read from odeCoeffs, the initial trial
    step from initialChemicalTimeStep, and cells at or below Treact are left
    alone as in the standard model.

    An unknown method or balancer name fails at construction with the list of
    the ones the library was built with. The species are mapped by name onto
    the ones the library reports, so a library built for a different
    mechanism fails there too, naming the specie it could not find.

SourceFiles
    kodesChemistryModel.C

\*---------------------------------------------------------------------------*/

#ifndef kodesChemistryModel_H
#define kodesChemistryModel_H

#include "StandardChemistryModel.H"

#include "kodesSpecieMap.H"

// The one KODES header: plain C++, no CUDA in it
#include "library_api.h"

#include <memory>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class kodesChemistryModel Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo, class ThermoType>
class kodesChemistryModel
:
    public StandardChemistryModel<ReactionThermo, ThermoType>
{
    dictionary settingsDict_;

    //- The mechanism's library, owned by the library itself. Opened once and
    //  never closed: unloading a CUDA module while the process still runs
    //  gains nothing and can only go wrong.
    const kodes::Library& library_;

    //- Translation between the KODES/pyJac and OpenFOAM specie orderings
    kodesSpecieMap specieMap_;

    //- What goes to the device and comes back integrated, one list per
    //  component of the KODES state: the temperature, then the mass
    //  fractions in KODES order. The fields still hold the state the step
    //  started from, which is what the reaction rates are taken against.
    List<List<double>> state_;

    //- The pressure, the one parameter of a pyJac system. A copy, since KODES
    //  copies the parameters back along with the state.
    List<double> pressure_;

    //- What solve() is handed: one pointer per state component and parameter
    List<double*> statePointers_;
    List<double*> parameterPointers_;

    //- Plans the run when it is made, then owns the method, the balancer,
    //  the mechanism and their device buffers until it is deleted
    std::unique_ptr<kodes::Solver> solver_;

    //- Open libkodes_<mechanism>.so, check it speaks the interface this
    //  model was built against, and hand back its mechanism
    static const kodes::Library& openLibrary(const word& mechanism);

    //- The pyJac specie names, in KODES order
    static wordList kodesSpecies(const kodes::Library& library);

    //- Which method integrates a cell. Unset means seulex, which is what the
    //  model did before there was anything else to pick.
    static word methodName(const dictionary& dict);

    //- What the batch is grouped by, or "none". 'balanceBatches false' is the
    //  same thing said the other way round, and still wins if it is set.
    static word balancerName(const dictionary& dict);

    //- Make the solver: plan the run against the free memory of the device,
    //  which has to happen before anything else is allocated there, and
    //  build everything at the sizes the plan settled on
    static kodes::Solver* newSolver
    (
        const kodes::Library& library,
        const dictionary& dict,
        const dictionary& odeCoeffs,
        const label nCells,
        const scalar Treact
    );

public:

    //- Runtime type information
    TypeName("kodesChemistryModel");


    // Constructors

        //- Construct from thermo
        kodesChemistryModel(ReactionThermo& thermo);


    //- Destructor
    virtual ~kodesChemistryModel();

    // Member Functions

    virtual scalar solve(const scalar deltaT) override;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "kodesChemistryModel.C"
#endif

#endif

// ************************************************************************* //


