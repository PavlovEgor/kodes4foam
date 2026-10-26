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

\*---------------------------------------------------------------------------*/

#include "kodesChemistryModel.H"
#include "reactingMixture.H"
#include "UniformField.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "OSspecific.H"

#include <algorithm>
#include <cstdint>
#include <limits>

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
const kodes::Library&
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::openLibrary
(
    const word& mechanism
)
{
    const fileName libName("libkodes_" + mechanism + ".so");

    // Opened again by every model of the run - one per region - which dlopen
    // answers with the same handle, so they share the one Library in it
    std::string errorMsg;
    void* handle = Foam::dlOpen(libName, errorMsg);

    if (!handle)
    {
        FatalErrorInFunction
            << "Cannot open " << libName << " for 'mechanism " << mechanism
            << "' in kodesSetting" << nl
            << "    " << errorMsg.c_str() << nl
            << "    Build it from the mechanism's pyJac output with"
            << " kodes4foam's Allwmake, which installs it next to"
            << " libkodesChemistryModel." << nl
            << exit(FatalError);
    }

    // The version is asked for first and through a plain function, because
    // it is the one call that works whatever shape the rest has
    const auto apiVersion = reinterpret_cast<kodes::LibraryApiVersionFunction>
    (
        Foam::dlSymFound(handle, kodes::kLibraryApiVersionSymbol)
    );

    const auto library = reinterpret_cast<kodes::LibraryFunction>
    (
        Foam::dlSymFound(handle, kodes::kLibrarySymbol)
    );

    if (!apiVersion || !library)
    {
        FatalErrorInFunction
            << libName << " is not a KODES library: it has no "
            << kodes::kLibraryApiVersionSymbol << " or no "
            << kodes::kLibrarySymbol << nl
            << exit(FatalError);
    }

    if (apiVersion() != kodes::kLibraryApiVersion)
    {
        FatalErrorInFunction
            << libName << " has KODES library interface version "
            << apiVersion() << ", this model was built against version "
            << kodes::kLibraryApiVersion << nl
            << "    Rebuild the library and the model from the same KODES."
            << nl << exit(FatalError);
    }

    return *library();
}


template<class ReactionThermo, class ThermoType>
Foam::wordList
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::kodesSpecies
(
    const kodes::Library& library
)
{
    wordList names(library.nSpecie());

    forAll(names, i)
    {
        names[i] = word(library.specie(i), false);
    }

    return names;
}


// Both names are handed straight to KODES, which fails with the list of the
// ones it was built with if it does not know one. Nothing here has to be kept
// in step with the library's tables any more - it used to be an enum, four
// switches and two member templates, all of which had to gain a case whenever
// KODES gained a balancer.
template<class ReactionThermo, class ThermoType>
Foam::word
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::methodName
(
    const dictionary& dict
)
{
    return dict.getOrDefault<word>("integrationMethod", "seulex");
}


template<class ReactionThermo, class ThermoType>
Foam::word
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::balancerName
(
    const dictionary& dict
)
{
    if (!dict.getOrDefault<bool>("balanceBatches", true))
    {
        return "none";
    }

    return dict.getOrDefault<word>("balancerKey", "temperature");
}


template<class ReactionThermo, class ThermoType>
kodes::Solver*
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::newSolver
(
    const kodes::Library& library,
    const dictionary& dict,
    const dictionary& odeCoeffs,
    const label nCells,
    const scalar Treact
)
{
    // KODES counts in int
    if (nCells > std::numeric_limits<int>::max())
    {
        FatalErrorInFunction
            << nCells << " cells are more than KODES can count in one"
            << " ensemble; decompose the case with 'gatherCells false'" << nl
            << exit(FatalError);
    }

    // The settings below point into these, so they have to outlive the
    // newSolver() call - which is all the solver needs them for
    const word method(methodName(dict));
    const word balancer(balancerName(dict));
    const word deviceShare(dict.getOrDefault<word>("deviceShare", "best"));

    kodes::SolverSettings settings;

    settings.method = method.c_str();
    settings.balancer = balancer.c_str();
    settings.deviceShare = deviceShare.c_str();

    settings.threadsPerBlock =
        dict.getOrDefault<label>("threadsPerBlock", settings.threadsPerBlock);

    // Sizing the run by hand takes both numbers: how many systems run at the
    // same time and how many of them travel per transfer
    const bool hasConcurrent = dict.found("concurrentSystems");
    const bool hasBatch = dict.found("batchSize");

    if (hasConcurrent && hasBatch)
    {
        settings.concurrentSystems = dict.get<label>("concurrentSystems");
        settings.batchSize = dict.get<label>("batchSize");
    }
    else if (hasConcurrent != hasBatch)
    {
        WarningInFunction
            << "Only one of 'concurrentSystems' and 'batchSize' was given in "
            << dict.name() << nl
            << "    Both are needed to size the run by hand, so the plan is"
            << " left to 'deviceShare' and the entry is ignored." << nl
            << endl;
    }

    // The call stack of a device thread, in bytes, or 0 to leave the driver's
    // own value alone. 0 is the default and is nearly always right: the driver
    // sizes the stack from the frames ptxas recorded, and overriding it
    // reserves the new size for every thread the device could hold resident,
    // which comes straight off the batch. See KODES_DEVICE_STACK_BYTES.
    const label deviceStackSize =
        dict.getOrDefault<label>("deviceStackSize", 0);

    if (deviceStackSize < 0)
    {
        FatalErrorInFunction
            << "'deviceStackSize' is " << deviceStackSize << " in "
            << dict.name() << ", it must be a number of bytes, or 0 to leave"
            << " the driver's own value alone" << nl
            << exit(FatalError);
    }

    settings.deviceStackBytes = size_t(deviceStackSize);

    // Cells at or below Treact are left alone by the solve kernel, the same
    // ones solve() then zeroes the reaction rates of
    settings.absTol = odeCoeffs.getOrDefault<scalar>("absTol", 1e-10);
    settings.relTol = odeCoeffs.getOrDefault<scalar>("relTol", 1e-1);
    settings.maxSteps = odeCoeffs.getOrDefault<label>("maxSteps", 10000);
    settings.Treact = Treact;

    return library.newSolver(int(nCells), settings);
}


template<class ReactionThermo, class ThermoType>
Foam::label
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::nComponents() const
{
    return library_.systemSize() + library_.parameterSize();
}


template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::pointInto
(
    List<double>& ensemble,
    const label nSystems
)
{
    // The layout KODES's HostResources takes: one array per component, as
    // long as the ensemble, here end to end in one buffer
    forAll(statePointers_, i)
    {
        statePointers_[i] = ensemble.data() + i*nSystems;
    }

    forAll(parameterPointers_, i)
    {
        parameterPointers_[i] =
            ensemble.data() + (statePointers_.size() + i)*nSystems;
    }
}


template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::packLocal()
{
    const scalarField& T = this->thermo().T();
    const scalarField& p = this->thermo().p();
    const label nCells = T.size();
    const label nState = library_.systemSize();

    // Component 0 is the temperature, which is also the key the batch is
    // sorted by, the rest are the mass fractions in KODES ordering. The
    // pressure, the one parameter, follows the state.
    forAll(T, celli)
    {
        local_[celli] = T[celli];
        local_[nState*nCells + celli] = p[celli];
    }

    for (label i = 1; i < nState; ++i)
    {
        const scalarField& Yi = this->Y_[specieMap_.kodesToFoam(i)];
        double* yi = local_.data() + i*nCells;

        forAll(Yi, celli)
        {
            yi[celli] = Yi[celli];
        }
    }
}


template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::unpackLocal
(
    const scalar deltaT
)
{
    tmp<volScalarField> trho(this->thermo().rho());
    const scalarField& rho = trho();
    const scalarField& T = this->thermo().T();
    const label nCells = T.size();
    const label nState = library_.systemSize();

    const label inertIndex = specieMap_.inertIndex();

    forAll(rho, celli)
    {
        if (T[celli] > this->Treact_)
        {
            scalar Ysum = 0;

            for (label i = 1; i < nState; ++i)
            {
                const label si = specieMap_.kodesToFoam(i);
                const scalar Yi = local_[i*nCells + celli];

                Ysum += Yi;
                this->RR_[si][celli] = rho[celli]*(Yi - this->Y_[si][celli])/deltaT;
            }

            this->RR_[inertIndex][celli] =
                rho[celli]*((1 - Ysum) - this->Y_[inertIndex][celli])/deltaT;
        }
        else
        {
            for (label i = 0; i < this->nSpecie_; ++i)
            {
                this->RR_[i][celli] = 0;
            }
        }
    }
}


// The master holds each rank's cells as that rank packed them - all of its
// first component, then all of its second - one rank after the other. KODES
// wants every cell's first component together, then every cell's second; the
// two orders differ only in which index runs fastest across the ranks.
template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::
receivedToEnsemble()
{
    const label nComp = nComponents();
    const label nTotal = gathered_.totalSize()/nComp;

    for (const label proci : gathered_.allProcs())
    {
        const label start = gathered_.localStart(proci);
        const label nCells = gathered_.localSize(proci)/nComp;
        const label firstCell = start/nComp;

        for (label i = 0; i < nComp; ++i)
        {
            const double* from = received_.cdata() + start + i*nCells;
            double* to = ensemble_.data() + i*nTotal + firstCell;

            std::copy(from, from + nCells, to);
        }
    }
}


template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::
ensembleToReceived()
{
    const label nComp = nComponents();
    const label nTotal = gathered_.totalSize()/nComp;

    for (const label proci : gathered_.allProcs())
    {
        const label start = gathered_.localStart(proci);
        const label nCells = gathered_.localSize(proci)/nComp;
        const label firstCell = start/nComp;

        for (label i = 0; i < nComp; ++i)
        {
            const double* from = ensemble_.cdata() + i*nTotal + firstCell;
            double* to = received_.data() + start + i*nCells;

            std::copy(from, from + nCells, to);
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::kodesChemistryModel
(
    ReactionThermo& thermo
)
:
    StandardChemistryModel<ReactionThermo, ThermoType>(thermo),
    settingsDict_
    (
        this->subDict("kodesSetting")
    ),
    library_
    (
        openLibrary(settingsDict_.get<word>("mechanism"))
    ),
    specieMap_
    (
        word(library_.mechanism(), false),
        kodesSpecies(library_),
        thermo.composition().species()
    ),
    gather_
    (
        UPstream::parRun()
     && settingsDict_.getOrDefault<bool>("gatherCells", true)
    ),
    local_(),
    gathered_(),
    received_(),
    ensemble_(),
    statePointers_(library_.systemSize()),
    parameterPointers_(library_.parameterSize()),
    solver_()
{
    if (settingsDict_.found("kodesMechanism"))
    {
        WarningInFunction
            << "'kodesMechanism' in " << settingsDict_.name()
            << " is no longer read: the species come from libkodes_"
            << library_.mechanism() << ".so itself, chosen by 'mechanism'."
            << nl << endl;
    }

    const label nCells = this->thermo().T().size();
    const label nComp = nComponents();

    // Every buffer here is one List, indexed by label: the whole ensemble
    // times its components has to fit in one
    const int64_t nSystems =
        gather_ ? returnReduce(int64_t(nCells), sumOp<int64_t>()) : nCells;

    if (nSystems*nComp > int64_t(labelMax))
    {
        FatalErrorInFunction
            << nSystems << " cells of " << nComp << " components each are"
            << " more values than one list of this build can hold" << nl
            << "    Decompose the case"
            << (gather_ ? " with 'gatherCells false'" : "")
            << ", or build OpenFOAM with 64-bit labels." << nl
            << exit(FatalError);
    }

    local_.resize(nCells*nComp);

    if (gather_)
    {
        // Every rank learns every rank's share; only the master needs them
        gathered_.reset(nCells*nComp);

        if (UPstream::master())
        {
            received_.resize(gathered_.totalSize());
            ensemble_.resize(gathered_.totalSize());

            pointInto(ensemble_, label(nSystems));
        }
    }
    else
    {
        pointInto(local_, nCells);
    }

    if (!gather_ || UPstream::master())
    {
        solver_.reset
        (
            newSolver
            (
                library_,
                settingsDict_,
                this->subDict("odeCoeffs"),
                label(nSystems),
                this->Treact_
            )
        );

        solver_->setDeltaT
        (
            this->template getOrDefault<scalar>("initialChemicalTimeStep", 1e-7)
        );
    }

    // Info writes on the master only, which in a gathered run is the one rank
    // with a solver to ask
    if (solver_)
    {
        Info<< "\n========================================" << nl
            << "  kodesChemistryModel: CUSTOM MODEL LOADED!" << nl
            << "  Using custom chemistry model by user" << nl
            << "  Number of species = " << this->nSpecie_ << nl
            << "  Number of reactions = " << this->nReaction_ << nl
            << "  Thermophysical type: " << ThermoType::typeName() << nl
            << "  Time: " << this->mesh().time().timeName() << nl
            << "  Mechanism: libkodes_" << library_.mechanism() << ".so, KODES "
            << library_.version() << nl
            << "  Eliminated specie = "
            << this->thermo().composition().species()[specieMap_.inertIndex()]
            << " (FOAM index " << specieMap_.inertIndex() << ')' << nl
            << "  Integration method = " << solver_->method() << nl
            << "  Ensemble = " << label(nSystems) << " cells"
            << (
                   gather_
                 ? string(", gathered from all " + Foam::name(UPstream::nProcs())
                        + " ranks")
                 : string(UPstream::parRun() ? " on each rank" : "")
               ).c_str()
            << nl
            << "  Concurrent systems = " << solver_->concurrentSystems()
            << " (" << solver_->blocks() << " blocks x "
            << solver_->threads() << " threads)" << nl
            << "  Batch = " << solver_->batchSize() << " cells, "
            << solver_->numOfBatches() << " batch(es) per step" << nl
            << "  Batch balancing = "
            << (word(solver_->balancer()) == "none" ? "off" : solver_->balancer())
            << nl
            << "  Device stack = " << label(solver_->deviceStack())
            << " B per thread" << nl
            << "========================================\n" << endl;
    }

    if (kodesSpecieMap::debug)
    {
        specieMap_.write(Info);
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::~kodesChemistryModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
Foam::scalar Foam::kodesChemistryModel<ReactionThermo, ThermoType>::solve
(
    const scalar deltaT
)
{
    BasicChemistryModel<ReactionThermo>::correct();

    if (!this->chemistry_)
    {
        return GREAT;
    }

    packLocal();

    if (gather_)
    {
        // MPI_Gatherv: one message per rank, the master's own share copied
        gathered_.mpiGather(local_, received_);

        if (UPstream::master())
        {
            receivedToEnsemble();

            solver_->solve
            (
                deltaT, statePointers_.cdata(), parameterPointers_.cdata()
            );

            ensembleToReceived();
        }

        gathered_.scatter(received_, local_);
    }
    else
    {
        solver_->solve
        (
            deltaT, statePointers_.cdata(), parameterPointers_.cdata()
        );
    }

    unpackLocal(deltaT);

    return GREAT;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

