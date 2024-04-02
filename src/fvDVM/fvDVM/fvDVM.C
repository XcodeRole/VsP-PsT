/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
     \\/     M anipulation  |
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

#include <mpi.h>
#include "fvDVM.H"
#include "constants.H"
#include "fvm.H"
#include "calculatedMaxwellFvPatchField.H"
#include "symmetryModFvPatchField.H"
#include "pressureInFvPatchField.H"
#include "pressureOutFvPatchField.H"
#include "scalarIOList.H"
#include "fieldMPIreducer.H"
#include "tickTock.H"
#define NUM_THREADS 2


using namespace Foam::constant;
using namespace Foam::constant::mathematical;

#if FOAM_MAJOR <= 3
    #define BOUNDARY_FIELD_REF boundaryField()
#else
    #define BOUNDARY_FIELD_REF boundaryFieldRef()
#endif


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(fvDVM, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::fvDVM::setDVgrid
(
    scalarField& weights,
    scalarField& Xis,
    scalar xiMin,
    scalar xiMax,
    label nXi
)
{
    // Read from file ./constant/Xis and ./constant/weights
    scalarIOList xiList
    (
        IOobject
        (
             "Xis",
             time_.caseConstant(),
             mesh_,
             IOobject::MUST_READ,
             IOobject::NO_WRITE
        )
    );

    scalarIOList weightList
    (
        IOobject
        (
             "weights",
             time_.caseConstant(),
             mesh_,
             IOobject::MUST_READ,
             IOobject::NO_WRITE
        )
    );

    // Check
    //if (
            //xiList.size() != nXi 
         //|| weightList.size() != nXi
       //)
    //{
        //WarningIn("setDVgrid")
            //<< "Num of discreteVelocity not consistent" << endl;
        //std::exit(-1);
    //}


    //if ( 
           //xiList[0]  != xiMin
        //|| xiList[nXi-1] != xiMax)
    //{
        //WarningIn("setDVgrid")
            //<< "xi not consistant" <<endl;
        //std::exit(-1);
    //}


    for (label i = 0; i < nXi ; i++)
    {
        weights[i] = weightList[i];
        Xis[i] = xiList[i];
    }
}

void Foam::fvDVM::initialiseDV()
{
    scalarField weights1D(nXiPerDim_);
    scalarField Xis(nXiPerDim_);
#ifdef VMESH
    xiMax_.value()=0.0;
//we have polyMesh and vMesh at the beginning
	string mesh_name = args_.path() / word("constant") / word("polyMesh");//polyMesh dir
	string vmesh_name = args_.path() / word("constant") / word("vMesh");//vMesh dir
	string temp_name = args_.path() / word("constant") / word("tMesh");//temp name

	//1.change folder name, polyMesh-->tMesh,vMesh-->polyMesh
	//as OF only recognizes certain path in the couse of reading mesh files
	
	if(mpiReducer_.rank() == 0 ) {
          std::rename(mesh_name.c_str(), temp_name.c_str());
          std::rename(vmesh_name.c_str(), mesh_name.c_str());
       }
    if(args_.optionFound("dvParallel"))
        MPI_Barrier(MPI_COMM_WORLD);
	//2.read and construct vmesh 
	Foam::fvMesh vmesh
	(
		Foam::IOobject
		(
			Foam::fvMesh::defaultRegion,
			time_.timeName(),
			time_,
			Foam::IOobject::MUST_READ
		)
	);

    if(args_.optionFound("dvParallel"))
        MPI_Barrier(MPI_COMM_WORLD);
	//3.change the folder name back, polyMesh-->vMesh,tMesh-->polyMesh,
	//so no one would know we change the name secretly :D
        if(mpiReducer_.rank() == 0) {
            std::rename(mesh_name.c_str(), vmesh_name.c_str());
            std::rename(temp_name.c_str(), mesh_name.c_str());
       }
	labelField  symmXtgID;
	labelField  symmYtgID;
	labelField  symmZtgID;

	if (mesh_.nSolutionD() == 3)    //3D(X & Y & Z)
	{
		nXiX_ = nXiY_ = nXiZ_ = vmesh.C().size();
		nXi_ = vmesh.C().size();

		weightsGlobal.setSize(nXi_);
		XisGlobal.setSize(nXi_);
		symmXtgID.setSize(nXi_);
		symmYtgID.setSize(nXi_);
		symmZtgID.setSize(nXi_);

		label i;
		for (i = 0; i < nXi_; i++) {
			vector xi(vmesh.C()[i].x(), vmesh.C()[i].y(), vmesh.C()[i].z());
			scalar weight(vmesh.V()[i]);
			weightsGlobal[i] = weight;
			XisGlobal[i] = xi;
			xiMax_.value()=max(xiMax_.value(),mag(xi));
		}
		/*

				for (label iz = 0; iz < nXiZ_; iz++)
				{
					for (label iy = 0; iy < nXiY_; iy++)
					{
						for (label ix = 0; ix < nXiZ_; ix++)
						{
							scalar weight = weights1D[iz]*weights1D[iy]*weights1D[ix];
							vector xi(Xis[ix], Xis[iy], Xis[iz]);
							weightsGlobal[i] = weight;
							XisGlobal[i] = xi;
							symmXtgID[i] = iz*nXiY_*nXiX_ + iy*nXiX_ + (nXiX_ - ix -1);
							symmYtgID[i] = iz*nXiY_*nXiX_ + (nXiY_ - iy - 1)*nXiX_ + ix;
							symmZtgID[i] = (nXiZ_ - iz -1)*nXiY_*nXiX_ + iy*nXiX_ + ix;
							i++;
						}
					}
				}
		*/
	}
	else
	{
		if (mesh_.nSolutionD() == 2)    //2D (X & Y)
		{
			nXiX_ = nXiY_ = vmesh.C().size();
			nXiZ_ = 1;
			nXi_ = vmesh.C().size();
			weightsGlobal.setSize(nXi_);
			XisGlobal.setSize(nXi_);
			symmXtgID.setSize(nXi_);
			symmYtgID.setSize(nXi_);
			symmZtgID.setSize(nXi_);
			label i;
			for (i = 0; i < nXi_; i++) {
				vector xi(vmesh.C()[i].x(), vmesh.C()[i].y(), 0.0);
				scalar weight(vmesh.V()[i]);
				weightsGlobal[i] = weight;
				XisGlobal[i] = xi;
				xiMax_.value()=max(xiMax_.value(),mag(xi));
			}
			/*
						for (label iy = 0; iy < nXiY_; iy++)
						{
							for (label ix = 0; ix < nXiX_; ix++)
							{
								scalar weight = weights1D[iy]*weights1D[ix]*1;
								vector xi(Xis[ix], Xis[iy], 0.0);
								weightsGlobal[i] = weight;
								XisGlobal[i] = xi;
								symmXtgID[i] = iy*nXiX_ + (nXiX_ - ix -1);
								symmYtgID[i] = (nXiY_ - iy - 1)*nXiX_ + ix;
								symmZtgID[i] = 0;
								i++;
							}
						}
			*/
		}
		else    //1D (X)
		{
			nXiX_ = vmesh.C().size();
			nXiY_ = nXiZ_ = 1;
			nXi_ = vmesh.C().size();
			weightsGlobal.setSize(nXi_);
			XisGlobal.setSize(nXi_);
			symmXtgID.setSize(nXi_);
			symmYtgID.setSize(nXi_);
			symmZtgID.setSize(nXi_);
			label i;
			for (i = 0; i < nXi_; i++) {
				vector xi(vmesh.C()[i].x(), 0.0, 0.0);
				scalar weight(vmesh.V()[i]);
				weightsGlobal[i] = weight;
				XisGlobal[i] = xi;
				xiMax_.value()=max(xiMax_.value(),mag(xi));
			}
			/*
						for (label ix = 0; ix < nXiX_; ix++)
						{
							scalar weight = weights1D[ix]*1*1;
							vector xi(Xis[ix], 0.0, 0.0);
							weightsGlobal[i] = weight;
							XisGlobal[i] = xi;
							symmXtgID[i] = (nXiX_ - ix -1);
							symmYtgID[i] = 0;
							symmZtgID[i] = 0;
							i++;
						}
			*/
		}
	}
#else
    //get discrete velocity points and weights
    setDVgrid
    (
         weights1D,
         Xis, 
         xiMin_.value(), 
         xiMax_.value(), 
         nXiPerDim_
    );

    labelField  symmXtgID;
    labelField  symmYtgID;
    labelField  symmZtgID;

    if (mesh_.nSolutionD() == 3)    //3D(X & Y & Z)
    {
        nXiX_ = nXiY_ = nXiZ_ = nXiPerDim_;
        nXi_ = nXiX_*nXiY_*nXiZ_;

        weightsGlobal.setSize(nXi_);
        XisGlobal.setSize(nXi_);
        symmXtgID.setSize(nXi_);
        symmYtgID.setSize(nXi_);
        symmZtgID.setSize(nXi_);

        label i = 0;
        for (label iz = 0; iz < nXiZ_; iz++)
        {
            for (label iy = 0; iy < nXiY_; iy++)
            {
                for (label ix = 0; ix < nXiZ_; ix++)
                {
                    scalar weight = weights1D[iz]*weights1D[iy]*weights1D[ix];
                    vector xi(Xis[ix], Xis[iy], Xis[iz]);
                    weightsGlobal[i] = weight;
                    XisGlobal[i] = xi;
                    symmXtgID[i] = iz*nXiY_*nXiX_ + iy*nXiX_ + (nXiX_ - ix -1);
                    symmYtgID[i] = iz*nXiY_*nXiX_ + (nXiY_ - iy - 1)*nXiX_ + ix;
                    symmZtgID[i] = (nXiZ_ - iz -1)*nXiY_*nXiX_ + iy*nXiX_ + ix;
                    i++;
                }
            }
        }
    }
    else
    {
        if (mesh_.nSolutionD() == 2)    //2D (X & Y)
        {
            nXiX_ = nXiY_ = nXiPerDim_;
            nXiZ_ = 1;
            nXi_ = nXiX_*nXiY_*nXiZ_;
            weightsGlobal.setSize(nXi_);
            XisGlobal.setSize(nXi_);
            symmXtgID.setSize(nXi_);
            symmYtgID.setSize(nXi_);
            symmZtgID.setSize(nXi_);
            label i = 0;
            for (label iy = 0; iy < nXiY_; iy++)
            {
                for (label ix = 0; ix < nXiX_; ix++)
                {
                    scalar weight = weights1D[iy]*weights1D[ix]*1;
                    vector xi(Xis[ix], Xis[iy], 0.0);
                    weightsGlobal[i] = weight;
                    XisGlobal[i] = xi;
                    symmXtgID[i] = iy*nXiX_ + (nXiX_ - ix -1);
                    symmYtgID[i] = (nXiY_ - iy - 1)*nXiX_ + ix;
                    symmZtgID[i] = 0;
                    i++;
                }
            }
        }
        else    //1D (X)
        {
            nXiX_ = nXiPerDim_;
            nXiY_ = nXiZ_ = 1;
            nXi_ = nXiX_*nXiY_*nXiZ_;
            weightsGlobal.setSize(nXi_);
            XisGlobal.setSize(nXi_);
            symmXtgID.setSize(nXi_);
            symmYtgID.setSize(nXi_);
            symmZtgID.setSize(nXi_);
            label i = 0;
            for (label ix = 0; ix < nXiX_; ix++)
            {
                scalar weight = weights1D[ix]*1*1;
                vector xi(Xis[ix], 0.0, 0.0);
                weightsGlobal[i] = weight;
                XisGlobal[i] = xi;
                symmXtgID[i] = (nXiX_ - ix -1);
                symmYtgID[i] = 0;              
                symmZtgID[i] = 0;              
                i++;
            }
        }
    }
#endif
    if(mpiReducer_.rank() == 0)
    {
        Info<< "fvDVM : Allocated " << XisGlobal.size()
            << " discrete velocities" << endl;
    }
    label nA = nXi_ / mpiReducer_.nproc();
    label nB = nXi_ - nA*mpiReducer_.nproc();
    label nXiPart = nA + (label)(mpiReducer_.rank() < nB);
    DV_.setSize(nXiPart);
    dvSize = nXiPart;
    cellSize = mesh_.nCells();
    faceSize = mesh_.nInternalFaces();
    nproc = mpiReducer_.nproc();
    rank = mpiReducer_.rank();
    // if(mpiReducer_.rank() == 0)
    // {
        Pout << "nproc    " << mpiReducer_.nproc()  << endl;
        Pout << "nXisPart " << nXiPart << endl;
        Pout << "cellSize " << cellSize << endl;
        Pout << "faceSize " << faceSize << endl;
    // }
    
    // Pout<<"dvSize*cellSize:"<<dvSize*cellSize<<endl;
    _gTildeVol = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _hTildeVol = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _gBarPvol  = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _hBarPvol  = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _gSurf     = (scalar*)malloc(dvSize*faceSize*sizeof(scalar));
    _hSurf     = (scalar*)malloc(dvSize*faceSize*sizeof(scalar));
    scalar* rho = rhoVol_.data();
    vector* U = Uvol_.data();
    scalar* T = Tvol_.data();
    scalar R = R_.value();
    label D = mesh_.nSolutionD();
#pragma omp parallel num_threads(NUM_THREADS)
{
    for(size_t i=0;i<dvSize;i++){
        #pragma omp for schedule (static) nowait
        for(size_t celli=0;celli<cellSize;celli++){

            scalar cSqrByRT = magSqr(U[celli] - XisGlobal[i*nproc+rank])/(R*T[celli]);

            scalar gEqBGK = rho[celli]/pow(sqrt(2.0*pi*R*T[celli]),D)*exp(-cSqrByRT/2.0);

            _gTildeVol[i*cellSize+celli] = gEqBGK;
            _hTildeVol[i*cellSize+celli] =  (KInner_ + 3.0 - D)  *gEqBGK*R*T[celli];

        }
    }
}
    // label chunk = 0;
    // label gid = 0;

    for(size_t i=0;i<dvSize;i++){
        volScalarField* gTildeVol_ = new volScalarField
        (
            IOobject
            (
                "gTildeVol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gTildeVol+cellSize*i
        );
        volScalarField* hTildeVol_ = new volScalarField
        (
            IOobject
            (
                "hTildeVol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hTildeVol+cellSize*i
        );
        volScalarField* gBarPvol_ =new  volScalarField
        (
            IOobject
            (
                "gBarPvol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gBarPvol+cellSize*i,
            "fixedGradient"
        );
        volScalarField* hBarPvol_ = new volScalarField
        (
            IOobject
            (
                "hBarPvol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hBarPvol+cellSize*i,
            "fixedGradient"
        );
        surfaceScalarField* gSurf_ =new surfaceScalarField
        (
            IOobject
            (
                "gSurf" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gSurf+faceSize*i
        );
        surfaceScalarField*  hSurf_ =new surfaceScalarField
        (
            IOobject
            (
                "hSurf" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hSurf+faceSize*i
        );
        gTildeVol.push_back(gTildeVol_);
        hTildeVol.push_back(hTildeVol_);
        gBarPvol.push_back(gBarPvol_);
        hBarPvol.push_back(hBarPvol_);
        gSurf.push_back(gSurf_);
        hSurf.push_back(hSurf_);

        // gid = chunk + mpiReducer_.rank();
        DV_.set
        (
            i,
            new discreteVelocity
            (
                *this,
                mesh_,
                time_,
                weightsGlobal[i*nproc+rank],
                dimensionedVector( "xi", dimLength/dimTime, XisGlobal[i*nproc+rank]),
                i,
                symmXtgID[i*nproc+rank],
                symmYtgID[i*nproc+rank],
                symmZtgID[i*nproc+rank],
                *gTildeVol_,
                *hTildeVol_,
                *gBarPvol_,
                *hBarPvol_,
                *gSurf_,
                *hSurf_
            )
        );
        // chunk += mpiReducer_.nproc();
    }
}

void Foam::fvDVM::setCalculatedMaxwellRhoBC()
{
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "calculatedMaxwell")
        {

            const vectorField& SfPatch = mesh_.Sf().boundaryField()[patchi];
            calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                refCast<calculatedMaxwellFvPatchField<scalar> >(rhoBCs[patchi]);
            const vectorField& Upatch = Uvol_.boundaryField()[patchi];
            const scalarField& Tpatch = Tvol_.boundaryField()[patchi];

            forAll(rhoPatch, facei)
            {
                vector faceSf = SfPatch[facei];
                rhoPatch.inComingByRho()[facei] = 0; // set to zero
                forAll(DV_, dvi) // add one by one
                {
                    vector xi = DV_[dvi].xi().value();
                    scalar weight = DV_[dvi].weight();
                    if ( (xi & faceSf) < 0) //inComing
                    {
                        rhoPatch.inComingByRho()[facei] += 
                          - weight*(xi & faceSf)
                          *DV_[dvi].equilibriumMaxwellByRho
                          (
                              Upatch[facei], 
                              Tpatch[facei]
                          );
                    }
                }
            }

            if(args_.optionFound("dvParallel"))
                mpiReducer_.reduceField(rhoPatch.inComingByRho());
        }

    }
}

void Foam::fvDVM::setSymmetryModRhoBC()
{
    //prepare the container (set size) to store all DF on the patchi
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        label ps = rhoBCs[patchi].size();
        if (rhoBCs[patchi].type() == "symmetryMod")
        {
            symmetryModFvPatchField<scalar>& rhoPatch = 
                refCast<symmetryModFvPatchField<scalar> >(rhoBCs[patchi]);
            rhoPatch.dfContainer().setSize(ps*nXi_*2); //*2 means g and h
        }
    }
}

void Foam::fvDVM::updateGHbarPvol()
{
    scalar dt = time_.deltaTValue();
    scalar* rho = rhoVol_.data();
    vector* U = Uvol_.data();
    scalar* T = Tvol_.data();
    vector* q = qVol_.data();
    scalar* tau = tauVol_.data();
    scalar R = R_.value();
    label D = mesh_.nSolutionD();

#pragma omp parallel num_threads(NUM_THREADS)
{
    for(size_t dvi=0;dvi<dvSize;dvi++){
        #pragma omp for schedule(static) nowait
        for(size_t celli = 0;celli<cellSize;celli++){
            scalar relaxFactor = 1.5*dt/(2.0*tau[celli] + dt);

            scalar cSqrByRT = magSqr(U[celli]-XisGlobal[dvi*nproc+rank])/(R*T[celli]);
            
            scalar cqBy5pRT = ((XisGlobal[dvi*nproc+rank] - U[celli])& q[celli])/(5.0*rho[celli]*R*T[celli]*R*T[celli]);    
            scalar gEqBGK = rho[celli]/pow(sqrt(2.0*pi*R*T[celli]),D)*exp(-cSqrByRT/2.0);
                
            _gBarPvol[dvi*cellSize+celli] = (1.0 - relaxFactor)*_gTildeVol[dvi*cellSize+celli] + relaxFactor*( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
            _hBarPvol[dvi*cellSize+celli] = (1.0 - relaxFactor)*_hTildeVol[dvi*cellSize+celli] + relaxFactor*( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*T[celli];
        }
    }
}
// if(mpiReducer_.rank()==0)
//     Info<<DV_[DV_size()-1].gBarPvol()<<nl;
}
//if split pythical space,then p2p message among processes
void Foam::fvDVM::message()
{
    forAll(DV_, DVid)
        DV_[DVid].message();
}
void Foam::fvDVM::updateGrad()
{
    forAll(DV_, DVid)
        DV_[DVid].updateGrad();
}


void Foam::fvDVM::updateGHbarSurf()
{
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceScalarField& magSf = mesh_.magSf();
    const vectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();
    const vectorField& C = mesh_.C();
    scalar dt = time_.deltaTValue();
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for(size_t dvi=0;dvi<dvSize;dvi++){
        discreteVelocity& dv = DV_[dvi];
        volVectorField& gBarPgrad_ = dv.gBarPgrad();
        volVectorField& hBarPgrad_ = dv.hBarPgrad();
        volScalarField& gBarPvol_ = dv.gBarPvol();
        volScalarField& hBarPvol_ = dv.hBarPvol();
        surfaceScalarField& gSurf_ = dv.gSurf();
        surfaceScalarField& hSurf_ = dv.hSurf();
        // #pragma omp critical
        // {
            forAll(gBarPgrad_.BOUNDARY_FIELD_REF, patchi)
            {
                const vectorField n
                (
                    Sf.boundaryField()[patchi]
                /magSf.boundaryField()[patchi]
                );

                if ( 
                    gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "empty" 
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processor"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "symmetryPlane"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "DVMsymmetry"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "cyclic"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processorCyclic"
                ) // only for fixed gradient g/hBarPvol
                {
                    // normal component of the grad field
                    fixedGradientFvPatchField<scalar>& gBarPvolPatch = 
                        refCast<fixedGradientFvPatchField<scalar> >
                        (gBarPvol_.BOUNDARY_FIELD_REF[patchi]);

                    fixedGradientFvPatchField<scalar>& hBarPvolPatch = 
                        refCast<fixedGradientFvPatchField<scalar> >
                        (hBarPvol_.BOUNDARY_FIELD_REF[patchi]);

                    forAll(gBarPvolPatch, pFacei)
                    {
                        gBarPvolPatch.gradient()[pFacei] =
                            gBarPgrad_.BOUNDARY_FIELD_REF[patchi][pFacei]&n[pFacei];
                        hBarPvolPatch.gradient()[pFacei] =
                            hBarPgrad_.BOUNDARY_FIELD_REF[patchi][pFacei]&n[pFacei];
                    }
                }
            }
        // }


        const Field<scalar>& iGbarPvol = gBarPvol_;
        const Field<scalar>& iHbarPvol = hBarPvol_;
        const Field<vector>& iGbarPgrad = gBarPgrad_;
        const Field<vector>& iHbarPgrad = hBarPgrad_;

        // This is what we want to update in this function
        Field<scalar>& iGsurf = gSurf_;
        Field<scalar>& iHsurf = hSurf_;

        vector xii = XisGlobal[dvi*nproc+rank];

        // internal faces first
        forAll(owner, facei)
        {
            label own = owner[facei];
            label nei = neighbour[facei];
            if ((xii&Sf[facei]) >=  VSMALL) // comming from own
            {

                iGsurf[facei] = iGbarPvol[own] 
                + (iGbarPgrad[own]&(Cf[facei] - C[own] - 0.5*xii*dt));

                iHsurf[facei] = iHbarPvol[own]
                + (iHbarPgrad[own]&(Cf[facei] - C[own] - 0.5*xii*dt));

            }
            // Debug, no = 0, =0 put to > 0
            else if ((xii&Sf[facei]) < -VSMALL) // comming form nei
            {
                iGsurf[facei] = iGbarPvol[nei]
                + (iGbarPgrad[nei]&(Cf[facei] - C[nei] - 0.5*xii*dt));
                iHsurf[facei] = iHbarPvol[nei]
                + (iHbarPgrad[nei]&(Cf[facei] - C[nei] - 0.5*xii*dt));
            }
            else 
            {
                iGsurf[facei] = 0.5*
                (
                    iGbarPvol[nei] + ((iGbarPgrad[nei])
                &(Cf[facei] - C[nei] - 0.5*xii*dt))
                + iGbarPvol[own] + ((iGbarPgrad[own])
                &(Cf[facei] - C[own] - 0.5*xii*dt))
                );
                iHsurf[facei] = 0.5*
                (
                iHbarPvol[nei] + ((iHbarPgrad[nei])
                &(Cf[facei] - C[nei] - 0.5*xii*dt))
                + iHbarPvol[own] + ((iHbarPgrad[own])
                &(Cf[facei] - C[own] - 0.5*xii*dt))
                );
            }
        }

        // boundary faces
        #pragma omp critical
        {
            forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
            {
                word type = gSurf_.BOUNDARY_FIELD_REF[patchi].type();
                fvsPatchField<scalar>& gSurfPatch = gSurf_.BOUNDARY_FIELD_REF[patchi];
                fvsPatchField<scalar>& hSurfPatch = hSurf_.BOUNDARY_FIELD_REF[patchi];
                const fvsPatchField<vector>& SfPatch =
                    mesh_.Sf().boundaryField()[patchi];
                const fvsPatchField<vector>& CfPatch =
                    mesh_.Cf().boundaryField()[patchi];
                const labelUList& faceCells = mesh_.boundary()[patchi].faceCells();

                const fvPatchScalarField& rhoVolPatch = 
                    rhoVol_.boundaryField()[patchi];
                const fvPatchScalarField& TvolPatch = 
                    Tvol_.boundaryField()[patchi];
                const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
                
                //- NOTE: outging DF can be treate unifily for all BCs, including processor BC
                if (type == "zeroGradient")
                {
                    gSurfPatch == gBarPvol_.BOUNDARY_FIELD_REF[patchi].patchInternalField();
                    hSurfPatch == hBarPvol_.BOUNDARY_FIELD_REF[patchi].patchInternalField();
                }
                else if (type == "mixed")
                {
                    //check each boundary face in the patch
                    forAll(gSurfPatch, facei)
                    {
                        //out or in ?
                        if ((xii&SfPatch[facei]) > 0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        //incoming and parallel to face, not changed.
                        }
                    }
                }
                else if (type == "farField")
                {
                    //check each boundary face in the patch
                    forAll(gSurfPatch, facei)
                    {
                        //out or in ?
                        if ((xii&SfPatch[facei]) > 0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        //incoming and parallel to face, not changed.
                        }
                        else // incomming, set to be equlibrium, give rho and T, extropolate U
                        {
                            // set to maxwellian
                            gSurfPatch[facei] = rhoVolPatch[facei]
                            *dv.equilibriumMaxwellByRho
                                (
                                    Uvol_[pOwner[facei]],
                                    TvolPatch[facei]
                                );
                            hSurfPatch[facei] = 
                                gSurfPatch[facei]*(R_.value()*TvolPatch[facei])
                            *(KInner_ + 3 - mesh_.nSolutionD());
                        }
                    }
                }
                else if (type == "maxwellWall")
                {
                    calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                        refCast<calculatedMaxwellFvPatchField<scalar> >
                        (rhoVol_.BOUNDARY_FIELD_REF[patchi]); //DEBUG

                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));

                            rhoPatch.outGoing()[facei] += //add outgoing normal momentum flux to outGoing container
                                weightsGlobal[dvi*nproc+rank]*(xii&faceSf)*gSurfPatch[facei];
                        }
                    }
                }
                else if (type == "processor"
                    || type == "cyclic"
                    || type == "processorCyclic"
                    ) // parallel
                {
                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  VSMALL ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        } 
                        else if ((xii&faceSf) <  -VSMALL )//incomming from processor boundaryField
                        {
                            gSurfPatch[facei] = gBarPvol_.BOUNDARY_FIELD_REF[patchi][facei]
                            + ((gBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei])
                            &(CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt));
                            hSurfPatch[facei] = hBarPvol_.BOUNDARY_FIELD_REF[patchi][facei]
                            + ((hBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei])
                            &(CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt));
                        }
                        else 
                        {
                            gSurfPatch[facei] = 0.5*(
                                    iGbarPvol[faceCells[facei]] + 
                                    (   (iGbarPgrad[faceCells[facei]]) & (CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt) )
                                    + gBarPvol_.BOUNDARY_FIELD_REF[patchi][facei] + 
                                    (   (gBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei]) & (CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt) )
                            );
                            hSurfPatch[facei] = 0.5*(
                                    iHbarPvol[faceCells[facei]] + 
                                    (   (iHbarPgrad[faceCells[facei]]) & (CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt) )
                                    + hBarPvol_.BOUNDARY_FIELD_REF[patchi][facei] + 
                                    (   (hBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei]) & (CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt) )
                            );

                        }

                    }
                }
                else if (type == "symmetryPlane" || type == "DVMsymmetry")
                {
                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  -VSMALL ) // outgoing and **reside(shouldn't be ignored)** DF, 
                                                    //incomming shoud be proceed after this function
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        } 
                    }
                }
            }
        }
    }
    // if(mpiReducer_.rank()==0)
    //     Info<<"_gBarPvol:"<<DV_[DV_.size()-1].gBarPvol()<<nl;
}


void Foam::fvDVM::updateMaxwellWallRho()
{
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "calculatedMaxwell")
        {
            calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                refCast<calculatedMaxwellFvPatchField<scalar> >(rhoBCs[patchi]);
            if(args_.optionFound("dvParallel"))
                mpiReducer_.reduceField(rhoPatch.outGoing());
        }
    }
    rhoVol_.correctBoundaryConditions();
}

void Foam::fvDVM::updateGHbarSurfMaxwellWallIn()
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    forAll(DV_, DVid){
        vector xii = XisGlobal[DVid*nproc+rank];
        discreteVelocity& dv = DV_[DVid];
        surfaceScalarField& gSurf_ = *gSurf[DVid];
        surfaceScalarField& hSurf_ = *hSurf[DVid];
        forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
        {
            if (gSurf_.BOUNDARY_FIELD_REF[patchi].type() == "maxwellWall")
            {
                fvsPatchScalarField& gSurfPatch = gSurf_.BOUNDARY_FIELD_REF[patchi];
                fvsPatchScalarField& hSurfPatch = hSurf_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchScalarField& rhoVolPatch = 
                    rhoVol_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchVectorField& UvolPatch = 
                    Uvol_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchScalarField& TvolPatch = 
                    Tvol_.BOUNDARY_FIELD_REF[patchi];
                const fvsPatchVectorField& SfPatch = 
                    mesh_.Sf().boundaryField()[patchi];
                // #pragma omp for nowait
                forAll(gSurfPatch, facei)
                {
                    vector faceSf = SfPatch[facei];
                    if ((xii & faceSf) <= 0) // incomming
                    {
                        // set to maxwellian
                        gSurfPatch[facei] = rhoVolPatch[facei]
                        *dv.equilibriumMaxwellByRho
                            (
                                UvolPatch[facei],
                                TvolPatch[facei]
                            );

                        //set hSurf at maxwellWall to zero! , WRONG!!!
                        hSurfPatch[facei] = 
                            gSurfPatch[facei]*(R_.value()*TvolPatch[facei])
                        *(KInner_ + 3 - mesh_.nSolutionD());
                    }
                }
            }
        }
    }
}

void Foam::fvDVM::updateGHbarSurfSymmetryIn()
{
    //1. copy all DV's g/h to rho patch's dfContainer
    //2. MPI_Allgather the rho patch's dfContainer
    //if(args_.optionFound("dvParallel"))
    //{
    label rank  = mpiReducer_.rank();
    label nproc = mpiReducer_.nproc();
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        label ps = rhoBCs[patchi].size();
        if (rhoBCs[patchi].type() == "symmetryMod")
        {
            symmetryModFvPatchField<scalar>& rhoPatch = 
                refCast<symmetryModFvPatchField<scalar> >(rhoBCs[patchi]);
            //compose the recvcout and displacement array
            labelField recvc(nproc);
            labelField displ(nproc);
            label chunck = nXi_/nproc;
            label left   = nXi_%nproc;
            forAll(recvc, i)
            {
                recvc[i] = 2*ps*(chunck + (i<left)) ;
                if(i<=left)
                    displ[i] = i*2*ps*(chunck + 1); // (i<=nXi_%nproc)
                else
                    displ[i] = 2*ps*(left*(chunck +1) + (i-left)*(chunck));
            }

            // check 12*28+15 dv's g
            label did = 1709;
            label pp  = did%nproc;
            label lid = did/nproc;
            //if(rank==pp)
            //{
                //Info << "processing by rank " << rank << endl;
                //Info << "12*28+15 outging g " << DV_[lid].gSurf()[0] << endl;
                //Info << "12*28+15 outging xi " << DV_[lid].xi() <<endl;
                //Info << "12*28+15 outging at boundary " << DV_[lid].gSurf().boundaryField()[patchi][0] << endl;
            //}
            // memcpy each dv's g/h to rho
            forAll(DV_, DVid)
            {
                //label shift = (nXi_ / nproc * rank + DVid)*2*ps;
                label shift = displ[rank] + DVid*2*ps;
                memcpy( (rhoPatch.dfContainer().data() + shift),
                        DV_[DVid].gSurf().boundaryField()[patchi].cdata(), ps*sizeof(scalar));
                memcpy( (rhoPatch.dfContainer().data() + shift + ps),
                        DV_[DVid].hSurf().boundaryField()[patchi].cdata(), ps*sizeof(scalar));
            }

            // check 
            //if(rank == pp)
                //Info << "dv gid 1709's g = " <<rhoPatch.dfContainer()[displ[pp]+lid*2*ps+32]<< endl;;


            //Allgather
            MPI_Allgatherv(
                //rhoPatch.dfContainer().data() + displ[rank],//2*ps*nXI_/nproc*rank, //send*
                MPI_IN_PLACE,
                2*ps*DV_.size(), //(how many DV i processed) * 2 * patch size
                MPI_DOUBLE,
                rhoPatch.dfContainer().data(),
                recvc.data(),
                displ.data(),
                MPI_DOUBLE,
                MPI_COMM_WORLD
                );
        }
    }
        forAll(DV_, DVid)
            DV_[DVid].updateGHbarSurfSymmetryIn();
}

void Foam::fvDVM::updateMacroSurf()
{
    // Init to zero before add one DV by one DV
    rhoSurf_ =  dimensionedScalar("0", rhoSurf_.dimensions(), 0);
    Usurf_ = dimensionedVector("0", Usurf_.dimensions(), vector(0, 0, 0));
    Tsurf_ = dimensionedScalar("0", Tsurf_.dimensions(), 0);
    qSurf_ = dimensionedVector("0", qSurf_.dimensions(), vector(0, 0, 0));
    stressSurf_ = dimensionedTensor
        (
            "0", 
            stressSurf_.dimensions(), 
            pTraits<tensor>::zero
        );

    surfaceVectorField rhoUsurf = rhoSurf_*Usurf_;
    surfaceScalarField rhoEsurf = rhoSurf_*magSqr(Usurf_);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for(size_t dvi=0;dvi<dvSize;dvi++)
        {
            #pragma omp for schedule(static)
            for(size_t facei=0;facei<faceSize;facei++){
                rhoSurf_[facei]  += weightsGlobal[dvi*nproc+rank]*_gSurf[dvi*faceSize+facei];
                rhoUsurf[facei]  += weightsGlobal[dvi*nproc+rank]*_gSurf[dvi*faceSize+facei]*XisGlobal[dvi*nproc+rank];
                rhoEsurf[facei]  += 0.5*weightsGlobal[dvi*nproc+rank]
                    *(
                        _gSurf[dvi*faceSize+facei]*magSqr(XisGlobal[dvi*nproc+rank]) 
                        + _hSurf[dvi*faceSize+facei]
                    );
            }
        }
    }
    forAll(mesh_.boundary(),patchi)
    {
        size_t patchSize = mesh_.boundary()[patchi].size();
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            for(size_t dvi=0;dvi<dvSize;dvi++){
                #pragma omp for schedule(static)
                for(size_t facei=0;facei<patchSize;facei++){
                    rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]  += weightsGlobal[dvi*nproc+rank]*gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei];
                    rhoUsurf.BOUNDARY_FIELD_REF[patchi][facei]  += weightsGlobal[dvi*nproc+rank]*gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei]*XisGlobal[dvi*nproc+rank];
                    rhoEsurf.BOUNDARY_FIELD_REF[patchi][facei]  += 0.5*weightsGlobal[dvi*nproc+rank]
                    *(
                            gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei]*magSqr(XisGlobal[dvi*nproc+rank]) 
                        + hSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei]
                        );
                }
            }
        }
    }
    

    if(args_.optionFound("dvParallel"))
    {
        mpiReducer_.reduceField(rhoSurf_);
        mpiReducer_.reduceField(rhoUsurf);
        mpiReducer_.reduceField(rhoEsurf);
    }

    scalar dt = time_.deltaTValue();
    //- get Prim. from Consv.
    scalar muRef = muRef_.value();
    scalar Tref = Tref_.value();
    scalar R = R_.value();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t facei=0;facei<faceSize;facei++){
        Usurf_[facei] = rhoUsurf[facei]/rhoSurf_[facei];

        Tsurf_[facei] = (rhoEsurf[facei] - 0.5*rhoSurf_[facei]*magSqr(Usurf_[facei]))/((KInner_ + 3)/2.0*R*rhoSurf_[facei]);

        tauSurf_[facei] = muRef*exp(omega_*log(Tsurf_[facei]/Tref))/rhoSurf_[facei]/Tsurf_[facei]/R;
        //- peculiar vel.

        //-get part heat flux 
        for(size_t dvi = 0;dvi<dvSize;dvi++)
        {
            vector c = XisGlobal[dvi*nproc+rank]- Usurf_[facei];
            qSurf_[facei] += 0.5*weightsGlobal[dvi*nproc+rank]*c
                *(
                    magSqr(c)*_gSurf[dvi*faceSize+facei] 
                + _hSurf[dvi*faceSize+facei] 
                );
            //- stressSurf is useless as we never update cell macro by macro flux 
            //- Comment out it as it is expansive
            //stressSurf_ += 
                //dXiCellSize_*dv.weight()*dv.gSurf()*c*c;
        }
        //- correction for bar to original
        qSurf_[facei] = 2.0*tauSurf_[facei]/(2.0*tauSurf_[facei] + 0.5*dt*Pr_)*qSurf_[facei];
    }
    forAll(mesh_.boundary(),patchi)
    {
        size_t patchSize = mesh_.boundary()[patchi].size();
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            #pragma omp for schedule(static)
            for(size_t facei=0;facei<patchSize;facei++){
                Usurf_.BOUNDARY_FIELD_REF[patchi][facei] = rhoUsurf.BOUNDARY_FIELD_REF[patchi][facei]/rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei];

                Tsurf_.BOUNDARY_FIELD_REF[patchi][facei] = (rhoEsurf.BOUNDARY_FIELD_REF[patchi][facei] - 0.5*rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]*magSqr(Usurf_.BOUNDARY_FIELD_REF[patchi][facei]))/((KInner_ + 3)/2.0*R*rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]);

                tauSurf_.BOUNDARY_FIELD_REF[patchi][facei] = muRef*exp(omega_*log(Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]/Tref))/rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]/Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]/R;
                //- peculiar vel.

                //-get part heat flux 
                for(size_t dvi = 0;dvi<dvSize;dvi++)
                {
                    vector c = XisGlobal[dvi*nproc+rank]- Usurf_.BOUNDARY_FIELD_REF[patchi][facei];
                    qSurf_.BOUNDARY_FIELD_REF[patchi][facei] += 0.5*weightsGlobal[dvi*nproc+rank]*c
                        *(
                            magSqr(c)*gSurf[dvi]-> BOUNDARY_FIELD_REF[patchi][facei]
                        + hSurf[dvi]-> BOUNDARY_FIELD_REF[patchi][facei]
                        );
                    //- stressSurf is useless as we never update cell macro by macro flux 
                    //- Comment out it as it is expansive
                    //stressSurf_ += 
                        //dXiCellSize_*dv.weight()*dv.gSurf()*c*c;
                }
                //- correction for bar to original
                qSurf_.BOUNDARY_FIELD_REF[patchi][facei] = 2.0*tauSurf_.BOUNDARY_FIELD_REF[patchi][facei]/(2.0*tauSurf_.BOUNDARY_FIELD_REF[patchi][facei] + 0.5*dt*Pr_)*qSurf_.BOUNDARY_FIELD_REF[patchi][facei];
            }

        }
    }
    
    //- Get global heat flux, via MPI_Allreuce
    if(args_.optionFound("dvParallel"))
        mpiReducer_.reduceField(qSurf_);


    //- stress at surf is not used, as we dont't update macro in cell by macro flux at surface
    //stressSurf_ = 
        //2.0*tauSurf_/(2.0*tauSurf_ + 0.5*time_.deltaT())*stressSurf_;

    //- heat flux at wall is specially defined. as it ignores the velocity and temperature slip
    //- NOTE: To be changed as it is part macro, but it will not affect the innner fields, so we change it later
// #if FOAM_MAJOR <= 3
//     GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
//         rhoBCs = rhoVol_.boundaryField();
// #else
//     GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
//         rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
// #endif
//     qWall_ = dimensionedVector("0", qWall_.dimensions(), vector(0, 0, 0));
//     stressWall_ = dimensionedTensor
//         (
//             "0", 
//             stressWall_.dimensions(), 
//             pTraits<tensor>::zero
//         );
//     forAll(rhoBCs, patchi)
//     {
//         if (rhoBCs[patchi].type() == "calculatedMaxwell")
//         {
//             fvPatchField<vector>& qPatch = qWall_.BOUNDARY_FIELD_REF[patchi];
//             fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
//             fvPatchField<tensor>& stressPatch = stressWall_.BOUNDARY_FIELD_REF[patchi];
//             //- tau at surface use the tau at slip temperature as it is.
//             fvsPatchField<scalar>&  tauPatch = tauSurf_.BOUNDARY_FIELD_REF[patchi];
//             forAll(qPatch, facei)
//             {
//                 forAll(DV_, dvi)
//                 {
//                     scalar dXiCellSize = dXiCellSize_.value();
//                     discreteVelocity& dv = DV_[dvi];
//                     vector xi = dv.xi().value();
//                     vector c = xi - Upatch[facei];
//                     qPatch[facei] += 0.5*dXiCellSize*dv.weight()*c  //sometimes wall moves, then c != \xi
//                         *(
//                              magSqr(c)*dv.gSurf().boundaryField()[patchi][facei]
//                            + dv.hSurf().boundaryField()[patchi][facei]
//                          );
//                     stressPatch[facei] += 
//                         dXiCellSize*dv.weight()*dv.gSurf().boundaryField()[patchi][facei]*xi*xi;
//                 }
//                 qPatch[facei] = 2.0*tauPatch[facei]/(2.0*tauPatch[facei] + 0.5*time_.deltaT().value()*Pr_)*qPatch[facei];
//                 stressPatch[facei] = 
//                     2.0*tauPatch[facei]/(2.0*tauPatch[facei] + 0.5*time_.deltaT().value())*stressPatch[facei];
//             }
//             if(args_.optionFound("dvParallel"))
//             {
//                 mpiReducer_.reduceField(qPatch);
//                 mpiReducer_.reduceField(stressPatch);
//             }
//         }
//     }
}

void Foam::fvDVM::updateGHsurf()
{
    scalar dt = time_.deltaTValue();
    const scalarField &rho = rhoSurf_;
    const vectorField &U = Usurf_;
    const scalarField &T = Tsurf_;
    const vectorField &q = qSurf_;
    const scalarField &tau = tauSurf_;
    scalar R = R_.value();
    label D = mesh_.nSolutionD();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for(size_t dvi=0;dvi<dvSize;dvi++){
            #pragma omp for schedule(static) nowait
            for(size_t facei = 0;facei<faceSize;facei++){
                // if constexpr (vol){
                // relaxFactor = 1.5*dt/(2.0*tau[facei] + dt);
                // }else{
                scalar relaxFactor = 0.5*dt/(2.0*tau[facei] + 0.5*dt);
                // }

                scalar cSqrByRT = magSqr(U[facei]-XisGlobal[dvi*nproc+rank])/(R*T[facei]);
                
                scalar cqBy5pRT = ((XisGlobal[dvi*nproc+rank] - U[facei])& q[facei])/(5.0*rho[facei]*R*T[facei]*R*T[facei]);    
                scalar gEqBGK = rho[facei]/pow(sqrt(2.0*pi*R*T[facei]),D)*exp(-cSqrByRT/2.0);
                    
                _gSurf[dvi*faceSize+facei] = (1.0 - relaxFactor)*_gSurf[dvi*faceSize+facei] + relaxFactor*( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
                _hSurf[dvi*faceSize+facei] = (1.0 - relaxFactor)*_hSurf[dvi*faceSize+facei] + relaxFactor*( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*T[facei];
            }
        }
    }
    forAll(mesh_.boundary(),patchi)
    {
        const fvsPatchVectorField& SfPatch = mesh_.Sf().boundaryField()[patchi];
        size_t patchSize = mesh_.boundary()[patchi].size();
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            for(size_t dvi=0;dvi<dvSize;dvi++){
                #pragma omp for schedule(static) nowait
                for(size_t facei=0;facei<patchSize;facei++){
                    if ( (XisGlobal[dvi*nproc+rank]&(SfPatch[facei])) > 0  
                    && gBarPvol[dvi]->BOUNDARY_FIELD_REF[patchi].type() != "processor"
                    && gBarPvol[dvi]->BOUNDARY_FIELD_REF[patchi].type() != "processorCyclic"
                    && gBarPvol[dvi]->BOUNDARY_FIELD_REF[patchi].type() != "cyclic")
                    {
                        scalar relaxFactor = 0.5*dt/(2.0*tauSurf_.BOUNDARY_FIELD_REF[patchi][facei] + 0.5*dt);

                        scalar cSqrByRT = magSqr(Usurf_.BOUNDARY_FIELD_REF[patchi][facei]-XisGlobal[dvi*nproc+rank])/(R*Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]);
                        
                        scalar cqBy5pRT = ((XisGlobal[dvi*nproc+rank] - Usurf_.BOUNDARY_FIELD_REF[patchi][facei])& qSurf_.BOUNDARY_FIELD_REF[patchi][facei])/(5.0*rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]*R*Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]*R*Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]);    
                        scalar gEqBGK = rhoSurf_.BOUNDARY_FIELD_REF[patchi][facei]/pow(sqrt(2.0*pi*R*Tsurf_.BOUNDARY_FIELD_REF[patchi][facei]),D)*exp(-cSqrByRT/2.0);
                        gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei] = (1.0 - relaxFactor)*gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei] + relaxFactor*( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
                        hSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei] = (1.0 - relaxFactor)*hSurf[dvi]->BOUNDARY_FIELD_REF[patchi][facei] + relaxFactor*( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*Tsurf_.BOUNDARY_FIELD_REF[patchi][facei];
                    }
                }
            }
        }
    }
    // if(mpiReducer_.rank()==0)
    //     Info<<"gSurf:"<<DV_[DV_.size()-1].gSurf()<<nl;
}

void Foam::fvDVM::updateGHtildeVol()
{
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const vectorField Sf = mesh_.Sf();
    const scalarField V = mesh_.V();
    const scalar dt = time_.deltaTValue();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for(size_t dvi=0;dvi<dvSize;dvi++){
            #pragma omp for schedule(static)    
            for(size_t celli = 0;celli<cellSize;celli++){
                _gTildeVol[dvi*cellSize+celli] = -1.0/3*_gTildeVol[dvi*cellSize+celli] + 4.0/3*_gBarPvol[dvi*cellSize+celli];
                _hTildeVol[dvi*cellSize+celli] = -1.0/3*_hTildeVol[dvi*cellSize+celli] + 4.0/3*_hBarPvol[dvi*cellSize+celli];
            }
            // forAll(gSurf_.BOUNDARY_FIELD_REF, patchi){
            //     #pragma omp for schedule(static) nowait
            //     for(size_t celli = 0;celli<gSurf_.BOUNDARY_FIELD_REF[patchi].size();celli++){
            //         gTildeVol_.BOUNDARY_FIELD_REF[patchi][celli] = -1.0/3*gTildeVol_.BOUNDARY_FIELD_REF[patchi][celli]  + 4.0/3*gBarPvol_.BOUNDARY_FIELD_REF[patchi][celli] ;
            //         hTildeVol_.BOUNDARY_FIELD_REF[patchi][celli] = -1.0/3*hTildeVol_.BOUNDARY_FIELD_REF[patchi][celli]  + 4.0/3*hBarPvol_.BOUNDARY_FIELD_REF[patchi][celli] ;
            //     } 
            // }

            // internal faces
            #pragma omp for schedule(static) nowait
            forAll(owner, facei)
            {
                const label own = owner[facei];
                const label nei = neighbour[facei];
                scalar gTildeVol_min = ((XisGlobal[dvi*nproc+rank]&Sf[facei])*_gSurf[dvi*faceSize+facei]*dt/V[own]);
                scalar gTildeVol_add = ((XisGlobal[dvi*nproc+rank]&Sf[facei])*_gSurf[dvi*faceSize+facei]*dt/V[nei]);
                scalar hTildeVol_min = ((XisGlobal[dvi*nproc+rank]&Sf[facei])*_hSurf[dvi*faceSize+facei]*dt/V[own]);
                scalar hTildeVol_add = ((XisGlobal[dvi*nproc+rank]&Sf[facei])*_hSurf[dvi*faceSize+facei]*dt/V[nei]);
                #pragma omp atomic
                _gTildeVol[dvi*cellSize+own] -= gTildeVol_min;
                #pragma omp atomic
                _gTildeVol[dvi*cellSize+nei] += gTildeVol_add;
                #pragma omp atomic
                _hTildeVol[dvi*cellSize+own] -= hTildeVol_min;
                #pragma omp atomic
                _hTildeVol[dvi*cellSize+nei] += hTildeVol_add;
            }
        }
    }

        // boundary faces
    forAll(mesh_.boundary(),patchi)
    {
        const fvsPatchField<vector>& SfPatch =
            mesh_.Sf().boundaryField()[patchi];
        const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            for(size_t dvi=0;dvi<dvSize;dvi++){
                
                #pragma omp for schedule(static) nowait
                forAll(pOwner, pFacei)
                {
                    const label own = pOwner[pFacei];
                    scalar gTildeVol_min = (XisGlobal[dvi*nproc+rank]&SfPatch[pFacei]) *gSurf[dvi]->BOUNDARY_FIELD_REF[patchi][pFacei]*dt/V[own];
                    scalar hTildeVol_min = (XisGlobal[dvi*nproc+rank]&SfPatch[pFacei]) *hSurf[dvi]->BOUNDARY_FIELD_REF[patchi][pFacei]*dt/V[own];
                    #pragma omp atomic
                    _gTildeVol[dvi*cellSize+own] -= gTildeVol_min;
                    #pragma omp atomic
                    _hTildeVol[dvi*cellSize+own] -= hTildeVol_min;
                }
            }
        }
    }
    // if(mpiReducer_.rank()==0)
    //     Info<<"_gTildeVol:"<<DV_[DV_.size()-1].gTildeVol()<<nl;
    

}

void Foam::fvDVM::updateMacroVol()
{
    //- Old macros, used only if we update using macro fluxes.
    volVectorField rhoUvol = rhoVol_*Uvol_;
    volScalarField rhoEvol = rhoVol_*(0.5* magSqr(Uvol_) + (KInner_ + 3)/2.0*R_*Tvol_);
    qVol_ = dimensionedVector("0", qVol_.dimensions(), vector(0, 0, 0));

    if(macroFlux_ == "no") // update cell macro by moment from DF
    {
        //- init to zeros
        rhoVol_ = dimensionedScalar("0", rhoVol_.dimensions(), 0);
        rhoUvol = dimensionedVector("0", rhoUvol.dimensions(), vector(0, 0, 0));
        rhoEvol = dimensionedScalar("0", rhoEvol.dimensions(), 0);

        //- get part macro
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            for(size_t dvi=0;dvi<dvSize;dvi++)
            {
                #pragma omp for schedule(static)
                for(size_t celli=0;celli<cellSize;celli++){
                    rhoVol_[celli]  += weightsGlobal[dvi*nproc+rank]*_gTildeVol[dvi*cellSize+celli];
                    rhoUvol[celli]  += weightsGlobal[dvi*nproc+rank]*_gTildeVol[dvi*cellSize+celli]*XisGlobal[dvi*nproc+rank];
                    rhoEvol[celli]  += 0.5*weightsGlobal[dvi*nproc+rank]
                        *(
                            _gTildeVol[dvi*cellSize+celli]*magSqr(XisGlobal[dvi*nproc+rank]) 
                            + _hTildeVol[dvi*cellSize+celli]
                        );
                }
            }
        }
        // forAll(mesh_.boundary(),patchi)
        // {
        //     size_t patchSize = mesh_.boundary()[patchi].size();
        //     #pragma omp parallel num_threads(NUM_THREADS)
        //     {
        //         for(size_t dvi=0;dvi<dvSize;dvi++){
        //             #pragma omp for schedule(static)
        //             for(size_t celli=0;celli<patchSize;celli++){
        //                 rhoVol_.BOUNDARY_FIELD_REF[patchi][celli]  += weightsGlobal[dvi*nproc+rank]*gTildeVol[dvi]->BOUNDARY_FIELD_REF[patchi][celli];
        //                 rhoUvol.BOUNDARY_FIELD_REF[patchi][celli]  += weightsGlobal[dvi*nproc+rank]*gTildeVol[dvi]->BOUNDARY_FIELD_REF[patchi][celli]*XisGlobal[dvi*nproc+rank];
        //                 rhoEvol.BOUNDARY_FIELD_REF[patchi][celli]  += 0.5*weightsGlobal[dvi*nproc+rank]
        //                 *(
        //                         gTildeVol[dvi]->BOUNDARY_FIELD_REF[patchi][celli]*magSqr(XisGlobal[dvi*nproc+rank]) 
        //                     + hTildeVol[dvi]->BOUNDARY_FIELD_REF[patchi][celli]
        //                     );
        //             }
        //         }
        //     }
        // }
        //- get global macro via MPI_Allreduce
        if(args_.optionFound("dvParallel"))
        {
            mpiReducer_.reduceField(rhoVol_);
            mpiReducer_.reduceField(rhoUvol);
            mpiReducer_.reduceField(rhoEvol);
        }
    }
    else // update by macro flux
    {
        const labelUList &owner = mesh_.owner();
		const labelUList &neighbour = mesh_.neighbour();
		const vectorField Sf = mesh_.Sf();
		const scalarField V = mesh_.V();
		const scalar dt = time_.deltaTValue();
#ifdef VMESH
	    //init flux to zero
		rhoflux_ = dimensionedScalar("0", rhoVol_.dimensions(), 0);
		volVectorField rhouflux_= rhoflux_*Uvol_;
		volScalarField rhoeflux_ = rhoflux_ * magSqr(Uvol_);

		forAll(DV_, dvi)
		{    
			//temp var 
			rho_ = dimensionedScalar("0", rhoSurf_.dimensions(), 0);
			surfaceVectorField u_= rho_*Usurf_;
			surfaceScalarField e_ = rho_ * magSqr(Usurf_);

			discreteVelocity &dv = DV_[dvi];
			vector xii = dv.xi().value();

			rho_ = dXiCellSize_ * dv.weight() * dv.gSurf(); 
			u_ = dXiCellSize_ * dv.weight() * dv.gSurf() * dv.xi();
			e_  = 0.5*dXiCellSize_*dv.weight()
           *(
                dv.gSurf()*magSqr(dv.xi()) 
              + dv.hSurf()
            );
			
			//internal faces
			forAll(owner, facei)
			{
				const label own = owner[facei];
				const label nei = neighbour[facei];

				rhoflux_[own] -= ((xii & Sf[facei]) * rho_[facei] * dt / V[own]);
				rhoflux_[nei] += ((xii & Sf[facei]) * rho_[facei] * dt / V[nei]);
				rhouflux_[own] -= ((xii & Sf[facei]) * u_[facei] * dt / V[own]);
				rhouflux_[nei] += ((xii & Sf[facei]) * u_[facei] * dt / V[nei]);
				rhoeflux_[own] -= ((xii & Sf[facei]) * e_[facei] * dt / V[own]);
				rhoeflux_[nei] += ((xii & Sf[facei]) * e_[facei] * dt / V[nei]);
			}

			forAll(rhoSurf_.boundaryField(), patchi)
			{   
				const fvsPatchField<vector> &SfPatch = mesh_.Sf().boundaryField()[patchi];
				const labelUList &pOwner = mesh_.boundary()[patchi].faceCells();

				forAll(pOwner, pFacei)
				{
					const label own = pOwner[pFacei];
					rhoflux_[own] -= ((xii & SfPatch[pFacei]) * rho_.boundaryField()[patchi][pFacei] * dt / V[own]);
					rhouflux_[own] -= ((xii & SfPatch[pFacei]) * u_.boundaryField()[patchi][pFacei] * dt / V[own]);
					rhoeflux_[own] -= ((xii & SfPatch[pFacei]) * e_.boundaryField()[patchi][pFacei]  * dt / V[own]);
				}
			}
		}
        

		if(args_.optionFound("dvParallel"))
		{
			mpiReducer_.reduceField(rhoflux_);
			mpiReducer_.reduceField(rhouflux_);
			mpiReducer_.reduceField(rhoeflux_);
		}
  
        rhoVol_ += rhoflux_;
		rhoUvol += rhouflux_;
		rhoEvol += rhoeflux_;
#else
        // internal faces
        forAll(owner, facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            rhoVol_[own] -= (rhoSurf_[facei]*Usurf_[facei]&Sf[facei])*dt/V[own];
            rhoVol_[nei] += (rhoSurf_[facei]*Usurf_[facei]&Sf[facei])*dt/V[nei];
            rhoUvol[own] -= (rhoSurf_[facei]*Usurf_[facei]*Usurf_[facei]
                    + stressSurf_[facei])&Sf[facei]*dt/V[own];
            rhoUvol[nei] += (rhoSurf_[facei]*Usurf_[facei]*Usurf_[facei]
                    + stressSurf_[facei])&Sf[facei]*dt/V[nei];
            scalar rhoEsurf = 
                rhoSurf_[facei]
                *(magSqr(Usurf_[facei]) + (KInner_ + 3)/2.0*R_.value()*Tsurf_[facei]);

            rhoEvol[own] -= (rhoEsurf*Usurf_[facei] + qSurf_[facei])
                &Sf[facei]*dt/V[own];
            rhoEvol[nei] += (rhoEsurf*Usurf_[facei] + qSurf_[facei])
                &Sf[facei]*dt/V[nei];
        }
        // boundary faces
        forAll(rhoSurf_.boundaryField(), patchi)
        {
            const fvsPatchField<scalar>& rhoSurfPatch =
                rhoSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& UsurfPatch =
                Usurf_.boundaryField()[patchi];
            const fvsPatchField<scalar>& TsurfPatch =
                Tsurf_.boundaryField()[patchi];
            const fvsPatchField<tensor>& stressSurfPatch =
                stressSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& qSurfPatch =
                qSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& SfPatch =
                mesh_.Sf().boundaryField()[patchi];

            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                rhoVol_[own] -= (rhoSurfPatch[pFacei]*UsurfPatch[pFacei]
                        &SfPatch[pFacei])*dt/V[own];
                rhoUvol[own] -= (rhoSurfPatch[pFacei]*UsurfPatch[pFacei]*UsurfPatch[pFacei]
                        + stressSurfPatch[pFacei])&Sf[pFacei]*dt/V[own];
                scalar rhoEsurf = 
                    rhoSurfPatch[pFacei]
                    *(magSqr(UsurfPatch[pFacei]) + (KInner_ + 3)/2.0*R_.value()*TsurfPatch[pFacei]);

                rhoEvol[own] -= (rhoEsurf*UsurfPatch[pFacei] + qSurfPatch[pFacei])
                    &SfPatch[pFacei]*dt/V[own];
            }
        }
#endif
    }
    scalar dt = time_.deltaTValue();
    //- get Prim. from Consv.
    scalar muRef = muRef_.value();
    scalar Tref = Tref_.value();
    scalar R = R_.value();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t celli=0;celli<cellSize;celli++){
        Uvol_[celli] = rhoUvol[celli]/rhoVol_[celli];

        Tvol_[celli] = (rhoEvol[celli] - 0.5*rhoVol_[celli]*magSqr(Uvol_[celli]))/((KInner_ + 3)/2.0*R*rhoVol_[celli]);

        tauVol_[celli] = muRef*exp(omega_*log(Tvol_[celli]/Tref))/rhoVol_[celli]/Tvol_[celli]/R;
        //- peculiar vel.

        //-get part heat flux 
        for(size_t dvi = 0;dvi<dvSize;dvi++)
        {
            vector c = XisGlobal[dvi*nproc+rank]- Uvol_[celli];
            qVol_[celli] += 0.5*weightsGlobal[dvi*nproc+rank]*c
                *(
                    magSqr(c)*_gTildeVol[dvi*cellSize+celli] 
                + _hTildeVol[dvi*cellSize+celli] 
                );
            //- stressSurf is useless as we never update cell macro by macro flux 
            //- Comment out it as it is expansive
            //stressSurf_ += 
                //dXiCellSize_*dv.weight()*dv.gSurf()*c*c;
        }
        //- correction for bar to original
        qVol_[celli] = 2.0*tauVol_[celli]/(2.0*tauVol_[celli] + dt*Pr_)*qVol_[celli];
    }

    //- get global heat flux via MPI_Allreduce
    if(args_.optionFound("dvParallel"))
        mpiReducer_.reduceField(qVol_);
    Uvol_.correctBoundaryConditions();
    Tvol_.correctBoundaryConditions();
}

void Foam::fvDVM::updatePressureInOutBC()
{
    // for pressureIn and pressureOut BC, the boundary value of Uvol(in/out) and Tvol(in/out) should be updated here!
    // boundary faces
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "pressureIn")
        {
            const fvsPatchField<vector>& SfPatch = mesh_.Sf().boundaryField()[patchi];
            const fvsPatchField<scalar>& magSfPatch = mesh_.magSf().boundaryField()[patchi];
            pressureInFvPatchField<scalar>& rhoPatch = 
                refCast<pressureInFvPatchField<scalar> >(rhoBCs[patchi]);
            fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
            const fvPatchField<scalar>& Tpatch = Tvol_.boundaryField()[patchi];
            const scalar pressureIn = rhoPatch.pressureIn();
            // now changed rho and U patch
            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(rhoPatch, facei)
            {
                const scalar  Tin = Tpatch[facei];
                // change density
                rhoPatch[facei] = pressureIn/R_.value()/Tin; // Accturally not changed at all :p

                // inner boundary cell data state data
                label own = pOwner[facei];
                vector Ui = Uvol_[own];
                scalar Ti = Tvol_[own];
                scalar rhoi = rhoVol_[own];
                scalar ai = sqrt(R_.value() * Ti * (KInner_ + 5)/(KInner_ + 3)); // sos

                // change normal velocity component based on the characteristics
                vector norm = SfPatch[facei]/magSfPatch[facei]; // boundary face normal vector
                scalar Un = Ui & norm; // normal component
                scalar UnIn = Un + (pressureIn - rhoi * R_.value() * Ti)/rhoi/ai; // change normal component
                Upatch[facei] = UnIn * norm + (Ui - Un * norm); // tangential component not changed.
            }
        }
        else if(rhoBCs[patchi].type() == "pressureOut")
        {
            const fvsPatchField<vector>& SfPatch = mesh_.Sf().boundaryField()[patchi];
            const fvsPatchField<scalar>& magSfPatch = mesh_.magSf().boundaryField()[patchi];
            pressureOutFvPatchField<scalar>& rhoPatch = 
                refCast<pressureOutFvPatchField<scalar> >(rhoBCs[patchi]);
            fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
            fvPatchField<scalar>& Tpatch = Tvol_.BOUNDARY_FIELD_REF[patchi];
            const scalar pressureOut = rhoPatch.pressureOut();
            // now changed rho and U patch
            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(rhoPatch, facei)
            {
                // inner cell data state data
                label own = pOwner[facei];
                vector Ui = Uvol_[own];
                scalar Ti = Tvol_[own];
                scalar rhoi = rhoVol_[own];
                scalar ai = sqrt(R_.value() * Ti * (KInner_ + 5)/(KInner_ + 3)); // sos

                // change outlet density
                rhoPatch[facei] = rhoi  +  (pressureOut - rhoi * R_.value() * Ti)/ai/ai; // Accturally not changed at all :p
                Tpatch[facei] = pressureOut/(R_.value() * rhoi);

                // change normal velocity component based on the characteristics
                vector norm = SfPatch[facei]/magSfPatch[facei]; // boundary face normal vector
                scalar Un = Ui & norm; // normal component
                scalar UnIn = Un + ( rhoi * R_.value() * Ti - pressureOut)/rhoi/ai; // change normal component
                Upatch[facei] = UnIn * norm + (Ui - Un * norm); // tangential component not changed.
            }
        }
    }
}

template<template<class> class PatchType, class GeoMesh>
void Foam::fvDVM::updateTau
(
 GeometricField<scalar, PatchType, GeoMesh>& tau,
 const GeometricField<scalar, PatchType, GeoMesh>& T, 
 const GeometricField<scalar, PatchType, GeoMesh>& rho
 )
{
    tau = muRef_*exp(omega_*log(T/Tref_))/rho/T/R_;
}


void Foam::fvDVM::writeDFonCell(label cellId)
{
    std::ostringstream convert;
    convert << cellId;
    scalarIOList df
    (
        IOobject
        (
             "DF"+convert.str(),
             "0",
             mesh_,
             IOobject::NO_READ,
             IOobject::AUTO_WRITE
        )
    );
    //set size of df
    df.setSize(nXi_);

    scalarList dfPart(DV_.size());
    //put cellId's DF to dfPart
    forAll(dfPart, dfi)
        dfPart[dfi] = DV_[dfi].gTildeVol()[cellId];

    label nproc = mpiReducer_.nproc();
    //gather
    //tmp list for recv
    scalarList dfRcv(nXi_);

    //Compose displc and recvc
    labelField recvc(nproc);
    labelField displ(nproc);
    label chunck = nXi_/nproc;
    label left   = nXi_%nproc;
    forAll(recvc, i)
    {
        recvc[i] = chunck + (i<left) ;
        if(i<=left)
            displ[i] = i*(chunck + 1); // (i<=nXi_%nproc)
        else
            displ[i] = left*(chunck +1) + (i-left)*(chunck);
    }
    MPI_Gatherv(dfPart.data(), dfPart.size(), MPI_DOUBLE,
            dfRcv.data(), recvc.data(), displ.data(),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //reposition
    if(mpiReducer_.rank() == 0)
    {
        forAll(df, i)
        {
            label p   = i%nproc;
            label ldi = i/nproc;
            df[i] = dfRcv[displ[p]+ldi];
        }
        df.write();
    }
}

void Foam::fvDVM::writeDFonCells()
{
    if (time_.outputTime())
        forAll(DFwriteCellList_, i)
            writeDFonCell(i);
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

    Foam::fvDVM::fvDVM
(
 volScalarField& rho,
 volVectorField& U,
 volScalarField& T,
 int* argc,
 char*** argv,
 Foam::argList& args
 )
    :
        IOdictionary
        (
         IOobject
         (
          "DVMProperties",
          T.time().constant(),
          T.mesh(),
          IOobject::MUST_READ,
          IOobject::NO_WRITE
         )
        ),
        mesh_(rho.mesh()),
        time_(rho.time()),
        rhoVol_(rho),
        Uvol_(U),
        Tvol_(T),
        args_(args),
        fvDVMparas_(subOrEmptyDict("fvDVMparas")),
        gasProperties_(subOrEmptyDict("gasProperties")),
        nXiPerDim_(readLabel(fvDVMparas_.lookup("nDV"))),
        xiMax_(fvDVMparas_.lookup("xiMax")),
        xiMin_(fvDVMparas_.lookup("xiMin")),
        dXi_((xiMax_-xiMin_)/(nXiPerDim_ - 1)),
        dXiCellSize_
        (
         "dXiCellSize",
         pow(dimLength/dimTime, 3),
         scalar(1.0)
        ),
        macroFlux_(fvDVMparas_.lookupOrDefault("macroFlux", word("no"))),
        //res_(fvDVMparas_.lookupOrDefault("res", 1.0e-12)),
        //checkSteps_(fvDVMparas_.lookupOrDefault("checkSteps", 100)),
        R_(gasProperties_.lookup("R")),
        omega_(readScalar(gasProperties_.lookup("omega"))),
        Tref_(gasProperties_.lookup("Tref")),
        muRef_(gasProperties_.lookup("muRef")),
        Pr_(readScalar(gasProperties_.lookup("Pr"))),
        KInner_((gasProperties_.lookupOrDefault("KInner", 0))),
        mpiReducer_(args, argc, argv), // args comes from setRootCase.H in dugksFoam.C;
        DV_(0),
        rhoSurf_
        (
         IOobject
         (
          "rhoSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", rho.dimensions(), 0)
        ),
        rhoflux_
        (
            IOobject
            (
                "rhoflux",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("0", rho.dimensions(), 0)
        ),
        rho_
        (
            IOobject
            (
                "rhotemp",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("0", rho.dimensions(), 0)
        ),
        Tsurf_
        (
         IOobject
         (
          "Tsurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", T.dimensions(), 0)
        ),
        Usurf_
        (
         IOobject
         (
          "Usurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", U.dimensions(), vector(0,0,0))
        ),
        qSurf_
        (
         IOobject
         (
          "qSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        stressSurf_
        (
         IOobject
         (
          "stressSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedTensor("0", dimensionSet(1,-1,-2,0,0,0,0), pTraits<tensor>::zero)
        ),
        qVol_
        (
         IOobject
         (
          "q",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        tauVol_
        (
         IOobject
         (
          "tauVol",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", dimTime, 0)
        ),
        tauSurf_
        (
         IOobject
         (
          "tauSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", dimTime, 0)
        ),
        qWall_
        (
         IOobject
         (
          "qWall",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        stressWall_
        (
         IOobject
         (
          "stressWall",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedTensor("0", dimensionSet(1,-1,-2,0,0,0,0), pTraits<tensor>::zero)
        )
{
    DFwriteCellList_ = lookupOrDefault<labelList>("DFwriteCellList", labelList()),
    initialiseDV();
    setCalculatedMaxwellRhoBC();
    setSymmetryModRhoBC();
    // set initial rho in pressureIn/Out BC
    updatePressureInOutBC();
    updateTau(tauVol_, Tvol_, rhoVol_); //calculate the tau at cell when init
    Usurf_ = fvc::interpolate(Uvol_, "linear"); // for first time Dt calculation.
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fvDVM::~fvDVM()
{
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fvDVM::evolution()
{
    // Info << "Begin evolution" << endl;
    TICK(updateGHbarPvol)
    updateGHbarPvol();
    TOCK(updateGHbarPvol)
    TICK(message)
    message();
    TOCK(message)
    TICK(updateGrad)
    updateGrad();
    TOCK(updateGrad)
    // Info << "Done updateGHbarPvol " << endl;
    TICK(updateGHbarSurf)
    updateGHbarSurf();
    TOCK(updateGHbarSurf)
    // Info << "Done updateGHbarSurf " << endl;
    TICK(updateMaxwellWallRho)
    updateMaxwellWallRho();
    TOCK(updateMaxwellWallRho)
    // Info << "Done updateMaxwellWallRho " << endl;
    TICK(updateGHbarSurfMaxwellWallIn)
    updateGHbarSurfMaxwellWallIn();
    TOCK(updateGHbarSurfMaxwellWallIn)
    // Info << "Done updateGHbarSurfMaxwellWallIn " << endl;
    TICK(updateGHbarSurfSymmetryIn)
    updateGHbarSurfSymmetryIn();
    TOCK(updateGHbarSurfSymmetryIn)
    // Info << "Done updateGHbarSurfSymmetryIn " << endl;
    TICK(updateMacroSurf)
    updateMacroSurf();
    TOCK(updateMacroSurf)
    // Info << "Done updateMacroSurf " << endl;
    TICK(updateGHsurf)
    updateGHsurf();
    TOCK(updateGHsurf)
    // Info << "Done updateGHsurf " << endl;
    TICK(updateGHtildeVol)
    updateGHtildeVol();
    TOCK(updateGHtildeVol)
    // Info << "Done updateGHtildeVol " << endl;
    TICK(updateMacroVol)
    updateMacroVol();
    TOCK(updateMacroVol)
    //
    updatePressureInOutBC();
}


void Foam::fvDVM::getCoNum(scalar& maxCoNum, scalar& meanCoNum)
{
    scalar dt = time_.deltaTValue();
    scalarField UbyDx =
        mesh_.surfaceInterpolation::deltaCoeffs()
        *(mag(Usurf_) + sqrt(scalar(mesh_.nSolutionD()))*xiMax_);
    maxCoNum = gMax(UbyDx)*dt;
    meanCoNum = gSum(UbyDx)/UbyDx.size()*dt;
}


const fieldMPIreducer& Foam::fvDVM::mpiReducer() const
{
    return mpiReducer_;
}


// ************************************************************************* //
