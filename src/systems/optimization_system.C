// The libMesh Finite Element Library.
// Copyright (C) 2002-2014 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



// C++ includes

// Local includes
#include "libmesh/equation_systems.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/optimization_solver.h"
#include "libmesh/optimization_system.h"

namespace libMesh
{

// ------------------------------------------------------------
// OptimizationSystem implementation
OptimizationSystem::OptimizationSystem (EquationSystems& es,
                                        const std::string& name_in,
                                        const unsigned int number_in) :

  Parent(es, name_in, number_in),
  optimization_solver(OptimizationSolver<Number>::build(*this)),
  C_eq(NumericVector<Number>::build(this->comm()))
{
}



OptimizationSystem::~OptimizationSystem ()
{
  // Clear data
  this->clear();
}



void OptimizationSystem::clear ()
{
  // clear the optimization solver
  optimization_solver->clear();

  // clear the parent data
  Parent::clear();
}



void OptimizationSystem::reinit ()
{
  optimization_solver->clear();

  Parent::reinit();
}


void OptimizationSystem::initialize_equality_constraints_storage(
  unsigned int n_eq_constraints)
{
  C_eq->init(n_eq_constraints, false, SERIAL);
}


void OptimizationSystem::solve ()
{
  START_LOG("solve()", "OptimizationSystem");

  optimization_solver->init();
  optimization_solver->solve ();

  STOP_LOG("solve()", "OptimizationSystem");

  this->update();
}


} // namespace libMesh
