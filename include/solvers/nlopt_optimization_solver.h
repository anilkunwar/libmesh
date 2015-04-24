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



#ifndef LIBMESH_NLOPT_OPTIMIZATION_SOLVER_H
#define LIBMESH_NLOPT_OPTIMIZATION_SOLVER_H

#include "libmesh/libmesh_config.h"

// Petsc include files.
#if defined(LIBMESH_HAVE_NLOPT) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)

// Local includes
#include "libmesh/optimization_solver.h"

// NLopt include (use C++ header)
#include "nlopt.hpp"

// C++ includes

namespace libMesh
{

  // Allow users access to these functions in case they want to reuse them.  Note that users shouldn't
  // need access to these most of the time as they are used internally by this object.
  Real __libmesh_nlopt_objective(const std::vector<double> &x, std::vector<double> &grad, void *data);
//  Real __libmesh_nlopt_equality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data);
//  Real __libmesh_nlopt_inequality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data);

/**
 * This class provides an interface to the Tao optimization solvers.
 *
 * @author David Knezevic, 2015
 */

template <typename T>
class NloptOptimizationSolver : public OptimizationSolver<T>
{
public:

  /**
   * The type of system that we use in conjunction with this solver.
   */
  typedef OptimizationSystem sys_type;

  /**
   *  Constructor. Initializes Tao data structures.
   */
  explicit
  NloptOptimizationSolver (sys_type& system);

  /**
   * Destructor.
   */
  ~NloptOptimizationSolver ();

  /**
   * Release all memory and clear data structures.
   */
  virtual void clear ();

  /**
   * Initialize data structures if not done so already.
   */
  virtual void init ();

  /**
   * Returns the raw NLopt object.
   */
  nlopt::opt get_nlopt_object() { this->init(); return *_nlopt; }

  /**
   * Call the NLopt solver.
   */
  virtual void solve ();

  /**
   * Prints a useful message about why the latest optimization solve
   * con(di)verged.
   */
  virtual void print_converged_reason();

  /**
   * Returns the currently-available (or most recently obtained, if the NLopt object has
   * been destroyed) convergence reason.  Refer to NLopt docs for the meaning of different
   * the value.
   */
  nlopt::result get_converged_reason();

protected:

  /**
   * Optimization solver context
   */
  UniquePtr<nlopt::opt> _nlopt;

  /**
   * Store the result (i.e. convergence/divergence) for the most recent NLopt solve.
   */
  nlopt::result _result;

private:

  // Callback functions to be passed to NLopt
  friend Real __libmesh_nlopt_objective (const std::vector<double> &x, std::vector<double> &grad, void *data);
//  friend Real __libmesh_nlopt_equality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data);
//  friend Real __libmesh_nlopt_inequality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data);
};



} // namespace libMesh


#endif // #if defined(LIBMESH_HAVE_NLOPT) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)
#endif // LIBMESH_NLOPT_OPTIMIZATION_SOLVER_H
