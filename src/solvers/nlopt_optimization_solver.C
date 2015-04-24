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



#include "libmesh/libmesh_common.h"

#if defined(LIBMESH_HAVE_NLOPT) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)


// C++ includes

// Local Includes
#include "libmesh/dof_map.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/nlopt_optimization_solver.h"

namespace libMesh
{

  Real __libmesh_nlopt_objective(const std::vector<double> &x, std::vector<double> &grad, void *data)
  {
    START_LOG("objective()", "NloptOptimizationSolver");

    // ctx should be a pointer to the solver (it was passed in as void*)
    NloptOptimizationSolver<Number>* solver =
      static_cast<NloptOptimizationSolver<Number>*> (data);

    OptimizationSystem &sys = solver->system();

    // We'll use current_local_solution below, so let's ensure that it's consistent
    // with the vector x that was passed in.
    for(unsigned int i=sys.solution->first_local_index(); i<sys.solution->last_local_index(); i++)
    {
      sys.solution->set(i, x[i]);
    }
    sys.solution->close();

    // Impose constraints on X
    sys.get_dof_map().enforce_constraints_exactly(sys);
    // Update sys.current_local_solution based on X
    sys.update();

    Real objective;
    if (solver->objective_object != NULL)
    {
      objective =
        solver->objective_object->objective(
          *(sys.current_local_solution), sys);
    }
    else
    {
      libmesh_error_msg("Objective function not defined in __libmesh_nlopt_objective");
    }

    // if grad.size() > 0, then the gradient has been requested
    if(grad.size() > 0)
    {
      if (solver->gradient_object != NULL)
      {
        solver->gradient_object->gradient(
          *(sys.current_local_solution), *(sys.rhs), sys);

        // we've filled up sys.rhs with the gradient data, now copy it to g
        libmesh_assert(sys.rhs->size() == grad.size());
        sys.rhs->localize_to_one(grad);
      }
      else
      {
        libmesh_error_msg("Gradient function not defined in __libmesh_nlopt_objective");
      }
    }    

    STOP_LOG("objective()", "NloptOptimizationSolver");

    return objective;
  }

//  Real __libmesh_nlopt_equality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data);
//
//  //---------------------------------------------------------------
//  // This function is called by Tao to evaluate the equality constraints at x
//  PetscErrorCode
//  __libmesh_tao_equality_constraints(Tao /*tao*/, Vec x, Vec ce, void *ctx)
//  {
//    START_LOG("equality_constraints()", "TaoOptimizationSolver");
//
//    PetscErrorCode ierr = 0;
//    
//    libmesh_assert(x);
//    libmesh_assert(ce);
//    libmesh_assert(ctx);
//
//    // ctx should be a pointer to the solver (it was passed in as void*)
//    TaoOptimizationSolver<Number>* solver =
//      static_cast<TaoOptimizationSolver<Number>*> (ctx);
//
//    OptimizationSystem &sys = solver->system();
//
//    // We'll use current_local_solution below, so let's ensure that it's consistent
//    // with the vector x that was passed in.
//    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
//    PetscVector<Number> X(x, sys.comm());
//
//    // Perform a swap so that sys.solution points to X
//    X.swap(X_sys);
//    // Impose constraints on X
//    sys.get_dof_map().enforce_constraints_exactly(sys);
//    // Update sys.current_local_solution based on X
//    sys.update();
//    // Swap back
//    X.swap(X_sys);
//
//    // We'll also pass the constraints vector ce into the assembly routine
//    // so let's make a PETSc vector for that too.
//    PetscVector<Number> eq_constraints(ce, sys.comm());
//
//    // Clear the gradient prior to assembly
//    eq_constraints.zero();
//
//    if (solver->equality_constraints_object != NULL)
//    {
//      solver->equality_constraints_object->equality_constraints(
//        *(sys.current_local_solution), eq_constraints, sys);
//    }
//    else
//    {
//      libmesh_error_msg("Constraints function not defined in __libmesh_tao_equality_constraints");
//    }
//
//    eq_constraints.close();
//
//    STOP_LOG("equality_constraints()", "TaoOptimizationSolver");
//
//    return ierr;
//  }
//
//  //---------------------------------------------------------------
//  // This function is called by Tao to evaluate the Jacobian of the
//  // equality constraints at x
//  PetscErrorCode
//  __libmesh_tao_equality_constraints_jacobian(Tao /*tao*/, Vec x, Mat J, Mat Jpre, void *ctx)
//  {
//    START_LOG("equality_constraints_jacobian()", "TaoOptimizationSolver");
//
//    PetscErrorCode ierr = 0;
//    
//    libmesh_assert(x);
//    libmesh_assert(J);
//    libmesh_assert(Jpre);
//
//    // ctx should be a pointer to the solver (it was passed in as void*)
//    TaoOptimizationSolver<Number>* solver =
//      static_cast<TaoOptimizationSolver<Number>*> (ctx);
//
//    OptimizationSystem &sys = solver->system();
//
//    // We'll use current_local_solution below, so let's ensure that it's consistent
//    // with the vector x that was passed in.
//    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
//    PetscVector<Number> X(x, sys.comm());
//
//    // Perform a swap so that sys.solution points to X
//    X.swap(X_sys);
//    // Impose constraints on X
//    sys.get_dof_map().enforce_constraints_exactly(sys);
//    // Update sys.current_local_solution based on X
//    sys.update();
//    // Swap back
//    X.swap(X_sys);
//
//    // Let's also wrap J and Jpre in PetscMatrix objects for convenience
//    PetscMatrix<Number> J_petsc(J, sys.comm());
//    PetscMatrix<Number> Jpre_petsc(Jpre, sys.comm());
//
//    if (solver->equality_constraints_jacobian_object != NULL)
//    {
//      solver->equality_constraints_jacobian_object->equality_constraints_jacobian(
//        *(sys.current_local_solution), J_petsc, sys);
//    }
//    else
//    {
//      libmesh_error_msg("Constraints function not defined in __libmesh_tao_equality_constraints_jacobian");
//    }
//
//    J_petsc.close();
//    Jpre_petsc.close();
//
//    STOP_LOG("equality_constraints_jacobian()", "TaoOptimizationSolver");
//
//    return ierr;
//  }
//
//  Real __libmesh_nlopt_inequality_constraints(const std::vector<Real> &x, std::vector<Real> &grad, void* data)
//  {
//  
//  }
//
//  //---------------------------------------------------------------
//  // This function is called by Tao to evaluate the inequality constraints at x
//  PetscErrorCode
//  __libmesh_tao_inequality_constraints(Tao /*tao*/, Vec x, Vec cineq, void *ctx)
//  {
//    START_LOG("inequality_constraints()", "TaoOptimizationSolver");
//
//    PetscErrorCode ierr = 0;
//    
//    libmesh_assert(x);
//    libmesh_assert(cineq);
//    libmesh_assert(ctx);
//
//    // ctx should be a pointer to the solver (it was passed in as void*)
//    TaoOptimizationSolver<Number>* solver =
//      static_cast<TaoOptimizationSolver<Number>*> (ctx);
//
//    OptimizationSystem &sys = solver->system();
//
//    // We'll use current_local_solution below, so let's ensure that it's consistent
//    // with the vector x that was passed in.
//    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
//    PetscVector<Number> X(x, sys.comm());
//
//    // Perform a swap so that sys.solution points to X
//    X.swap(X_sys);
//    // Impose constraints on X
//    sys.get_dof_map().enforce_constraints_exactly(sys);
//    // Update sys.current_local_solution based on X
//    sys.update();
//    // Swap back
//    X.swap(X_sys);
//
//    // We'll also pass the constraints vector ce into the assembly routine
//    // so let's make a PETSc vector for that too.
//    PetscVector<Number> ineq_constraints(cineq, sys.comm());
//
//    // Clear the gradient prior to assembly
//    ineq_constraints.zero();
//
//    if (solver->inequality_constraints_object != NULL)
//    {
//      solver->inequality_constraints_object->inequality_constraints(
//        *(sys.current_local_solution), ineq_constraints, sys);
//    }
//    else
//    {
//      libmesh_error_msg("Constraints function not defined in __libmesh_tao_inequality_constraints");
//    }
//
//    ineq_constraints.close();
//
//    STOP_LOG("inequality_constraints()", "TaoOptimizationSolver");
//
//    return ierr;
//  }
//
//  //---------------------------------------------------------------
//  // This function is called by Tao to evaluate the Jacobian of the
//  // equality constraints at x
//  PetscErrorCode
//  __libmesh_tao_inequality_constraints_jacobian(Tao /*tao*/, Vec x, Mat J, Mat Jpre, void *ctx)
//  {
//    START_LOG("inequality_constraints_jacobian()", "TaoOptimizationSolver");
//
//    PetscErrorCode ierr = 0;
//    
//    libmesh_assert(x);
//    libmesh_assert(J);
//    libmesh_assert(Jpre);
//
//    // ctx should be a pointer to the solver (it was passed in as void*)
//    TaoOptimizationSolver<Number>* solver =
//      static_cast<TaoOptimizationSolver<Number>*> (ctx);
//
//    OptimizationSystem &sys = solver->system();
//
//    // We'll use current_local_solution below, so let's ensure that it's consistent
//    // with the vector x that was passed in.
//    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
//    PetscVector<Number> X(x, sys.comm());
//
//    // Perform a swap so that sys.solution points to X
//    X.swap(X_sys);
//    // Impose constraints on X
//    sys.get_dof_map().enforce_constraints_exactly(sys);
//    // Update sys.current_local_solution based on X
//    sys.update();
//    // Swap back
//    X.swap(X_sys);
//
//    // Let's also wrap J and Jpre in PetscMatrix objects for convenience
//    PetscMatrix<Number> J_petsc(J, sys.comm());
//    PetscMatrix<Number> Jpre_petsc(Jpre, sys.comm());
//
//    if (solver->inequality_constraints_jacobian_object != NULL)
//    {
//      solver->inequality_constraints_jacobian_object->inequality_constraints_jacobian(
//        *(sys.current_local_solution), J_petsc, sys);
//    }
//    else
//    {
//      libmesh_error_msg("Constraints function not defined in __libmesh_tao_inequality_constraints_jacobian");
//    }
//
//    J_petsc.close();
//    Jpre_petsc.close();
//
//    STOP_LOG("inequality_constraints_jacobian()", "TaoOptimizationSolver");
//
//    return ierr;
//  }
//
//---------------------------------------------------------------------



//---------------------------------------------------------------------
// NloptOptimizationSolver<> methods
template <typename T>
NloptOptimizationSolver<T>::NloptOptimizationSolver (OptimizationSystem& system_in)
  :
  OptimizationSolver<T>(system_in),
  _nlopt(NULL),
  _result(nlopt::SUCCESS/*==0*/) // Arbitrary initial value...
{
}



template <typename T>
NloptOptimizationSolver<T>::~NloptOptimizationSolver ()
{
  this->clear ();
}



template <typename T>
void NloptOptimizationSolver<T>::clear ()
{
  if (this->initialized())
  {
    this->_is_initialized = false;

    _nlopt.reset(NULL);
  }
}



template <typename T>
void NloptOptimizationSolver<T>::init ()
{
  // Initialize the data structures if not done so already.
  if (!this->initialized())
  {
    this->_is_initialized = true;

    _nlopt.reset(
      new nlopt::opt(nlopt::LD_SLSQP, this->system().solution->size()) );
  }
}

template <typename T>
void NloptOptimizationSolver<T>::solve ()
{
  START_LOG("solve()", "NloptOptimizationSolver");

  this->init ();

  unsigned int nlopt_size = this->system().solution->size();

  // We have to have an objective function
  libmesh_assert( this->objective_object );

  // Set routines for objective, and (optionally) gradient evaluation
  _nlopt->set_min_objective(__libmesh_nlopt_objective, this);

  if ( this->lower_and_upper_bounds_object )
  {
    // Need to actually compute the bounds vectors first
    this->lower_and_upper_bounds_object->lower_and_upper_bounds(this->system());

    std::vector<Real> nlopt_lb(nlopt_size);
    std::vector<Real> nlopt_ub(nlopt_size);
    for(unsigned int i=0; i<nlopt_size; i++)
    {
      nlopt_lb[i] = this->system().get_vector("lower_bounds")(i);
      nlopt_ub[i] = this->system().get_vector("upper_bounds")(i);
    }

    _nlopt->set_lower_bounds(nlopt_lb);
    _nlopt->set_upper_bounds(nlopt_ub);
  }

//  if ( this->equality_constraints_object )
//  {
//    ierr = TaoSetEqualityConstraintsRoutine(_tao, ceq->vec(), __libmesh_tao_equality_constraints, this);
//    LIBMESH_CHKERRABORT(ierr);
//  }
//
//  // Optionally set inequality constraints
//  if ( this->inequality_constraints_object )
//  {
//    ierr = TaoSetInequalityConstraintsRoutine(_tao, cineq->vec(), __libmesh_tao_inequality_constraints, this);
//    LIBMESH_CHKERRABORT(ierr);
//  }

  // Perform the optimization
  std::vector<Real> x(nlopt_size);
  Real min_val = 0.;
  _result = _nlopt->optimize(x, min_val);

  STOP_LOG("solve()", "NloptOptimizationSolver");
}


template <typename T>
void NloptOptimizationSolver<T>::print_converged_reason()
{
  libMesh::out << "NLopt optimization solver convergence/divergence reason: "
               << this->get_converged_reason() << std::endl;
}



template <typename T>
nlopt::result NloptOptimizationSolver<T>::get_converged_reason()
{
  if (this->initialized())
    {
      _result = _nlopt->last_optimize_result();
    }

  return _result;
}


//------------------------------------------------------------------
// Explicit instantiations
template class NloptOptimizationSolver<Number>;

} // namespace libMesh



#endif // #if defined(LIBMESH_HAVE_NLOPT) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)
