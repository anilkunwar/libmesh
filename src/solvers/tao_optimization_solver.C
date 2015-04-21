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

#if defined(LIBMESH_HAVE_PETSC) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)


// C++ includes

// Local Includes
#include "libmesh/petsc_vector.h"
#include "libmesh/petsc_matrix.h"
#include "libmesh/dof_map.h"
#include "libmesh/tao_optimization_solver.h"

namespace libMesh
{

//--------------------------------------------------------------------
// Functions with C linkage to pass to Tao. Tao will call these
// methods as needed.
//
// Since they must have C linkage they have no knowledge of a namespace.
// Give them an obscure name to avoid namespace pollution.
extern "C"
{

  //---------------------------------------------------------------
  // This function is called by Tao to evaluate objective function at x
  PetscErrorCode
  __libmesh_tao_objective (Tao /*tao*/, Vec x, PetscReal* objective, void *ctx)
  {
    START_LOG("objective()", "TaoOptimizationSolver");

    PetscErrorCode ierr = 0;

    libmesh_assert(x);
    libmesh_assert(objective);
    libmesh_assert(ctx);

    // ctx should be a pointer to the solver (it was passed in as void*)
    TaoOptimizationSolver<Number>* solver =
      static_cast<TaoOptimizationSolver<Number>*> (ctx);

    OptimizationSystem &sys = solver->system();

    // We'll use current_local_solution below, so let's ensure that it's consistent
    // with the vector x that was passed in.
    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
    PetscVector<Number> X(x, sys.comm());

    // Perform a swap so that sys.solution points to X
    X.swap(X_sys);
    // Impose constraints on X
    sys.get_dof_map().enforce_constraints_exactly(sys);
    // Update sys.current_local_solution based on X
    sys.update();
    // Swap back
    X.swap(X_sys);

    if (solver->objective_object != NULL)
    {
      (*objective) =
        solver->objective_object->objective(
          *(sys.current_local_solution), sys);
    }
    else
    {
      libmesh_error_msg("Objective function not defined in __libmesh_tao_objective");
    }

    STOP_LOG("objective()", "TaoOptimizationSolver");

    return ierr;
  }



  //---------------------------------------------------------------
  // This function is called by Tao to evaluate the gradient at x
  PetscErrorCode
  __libmesh_tao_gradient(Tao /*tao*/, Vec x, Vec g, void *ctx)
  {
    START_LOG("gradient()", "TaoOptimizationSolver");

    PetscErrorCode ierr = 0;
    
    libmesh_assert(x);
    libmesh_assert(g);
    libmesh_assert(ctx);

    // ctx should be a pointer to the solver (it was passed in as void*)
    TaoOptimizationSolver<Number>* solver =
      static_cast<TaoOptimizationSolver<Number>*> (ctx);

    OptimizationSystem &sys = solver->system();

    // We'll use current_local_solution below, so let's ensure that it's consistent
    // with the vector x that was passed in.
    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
    PetscVector<Number> X(x, sys.comm());

    // Perform a swap so that sys.solution points to X
    X.swap(X_sys);
    // Impose constraints on X
    sys.get_dof_map().enforce_constraints_exactly(sys);
    // Update sys.current_local_solution based on X
    sys.update();
    // Swap back
    X.swap(X_sys);

    // We'll also pass the gradient in to the assembly routine
    // so let's make a PETSc vector for that too.
    PetscVector<Number> gradient(g, sys.comm());

    // Clear the gradient prior to assembly
    gradient.zero();

    if (solver->gradient_object != NULL)
    {
      solver->gradient_object->gradient(
        *(sys.current_local_solution), gradient, sys);
    }
    else
    {
      libmesh_error_msg("Gradient function not defined in __libmesh_tao_gradient");
    }

    gradient.close();

    STOP_LOG("gradient()", "TaoOptimizationSolver");

    return ierr;
  }

  //---------------------------------------------------------------
  // This function is called by Tao to evaluate the Hessian at x
  PetscErrorCode
  __libmesh_tao_hessian(Tao /*tao*/, Vec x, Mat h, Mat pc, void *ctx)
  {
    START_LOG("hessian()", "TaoOptimizationSolver");

    PetscErrorCode ierr = 0;
    
    libmesh_assert(x);
    libmesh_assert(h);
    libmesh_assert(pc);
    libmesh_assert(ctx);

    // ctx should be a pointer to the solver (it was passed in as void*)
    TaoOptimizationSolver<Number>* solver =
      static_cast<TaoOptimizationSolver<Number>*> (ctx);

    OptimizationSystem &sys = solver->system();

    // We'll use current_local_solution below, so let's ensure that it's consistent
    // with the vector x that was passed in.
    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
    PetscVector<Number> X(x, sys.comm());

    // Perform a swap so that sys.solution points to X
    X.swap(X_sys);
    // Impose constraints on X
    sys.get_dof_map().enforce_constraints_exactly(sys);
    // Update sys.current_local_solution based on X
    sys.update();
    // Swap back
    X.swap(X_sys);

    // Let's also set up matrices that will be
    PetscMatrix<Number> PC(pc, sys.comm());
    PetscMatrix<Number> hessian(h, sys.comm());
    PC.attach_dof_map(sys.get_dof_map());
    hessian.attach_dof_map(sys.get_dof_map());

    if (solver->hessian_object != NULL)
    {
      // Following PetscNonlinearSolver by passing in PC. It's not clear
      // why we pass in PC and not hessian though?
      solver->hessian_object->hessian(
        *(sys.current_local_solution), PC, sys);
    }
    else
    {
      libmesh_error_msg("Hessian function not defined in __libmesh_tao_hessian");
    }

    PC.close();
    hessian.close();

    STOP_LOG("hessian()", "TaoOptimizationSolver");

    return ierr;
  }


  //---------------------------------------------------------------
  // This function is called by Tao to evaluate the equality constraints at x
  PetscErrorCode
  __libmesh_tao_equality_constraints(Tao /*tao*/, Vec x, Vec ce, void *ctx)
  {
    START_LOG("equality_constraints()", "TaoOptimizationSolver");

    PetscErrorCode ierr = 0;
    
    libmesh_assert(x);
    libmesh_assert(ce);
    libmesh_assert(ctx);

    // ctx should be a pointer to the solver (it was passed in as void*)
    TaoOptimizationSolver<Number>* solver =
      static_cast<TaoOptimizationSolver<Number>*> (ctx);

    OptimizationSystem &sys = solver->system();

    // We'll use current_local_solution below, so let's ensure that it's consistent
    // with the vector x that was passed in.
    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(sys.solution.get());
    PetscVector<Number> X(x, sys.comm());

    // Perform a swap so that sys.solution points to X
    X.swap(X_sys);
    // Impose constraints on X
    sys.get_dof_map().enforce_constraints_exactly(sys);
    // Update sys.current_local_solution based on X
    sys.update();
    // Swap back
    X.swap(X_sys);

    // We'll also pass the gradient in to the assembly routine
    // so let's make a PETSc vector for that too.
    PetscVector<Number> eq_constraints(ce, sys.comm());

    // Clear the gradient prior to assembly
    eq_constraints.zero();

    if (solver->equality_constraints != NULL)
    {
      solver->equality_constraints->equality_constraints(
        *(sys.current_local_solution), eq_constraints, sys);
    }
    else
    {
      libmesh_error_msg("Constraints function not defined in __libmesh_tao_equality_constraints");
    }

    eq_constraints.close();

    STOP_LOG("equality_constraints()", "TaoOptimizationSolver");

    return ierr;
  }

} // end extern "C"
//---------------------------------------------------------------------



//---------------------------------------------------------------------
// TaoOptimizationSolver<> methods
template <typename T>
TaoOptimizationSolver<T>::TaoOptimizationSolver (OptimizationSystem& system_in)
  :
  OptimizationSolver<T>(system_in),
  _reason(TAO_CONVERGED_FATOL/*==0*/) // Arbitrary initial value...
{
}



template <typename T>
TaoOptimizationSolver<T>::~TaoOptimizationSolver ()
{
  this->clear ();
}



template <typename T>
void TaoOptimizationSolver<T>::clear ()
{
  if (this->initialized())
  {
    this->_is_initialized = false;

    PetscErrorCode ierr=0;

    ierr = TaoDestroy(&_tao);
    LIBMESH_CHKERRABORT(ierr);
  }
}



template <typename T>
void TaoOptimizationSolver<T>::init ()
{
  // Initialize the data structures if not done so already.
  if (!this->initialized())
  {
    this->_is_initialized = true;

    PetscErrorCode ierr=0;

    ierr = TaoCreate(this->comm().get(),&_tao);
    LIBMESH_CHKERRABORT(ierr);
  }
}

template <typename T>
void TaoOptimizationSolver<T>::solve ()
{
  START_LOG("solve()", "TaoOptimizationSolver");

  this->init ();

  this->system().solution->zero();

  PetscMatrix<T>* hessian  = cast_ptr<PetscMatrix<T>*>(this->system().matrix);
  // PetscVector<T>* gradient = cast_ptr<PetscVector<T>*>(this->system().rhs);
  PetscVector<T>* x        = cast_ptr<PetscVector<T>*>(this->system().solution.get());
  PetscVector<T>* ce       = cast_ptr<PetscVector<T>*>(this->system().C_eq.get());

  // Set the starting guess to zero.
  x->zero();

  PetscErrorCode ierr = 0;

  // Set solution vec and an initial guess
  ierr = TaoSetInitialVector(_tao, x->vec());
  LIBMESH_CHKERRABORT(ierr);

  // We have to have an objective function
  libmesh_assert( this->objective_object );

  // Set routines for objective, gradient, hessian evaluation
  ierr = TaoSetObjectiveRoutine(_tao, __libmesh_tao_objective, this);
  LIBMESH_CHKERRABORT(ierr);

  if ( this->gradient_object )
  {
    ierr = TaoSetGradientRoutine(_tao, __libmesh_tao_gradient, this);
    LIBMESH_CHKERRABORT(ierr);
  }

  if ( this->hessian_object )
  {
    ierr = TaoSetHessianRoutine(_tao, hessian->mat(), hessian->mat(), __libmesh_tao_hessian, this);
    LIBMESH_CHKERRABORT(ierr);
  }

  // Optionally set equality constraints
  if ( this->equality_constraints )
  {
    ierr = TaoSetEqualityConstraintsRoutine(_tao, ce->vec(), __libmesh_tao_equality_constraints, this);
    LIBMESH_CHKERRABORT(ierr);
  }

  // Check for Tao command line options
  ierr = TaoSetFromOptions(_tao);
  LIBMESH_CHKERRABORT(ierr);

  // Perform the optimization
  ierr = TaoSolve(_tao);
  LIBMESH_CHKERRABORT(ierr);

  // Store the convergence/divergence reason
  ierr = TaoGetConvergedReason(_tao, &_reason);
  LIBMESH_CHKERRABORT(ierr);

  // Print termination information
  libMesh::out << "Converged reason: " << _reason << std::endl;
  if (_reason <= 0)
  {
    libMesh::out << "Tao failed to converge." << std::endl;
  }
  else
  {
    libMesh::out << "Tao converged." << std::endl;
  }

  STOP_LOG("solve()", "TaoOptimizationSolver");
}



//template <typename T>
//void TaoOptimizationSolver<T>::print_converged_reason()
//{
//
//  libMesh::out << "Tao optimization solver convergence/divergence reason: "
//               << TaoConvergedReasons[this->get_converged_reason()] << std::endl;
//}



template <typename T>
TaoConvergedReason TaoOptimizationSolver<T>::get_converged_reason()
{
  PetscErrorCode ierr=0;

  if (this->initialized())
    {
      ierr = TaoGetConvergedReason(_tao, &_reason);
      LIBMESH_CHKERRABORT(ierr);
    }

  return _reason;
}


//------------------------------------------------------------------
// Explicit instantiations
template class TaoOptimizationSolver<Number>;

} // namespace libMesh



#endif // #if defined(LIBMESH_HAVE_PETSC) && !defined(LIBMESH_USE_COMPLEX_NUMBERS)
