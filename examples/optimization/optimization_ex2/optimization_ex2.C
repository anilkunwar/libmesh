/* The libMesh Finite Element Library. */
/* Copyright (C) 2003  Benjamin S. Kirk */

/* This library is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU Lesser General Public */
/* License as published by the Free Software Foundation; either */
/* version 2.1 of the License, or (at your option) any later version. */

/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU */
/* Lesser General Public License for more details. */

/* You should have received a copy of the GNU Lesser General Public */
/* License along with this library; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

// <h1>Optimization Example 2 - Optimization with equality constraints</h1>
//
// In this example we extend example 1 to demonstrate how to use OptimizationSystem's
// interface for imposing equality constraints. Note that we consider the same
// quadratic objective function in this case, but we do not impose any Dirichlet
// boundary conditions (we impose equality constraints instead).

// C++ include files that we need
#include <iostream>

// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/boundary_info.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/getpot.h"
#include "libmesh/zero_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/optimization_system.h"
#include "libmesh/optimization_solver.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

/**
 * This class encapsulate all functionality required for assembling
 * and solving a linear elastic model with contact.
 */
class AssembleOptimization :
  public OptimizationSystem::ComputeObjective,
  public OptimizationSystem::ComputeGradient,
  public OptimizationSystem::ComputeHessian,
  public OptimizationSystem::ComputeEqualityConstraints,
  public OptimizationSystem::ComputeEqualityConstraintsJacobian
{
private:

  /**
   * Keep a reference to the OptimizationSystem.
   */
  OptimizationSystem& _sys;

public:

  /**
   * Constructor.
   */
  AssembleOptimization(
    OptimizationSystem &sys_in);

  /**
   * The optimization problem we consider here is:
   *   min_U 0.5*U^T A U - U^T F.
   * In this method, we assemble A and F.
   */
  void assemble_A_and_F();

  /**
   * Evaluate the objective function.
   */
  virtual Number objective (
    const NumericVector<Number>& soln,
    OptimizationSystem& /*sys*/);

  /**
   * Evaluate the gradient.
   */
  virtual void gradient (
    const NumericVector<Number>& soln,
    NumericVector<Number>& grad_f,
    OptimizationSystem& /*sys*/);

  /**
   * Evaluate the Hessian.
   */
  virtual void hessian (
    const NumericVector<Number>& soln,
    SparseMatrix<Number>& H_f,
    OptimizationSystem& /*sys*/);

  /**
   * Evaluate the equality constraints.
   */
  virtual void equality_constraints (const NumericVector<Number>& X,
                                     NumericVector<Number>& C_eq,
                                     OptimizationSystem& /*sys*/);

  /**
   * Evaluate the equality constraints Jacobian.
   */
  virtual void equality_constraints_jacobian (const NumericVector<Number>& X,
                                              SparseMatrix<Number>& C_eq_jac,
                                              OptimizationSystem& /*sys*/);

  /**
   * Sparse matrix for storing the matrix A. We use
   * this to facilitate computation of objective, gradient
   * and hessian.
   */
  SparseMatrix<Number>* A_matrix;

  /**
   * Vector for storing F. We use this to facilitate 
   * computation of objective, gradient and hessian.
   */
  NumericVector<Number>* F_vector;

};

AssembleOptimization::AssembleOptimization(OptimizationSystem &sys_in)
  :
  _sys(sys_in)
{}

void AssembleOptimization::assemble_A_and_F()
{
  A_matrix->zero();
  F_vector->zero();

  const MeshBase& mesh = _sys.get_mesh();

  const unsigned int dim = mesh.mesh_dimension();
  const unsigned int u_var = _sys.variable_number ("u");

  const DofMap& dof_map = _sys.get_dof_map();
  FEType fe_type = dof_map.variable_type(u_var);
  UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
  QGauss qrule (dim, fe_type.default_quadrature_order());
  fe->attach_quadrature_rule (&qrule);

  const std::vector<Real>& JxW = fe->get_JxW();
  const std::vector<std::vector<Real> >& phi = fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();

  std::vector<dof_id_type> dof_indices;

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el; ++el)
  {
    const Elem* elem = *el;

    dof_map.dof_indices (elem, dof_indices);

    const unsigned int n_dofs = dof_indices.size();

    fe->reinit (elem);

    Ke.resize (n_dofs,n_dofs);
    Fe.resize (n_dofs);

    for (unsigned int qp=0; qp<qrule.n_points(); qp++)
    {
      for (unsigned int dof_i=0; dof_i<n_dofs; dof_i++)
      {
        for (unsigned int dof_j=0; dof_j<n_dofs; dof_j++)
        {
          Ke(dof_i, dof_j) += JxW[qp] * (dphi[dof_j][qp]* dphi[dof_i][qp]);
        }
        Fe(dof_i) += JxW[qp] * phi[dof_i][qp];
      }
    }

    A_matrix->add_matrix (Ke, dof_indices);
    F_vector->add_vector (Fe, dof_indices);
  }

  A_matrix->close();
  F_vector->close();
}

Number AssembleOptimization::objective (
  const NumericVector<Number>& soln,
  OptimizationSystem& /*sys*/)
{
  UniquePtr< NumericVector<Number> > AxU = soln.zero_clone();

  A_matrix->vector_mult(*AxU, soln);
  Real UTxAxU = AxU->dot(soln);

  Real UTxF = F_vector->dot(soln);

  return 0.5 * UTxAxU - UTxF;
}

void AssembleOptimization::gradient (
  const NumericVector<Number>& soln,
  NumericVector<Number>& grad_f,
  OptimizationSystem& /*sys*/)
{
  grad_f.zero();

  A_matrix->vector_mult(grad_f, soln);
  grad_f.add(-1, *F_vector);
}


void AssembleOptimization::hessian (
  const NumericVector<Number>& /*soln*/,
  SparseMatrix<Number>& H_f,
  OptimizationSystem& /*sys*/)
{
  H_f.zero();
  H_f.add(1., *A_matrix);
}

void AssembleOptimization::equality_constraints (
  const NumericVector<Number>& X,
  NumericVector<Number>& C_eq,
  OptimizationSystem& /*sys*/)
{
  C_eq.zero();

  C_eq.set(0, X(17));
  C_eq.set(1, X(23));
  C_eq.set(2, X(98) + X(185));
}

void AssembleOptimization::equality_constraints_jacobian (
  const NumericVector<Number>& X,
  SparseMatrix<Number>& C_eq_jac,
  OptimizationSystem& /*sys*/)
{
  C_eq_jac.zero();

  C_eq_jac.set(0, 17, 1.);
  C_eq_jac.set(1, 23, 1.);
  C_eq_jac.set(2, 98, 1.);
  C_eq_jac.set(2, 185, 1.);
}


int main (int argc, char** argv)
{
  LibMeshInit init (argc, argv);

#ifndef LIBMESH_HAVE_PETSC_TAO
  libmesh_example_requires(false, "PETSc >= 3.5.0 with built-in TAO support");
#endif

  GetPot infile("optimization_ex2.in");
  const std::string approx_order = infile("approx_order", "FIRST");
  const std::string fe_family = infile("fe_family", "LAGRANGE");
  const unsigned int n_elem = infile("n_elem", 10);

  Mesh mesh(init.comm());
  MeshTools::Generation::build_square (mesh,
                                       n_elem,
                                       n_elem,
                                       -1., 1.,
                                       -1., 1.,
                                       QUAD9);

  mesh.print_info();

  EquationSystems equation_systems (mesh);

  OptimizationSystem& system =
    equation_systems.add_system<OptimizationSystem> ("Optimization");

  AssembleOptimization assemble_opt(system);

  unsigned int u_var = system.add_variable("u",
                       Utility::string_to_enum<Order>   (approx_order),
                       Utility::string_to_enum<FEFamily>(fe_family));

  system.optimization_solver->objective_object = &assemble_opt;
  system.optimization_solver->gradient_object = &assemble_opt;
  system.optimization_solver->hessian_object = &assemble_opt;
  system.optimization_solver->equality_constraints_object = &assemble_opt;
  system.optimization_solver->equality_constraints_jacobian_object = &assemble_opt;

  // system.matrix and system.rhs are used for the gradient and Hessian,
  // so in this case we add an extra matrix and vector to store A and F.
  // This makes it easy to write the code for evaluating the objective,
  // gradient, and hessian.
  system.add_matrix("A_matrix");
  system.add_vector("F_vector");
  assemble_opt.A_matrix = &system.get_matrix("A_matrix");
  assemble_opt.F_vector = &system.get_vector("F_vector");


  equation_systems.init();
  equation_systems.print_info();

  assemble_opt.assemble_A_and_F();

  std::vector<unsigned int> n_dofs_per_constraint;
  n_dofs_per_constraint.push_back(1);
  n_dofs_per_constraint.push_back(1);
  n_dofs_per_constraint.push_back(2);
  system.initialize_equality_constraints_storage(n_dofs_per_constraint);

  // We need to close the matrix so that we can use it to store the
  // Hessian during the solve.
  system.matrix->close();
  system.solve();
  
  // Print convergence information
  system.optimization_solver->print_converged_reason();

  std::stringstream filename;
  ExodusII_IO (mesh).write_equation_systems(
    "optimization_soln.exo",
    equation_systems);

  return 0;
}

