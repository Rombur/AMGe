// Do the comparison with step-40

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>


#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <iostream>
#include <random>

namespace dealii
{
  namespace LA = LinearAlgebra;
}

/**************************************
 * AMGe preconditioner                *
 **************************************/

// First, we want to do a two-level grid. We need to create non-overlapping
// patches, do a local assembly, compute local eigenvectors, create the
// restriction/prolongation matrices, build and invert the coarse matrix. To
// compute the eigenvalues, first we need to have an approximation of the largest
// eigenvalue. We then substract this eigenvalue to the diagonal of the 
// patch_system_matrix. Be careful not to touch the constrained entries. Then,
// use Lanczos to compute the eigenvalues and more importantly the eigenvectors.
template <int dim, typename VectorType>
class AMGe
{
  public:
    AMGe(unsigned int fe_degree, int n_cells_per_patch, 
         std::shared_ptr<dealii::DoFHandler<dim>> dof_handler);
    
    void vmult(VectorType &dst, VectorType const &src) const;

    void Tvmult(VectorType &dst, VectorType const &src) const;

  private:
    void initialize(int const n_cells_per_patch);

    void build_patch_triangulation(dealii::types::subdomain_id const id,
                                   dealii::Triangulation<dim> &patch_triangulation,
                                   std::map<typename 
                                     dealii::Triangulation<dim>::active_cell_iterator, 
                                            typename
                                     dealii::DoFHandler<dim>::active_cell_iterator> 
                                       &path_to_global_tria_map);

    void setup_on_patch(dealii::DoFHandler<dim> &patch_dof_handler,       
                        dealii::ConstraintMatrix &patch_constraints,      
                        dealii::SparsityPattern &patch_sparsity_pattern,  
                        dealii::SparseMatrix<double> &patch_system_matrix);

    void assemble_system_on_patch(dealii::DoFHandler<dim> &patch_dof_handler,      
                                  dealii::ConstraintMatrix &patch_constraints,     
                                  dealii::SparseMatrix<double> &patch_system_matrix);

    void compute_largest_eigenvalue(dealii::SparseMatrix<double> const &patch_system_matrix,
                                    double & largest_eigenvalue);

    void shift_eigenvalues(double largest_eigenvalue,
                           dealii::ConstraintMatrix const &patch_constraints,
                           dealii::SparseMatrix<double> &patch_system_matrix);

    std::shared_ptr<dealii::DoFHandler<dim>> _dof_handler;
    unsigned int _fe_degree;
    dealii::FE_Q<dim> _fe;
};


template <int dim, typename VectorType>
AMGe<dim,VectorType>::AMGe(unsigned int fe_degree, int n_cells_per_patch,
                           std::shared_ptr<dealii::DoFHandler<dim>> dof_handler)
  :
    _dof_handler(dof_handler),
    _fe_degree(fe_degree),
    _fe(_fe_degree)
{
  initialize(n_cells_per_patch);
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::vmult(VectorType &dst, VectorType const &src) const
{
  dst = src;
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::Tvmult(VectorType &dst, VectorType const &src) const
{
  throw std::runtime_error("Not implemented");
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::initialize(int const n_cells_per_patch)
{
  // Create patches of contiguous cells using Metis. Different patches have
  // different subdomain_id.
  unsigned int const n_partitions = 
    _dof_handler->get_triangulation().n_active_cells() / n_cells_per_patch;
  dealii::Triangulation<dim> tmp_triangulation;
  //TODO this will be slow
  tmp_triangulation.copy_triangulation(_dof_handler->get_triangulation());
  dealii::GridTools::partition_triangulation(n_partitions, tmp_triangulation);
  // Copy the partitioning
  auto dof_cell = _dof_handler->begin_active();
  auto end_dof_cell = _dof_handler->end();
  auto tmp_tria_cell = tmp_triangulation.begin_active();
  for (; dof_cell != end_dof_cell; ++dof_cell, ++tmp_tria_cell)
    dof_cell->set_subdomain_id(tmp_tria_cell->subdomain_id());

  for (unsigned int i=0; i<n_partitions; ++i)
  {
    // Create the local Triangulation
    dealii::Triangulation<dim> patch_triangulation;
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator, 
      typename dealii::DoFHandler<dim>::active_cell_iterator> path_to_global_tria_map;
    build_patch_triangulation(static_cast<dealii::types::subdomain_id>(i), 
                              patch_triangulation, path_to_global_tria_map);

    // Setup on a patch
    dealii::DoFHandler<dim> patch_dof_handler(patch_triangulation);
    dealii::ConstraintMatrix patch_constraints;
    dealii::SparsityPattern patch_sparsity_pattern;
    dealii::SparseMatrix<double> patch_system_matrix;
    setup_on_patch(patch_dof_handler, patch_constraints, patch_sparsity_pattern,
                   patch_system_matrix);

    // Assembly on a patch.
    assemble_system_on_patch(patch_dof_handler, patch_constraints, 
                             patch_system_matrix);

    // Compute largest eigenvalue.
    double largest_eigenvalue = 0.;
    compute_largest_eigenvalue(patch_system_matrix, largest_eigenvalue);

    // Shift the eigenvalues so that the smallest ones becomes the largest ones
    // (in magnitude).
    shift_eigenvalues(largest_eigenvalue, patch_constraints, patch_system_matrix);
  }
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::build_patch_triangulation(dealii::types::subdomain_id const id,
                                                     dealii::Triangulation<dim> &patch_triangulation,
                                                     std::map<typename
                                                       dealii::Triangulation<dim>::active_cell_iterator, 
                                                              typename
                                                       dealii::DoFHandler<dim>::active_cell_iterator> 
                                                     &path_to_global_tria_map)
{
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> patch;
  for (auto cell : _dof_handler->active_cell_iterators())
  {
    if (cell->subdomain_id() == id)
      patch.push_back(cell);
  }
  dealii::GridTools::build_triangulation_from_patch<dealii::DoFHandler<dim>>(
    patch, patch_triangulation, path_to_global_tria_map);
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::setup_on_patch(dealii::DoFHandler<dim> &patch_dof_handler,
                                          dealii::ConstraintMatrix &patch_constraints,
                                          dealii::SparsityPattern &patch_sparsity_pattern,
                                          dealii::SparseMatrix<double> &patch_system_matrix)
{
  patch_dof_handler.distribute_dofs(_dof_handler->get_fe());

  unsigned int const n_dofs = patch_dof_handler.n_dofs();

  patch_constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(patch_dof_handler, patch_constraints);
  patch_constraints.close();

  dealii::DynamicSparsityPattern dsp(n_dofs);

  dealii::DoFTools::make_sparsity_pattern(patch_dof_handler, dsp, 
                                          patch_constraints, false);

  patch_sparsity_pattern.copy_from(dsp);
  patch_system_matrix.reinit(patch_sparsity_pattern);
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::assemble_system_on_patch(dealii::DoFHandler<dim> &patch_dof_handler,
                                                    dealii::ConstraintMatrix &patch_constraints,
                                                    dealii::SparseMatrix<double> &patch_system_matrix)
{
  dealii::QGauss<dim> const quadrature_formula(_fe_degree+1);

  dealii::FEValues<dim> fe_values (_fe, quadrature_formula,
                                   dealii::update_values | 
                                   dealii::update_gradients |
                                   dealii::update_quadrature_points |
                                   dealii::update_JxW_values);

  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  unsigned int const n_q_points = quadrature_formula.size();

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell : patch_dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;

    fe_values.reinit(cell);

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                               fe_values.shape_grad(j,q_point) *
                               fe_values.JxW(q_point));


    cell->get_dof_indices(local_dof_indices);

    // Diagonal entries corresponding to eliminated degrees of freedom are not
    // set, the result have a zero eigenvalue. For solving a source problem
    // Au=f, it is possible to set the diagonal entry after building the matrix
    // for (unsigned int i=0; i<matrix.m(); ++i)
    //   if (constraints.is_constrained(i))
    //     matrix.diag_element(i) = 1.;
    // This will will add one spurious zero eigenvalue.
    patch_constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 local_dof_indices,
                                                 patch_system_matrix);
  }
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::compute_largest_eigenvalue(
  dealii::SparseMatrix<double> const &patch_system_matrix,
  double &largest_eigenvalue)
{
  std::default_random_engine generator(0);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  unsigned int const size = patch_system_matrix.m();
  VectorType eigen_vector(size);
  for (unsigned int i=0; i<size; ++i)
    eigen_vector[i] = distribution(generator);

  dealii::SolverControl solver_control(10, 0.1);
  dealii::GrowingVectorMemory<VectorType> vector_memory;
  dealii::EigenPower<VectorType> eigen_power(solver_control, vector_memory);
  eigen_power.solve(largest_eigenvalue, patch_system_matrix, eigen_vector);
}


template <int dim, typename VectorType>
void AMGe<dim,VectorType>::shift_eigenvalues(double largest_eigenvalue,
                                             dealii::ConstraintMatrix const &patch_constraints,
                                             dealii::SparseMatrix<double> &patch_system_matrix)
{
  unsigned int const size = patch_system_matrix.m();
  for (unsigned int i=0; i<size; ++i)
    if (!patch_constraints.is_constrained(i))
      patch_system_matrix.diag_element(i) -= largest_eigenvalue;
}


/**************************************
 * Laplace solver                     *
 **************************************/

template <int dim, typename VectorType>
class Laplace
{
  public:
    Laplace(unsigned int fe_degree);

    void run(std::string const &preconditioner_type);

  private:
    void setup_system ();

    void assemble_system ();

    void solve(std::string const &preconditioner_type);

    void refine_grid ();

    void output_results (const unsigned int cycle) const;

    unsigned int _fe_degree;
    dealii::Triangulation<dim> _triangulation;
    dealii::FE_Q<dim> _fe;
    std::shared_ptr<dealii::DoFHandler<dim>> _dof_handler;
    dealii::ConstraintMatrix _constraints;
    dealii::SparsityPattern _sparsity_pattern;
    dealii::SparseMatrix<double> _system_matrix;
    VectorType _solution;
    VectorType _system_rhs;
    dealii::TimerOutput _computing_timer;
};


template <int dim, typename VectorType>
Laplace<dim, VectorType>::Laplace(unsigned int fe_degree)
  :
    _fe_degree(fe_degree),
    _fe(_fe_degree),
    _dof_handler(std::make_shared<dealii::DoFHandler<dim>>(_triangulation)),
    _computing_timer(std::cout,
                     dealii::TimerOutput::summary,
                     dealii::TimerOutput::wall_times)
{}


template <int dim, typename VectorType>
void Laplace<dim, VectorType>::setup_system()
{
  dealii::TimerOutput::Scope t(_computing_timer, "setup");

  _dof_handler->distribute_dofs(_fe);

  unsigned int const n_dofs = _dof_handler->n_dofs();
  _solution.reinit(n_dofs);
  _system_rhs.reinit(n_dofs);

  _constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(*_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(*_dof_handler,
                                           0,
                                           dealii::ZeroFunction<dim>(),
                                           _constraints);
  _constraints.close();

  dealii::DynamicSparsityPattern dsp(n_dofs);

  dealii::DoFTools::make_sparsity_pattern(*_dof_handler, dsp,
                                  _constraints, false);

  _sparsity_pattern.copy_from(dsp);
  _system_matrix.reinit(_sparsity_pattern);
}


template <int dim, typename VectorType>
void Laplace<dim, VectorType>::assemble_system()
{
  dealii::TimerOutput::Scope t(_computing_timer, "assembly");

  dealii::QGauss<dim> const quadrature_formula(_fe_degree+1);

  dealii::FEValues<dim> fe_values (_fe, quadrature_formula,
                                   dealii::update_values | 
                                   dealii::update_gradients |
                                   dealii::update_quadrature_points |
                                   dealii::update_JxW_values);

  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  unsigned int const n_q_points = quadrature_formula.size();

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  VectorType cell_rhs(dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell : _dof_handler->active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      double const rhs_value = (fe_values.quadrature_point(q_point)[0] > 0.5 
                                ? 1 : -1);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                               fe_values.shape_grad(j,q_point) *
                               fe_values.JxW(q_point));

        cell_rhs(i) += (rhs_value *
                        fe_values.shape_value(i,q_point) *
                        fe_values.JxW(q_point));
      }
    }

    cell->get_dof_indices(local_dof_indices);
    _constraints.distribute_local_to_global(cell_matrix,
                                            cell_rhs,
                                            local_dof_indices,
                                            _system_matrix,
                                            _system_rhs);
  }
}


template <int dim, typename VectorType>
void Laplace<dim, VectorType>::solve(std::string const &preconditioner_type)
{
  dealii::TimerOutput::Scope t(_computing_timer, "solve");

  dealii::SolverControl solver_control(_dof_handler->n_dofs(), 1e-12);

  dealii::SolverCG<VectorType> solver(solver_control);

  if (preconditioner_type == "identity")
  {
    dealii::PreconditionIdentity preconditioner;

    solver.solve(_system_matrix, _solution, _system_rhs,
                 preconditioner);
  }
  else if (preconditioner_type == "AMGe")
  {
    unsigned int n_cells_per_patch = 2;
    AMGe<dim, VectorType> preconditioner(_fe_degree, n_cells_per_patch,
                                         _dof_handler);

    solver.solve(_system_matrix, _solution, _system_rhs,
                 preconditioner);
  }
  else
    throw std::runtime_error("Unknown preconditioner.");

  std::cout << "   Solved in " << solver_control.last_step()
    << " iterations." << std::endl;

  _constraints.distribute(_solution);
}


template <int dim, typename VectorType>
void Laplace<dim, VectorType>::refine_grid ()
{
  dealii::TimerOutput::Scope t(_computing_timer, "refine");

  dealii::Vector<float> estimated_error_per_cell(_triangulation.n_active_cells());
  dealii::KellyErrorEstimator<dim>::estimate(*_dof_handler,
                                     dealii::QGauss<dim-1>(3),
                                     typename dealii::FunctionMap<dim>::type(),
                                     _solution,
                                     estimated_error_per_cell);
  dealii::GridRefinement::
    refine_and_coarsen_fixed_number(_triangulation,
                                    estimated_error_per_cell,
                                    0.3, 0.03);
  _triangulation.execute_coarsening_and_refinement ();
}



template <int dim, typename VectorType>
void Laplace<dim, VectorType>::output_results (const unsigned int cycle) const
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(*_dof_handler);
  data_out.add_data_vector(_solution, "solution");

  data_out.build_patches ();

  std::string const filename = "solution-" +
    dealii::Utilities::int_to_string (cycle, 2);
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);
}


template <int dim, typename VectorType>
void Laplace<dim, VectorType>::run(std::string const &preconditioner_type)
{
  unsigned int const n_cycles = 6;
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      dealii::GridGenerator::hyper_cube(_triangulation);
      _triangulation.refine_global(5);
    }
    else
      refine_grid();

    setup_system();

    std::cout << "   Number of active cells:       "
      << _triangulation.n_global_active_cells()
      << std::endl
      << "   Number of degrees of freedom: "
      << _dof_handler->n_dofs()
      << std::endl;

    assemble_system ();
    solve(preconditioner_type);
    output_results (cycle);

    _computing_timer.print_summary ();
    _computing_timer.reset ();

    std::cout << std::endl;
  }
}



int main()
{
  try
    {
      unsigned int const fe_degree = 2;
      std::string preconditioner = "AMGe";
      Laplace<2, dealii::Vector<double>> laplace(fe_degree);
      laplace.run(preconditioner);
    }
  catch (std::exception &exception)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "Error: " << exception.what() << std::endl;
    std::cerr << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "No error message." << std::endl;
    std::cerr << std::endl;

    return 1;
  }

  return 0;
}
