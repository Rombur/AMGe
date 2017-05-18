// Do the comparison with step-40

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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

#include <fstream>
#include <iostream>

namespace dealii
{
  namespace LA = LinearAlgebra;
}

template <int dim>
class Laplace
{
  public:
    Laplace(unsigned int fe_degree);

    void run(std::string const &preconditioner);

  private:
    void setup_system ();

    void assemble_system ();

    void solve(std::string const &preconditioner);

    void refine_grid ();

    void output_results (const unsigned int cycle) const;

    unsigned int _fe_degree;
    dealii::Triangulation<dim> _triangulation;
    dealii::DoFHandler<dim> _dof_handler;
    dealii::FE_Q<dim> _fe;
    dealii::ConstraintMatrix _constraints;
    dealii::SparsityPattern _sparsity_pattern;
    dealii::SparseMatrix<double> _system_matrix;
    dealii::Vector<double> _solution;
    dealii::Vector<double> _system_rhs;
    dealii::TimerOutput _computing_timer;
};


template <int dim>
Laplace<dim>::Laplace(unsigned int fe_degree)
  :
    _fe_degree(fe_degree),
    _dof_handler(_triangulation),
    _fe(_fe_degree),
    _computing_timer(std::cout,
                     dealii::TimerOutput::summary,
                     dealii::TimerOutput::wall_times)
{}


template <int dim>
void Laplace<dim>::setup_system()
{
  dealii::TimerOutput::Scope t(_computing_timer, "setup");

  _dof_handler.distribute_dofs(_fe);

  unsigned int const n_dofs = _dof_handler.n_dofs();
  _solution.reinit(n_dofs);
  _system_rhs.reinit(n_dofs);

  _constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(_dof_handler,
                                           0,
                                           dealii::ZeroFunction<dim>(),
                                           _constraints);
  _constraints.close();

  dealii::DynamicSparsityPattern dsp(n_dofs);

  dealii::DoFTools::make_sparsity_pattern(_dof_handler, dsp,
                                  _constraints, false);

  _sparsity_pattern.copy_from(dsp);
  _system_matrix.reinit(_sparsity_pattern);
}


template <int dim>
void Laplace<dim>::assemble_system()
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
  dealii::Vector<double> cell_rhs(dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell : _dof_handler.active_cell_iterators())
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


template <int dim>
void Laplace<dim>::solve(std::string const &preconditioner)
{
  dealii::TimerOutput::Scope t(_computing_timer, "solve");

  dealii::SolverControl solver_control(_dof_handler.n_dofs(), 1e-12);

  dealii::SolverCG<dealii::Vector<double>> solver(solver_control);

  if (preconditioner == "identity")
  {
    dealii::PreconditionIdentity preconditioner;

    solver.solve(_system_matrix, _solution, _system_rhs,
                 preconditioner);
  }
  else
    throw std::runtime_error("Unknown preconditioner.");

  std::cout << "   Solved in " << solver_control.last_step()
    << " iterations." << std::endl;

  _constraints.distribute(_solution);
}


template <int dim>
void Laplace<dim>::refine_grid ()
{
  dealii::TimerOutput::Scope t(_computing_timer, "refine");

  dealii::Vector<float> estimated_error_per_cell(_triangulation.n_active_cells());
  dealii::KellyErrorEstimator<dim>::estimate(_dof_handler,
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



template <int dim>
void Laplace<dim>::output_results (const unsigned int cycle) const
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(_dof_handler);
  data_out.add_data_vector(_solution, "solution");

  data_out.build_patches ();

  std::string const filename = "solution-" +
    dealii::Utilities::int_to_string (cycle, 2);
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);
}


template <int dim>
void Laplace<dim>::run(std::string const &preconditioner)
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
      << _dof_handler.n_dofs()
      << std::endl;

    assemble_system ();
    solve(preconditioner);
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
      std::string preconditioner = "identity";
      Laplace<2> laplace(fe_degree);
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
