// Copyright (C) 2014 by Luca Heltai (1), Saswati Roy (2), and
// Francesco Costanzo (3)
//
// (1) Scuola Internazionale Superiore di Studi Avanzati
//     E-mail: luca.heltai@sissa.it
// (2) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: sur164@psu.edu
// (3) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: costanzo@engr.psu.edu
//
// This file is subject to LGPL and may not be distributed without
// copyright and license information. Please refer to the webpage
// http://www.dealii.org/ -> License for the text and further
// information on this license.

#ifndef immersed_fem_h
#define immersed_fem_h

#include <deal.II/grid/tria.h>

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/patterns.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector_view.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/data_out.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// Elements of the C++ standard library
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <typeinfo>

// Our own include files
#include "ifem_parameters.h"
#include "exact_solution_ring_with_fibers.h"

using namespace std;

template <typename T>
std::string to_string(const T &object)
{
  return Patterns::Tools::Convert<T>::to_string(object);
}

//! This class defines simulations objects. The only method in the public
//! interface is <code>run()</code>, which is invoked to carry out the
//! simulation.
template <int dim>
class IFEM
{
public:


  // No default constructor is defined. Simulation objects must be
  // initialized by assigning the simulation parameters, which are
  // elements of objects of type IFEMParametersGeneralized.

  IFEM(IFEMParameters<dim> &par);
  ~IFEM();

  void run ();

private:


  // The parameters of the problem.

  IFEMParameters<dim> &par;


  // Vector of boundary indicators. The type of this vector matches the
  // return type of the function <code>Triangulation< dim, spacedim
  // >::get_boundary_indicator()</code>.

  vector<types::boundary_id> boundary_ids;


  // Triangulation over the control volume (fluid domain).  Following
  // <code>deal.II</code> conventions, a triangulation pertains to a manifold
  // of dimension <i>dim</i> embedded in a space of dimension
  // <i>spacedim</i>. In this case, only a single dimensional parameter
  // is specified so that the dimension of the manifold and of the
  // containing space are the same.

  Triangulation<dim> tria_f;


  // Triangulations of the immersed domain (solid domain).  Following
  // <code>deal.II</code> conventions, a triangulation pertains to a manifold
  // of dimension <i>dim</i> embedded in a space of dimension
  // <i>spacedim</i>.
  Triangulation<dim, dim> tria_s;


  // <code>FESystem</code> for the control volume. It consists of two fields:
  // velocity (a vector field of dimension <i>dim</i>) and pressure (a
  // scalar field). The meaning of the parameter <i>dim</i> is as for
  // the <code>Triangulation<dim> tria_f</code> element of the class.

  FESystem<dim> fe_f;


  // A variable to check whether the pressure field is approximated
  // using the <code>FE_DGP</code> elements.

  bool dgp_for_p;


  // This is the <code>FESystem</code> for the immersed domain.

  FESystem<dim, dim> fe_s;


  // The dof_handler for the control volume.

  DoFHandler<dim> dh_f;


  // The dof_handler for the immersed domain.

  DoFHandler<dim, dim> dh_s;


  // The triangulation of for the immersed domain defines the reference
  // configuration of the immersed domain. As the immersed domain moves
  // through the fluid, it is important to be able to conveniently
  // describe quantities defined over the immersed domain according to
  // an Eulerian view. It is therefore convenient to define a
  // <code>MappingQEulerian</code> object that will support such a
  // description.

  std::unique_ptr<MappingQEulerian<dim, Vector<double>, dim> > mapping;


  // The quadrature object for the control volume.

  QGauss<dim> quad_f;


  // The quadrature object for the immersed domain.

  Quadrature<dim> quad_s;


  // Constraints matrix for the control volume.

  ConstraintMatrix constraints_f;


  // Constraints matrix for the immersed domain.

  ConstraintMatrix constraints_s;


  // Sparsity pattern.

  BlockSparsityPattern sparsity;


  // Jacobian of the residual.

  BlockSparseMatrix<double> JF;


  // Object of <code>BlockSparseMatrix<double></code> type to be used in
  // place of the real Jacobian when the real Jacobian is not to be modified.


  BlockSparseMatrix<double> dummy_JF;


  // State of the system at current time step: velocity, pressure, and
  // displacement of the immersed domain.

  BlockVector<double> current_xi;


  // State of the system at previous time step: velocity, pressure, and
  // displacement of the immersed domain.

  BlockVector<double> previous_xi;


  // Approximation of the time derivative of the state of the system.

  BlockVector<double> current_xit;


  // Current value of the residual.

  BlockVector<double> current_res;


  // Newton iteration update.

  BlockVector<double> newton_update;


  // Vector to compute the average pressure when the average pressure is
  // set to zero.

  Vector<double> pressure_average;


  // Vector to represent a uniform unit pressure.

  Vector<double> unit_pressure;

  // Number of degrees of freedom for each component of the system.

  unsigned int n_dofs_u, n_dofs_p, n_dofs_up, n_dofs_W, n_total_dofs;

  // A couple of vectors that can be used as temporary storage. They are
  // defined as a private member of the class to avoid that the object
  // is allocated and deallocated when used, so to gain in efficiency.
  Vector<double> tmp_vec_n_total_dofs;
  Vector<double> tmp_vec_n_dofs_up;
  Vector<double> tmp_vec_n_dofs_W;


  // Matrix to be inverted when solving the problem.
  SparseDirectUMFPACK JF_inv;


  // Scalar used for conditioning purposes.
  double scaling;


  // Variable to keep track of the previous time.
  double previous_time;


  // The first dof of the pressure field.
  unsigned int constraining_dof;


  // A container to store the dofs corresponding to the pressure field.
  std::set<unsigned int> pressure_dofs;


  // Storage for the elasticity operator of the immersed domain.
  Vector <double> A_gamma;


  // Mass matrix of the immersed domain.
  SparseMatrix<double> M_gamma3;


  // Inverse of M_gamma3.
  SparseDirectUMFPACK M_gamma3_inv;


  // M_gamma3_inv * A_gamma.
  Vector <double> M_gamma3_inv_A_gamma;


  //Vector to store the volume flux due to the point-source
  Vector <double> volume_flux;


  // Area of the control volume.
  double area;


  // File stream that is used to output a file containing information
  // about the fluid flux, area and the centroid of the immersed domain
  // over time.
  ofstream global_info_file;

  // File stream that is used to output a file containing information
  // about the tip displacement of the flag in the Turek-Hron FSI Benchmark
  ofstream fsi_bm_out_file;


  //Variable to store the current_time;
  double current_time;


  //Variable to store the time step
  unsigned int time_step;


  // Variable to store time step size
  double dt;


  //The following be necessary for serialization purposes
  friend class boost::serialization::access;


  // ---------------------
  // Function declarations
  // ---------------------
  void create_triangulation_and_dofs ();

  void apply_constraints (vector<double> &local_res,
                          FullMatrix<double> &local_jacobian,
                          const Vector<double> &local_up,
                          const vector<unsigned int> &dofs,
                          unsigned int offset);

  void compute_current_bc (const double time);

  void apply_current_bc (
    BlockVector<double> &vec,
    const double time);

  void assemble_sparsity (Mapping<dim, dim> &mapping);

  void  get_area_and_first_pressure_dof ();

  void residual_and_or_Jacobian (
    BlockVector<double> &residual,
    BlockSparseMatrix<double> &Jacobian,
    const BlockVector<double> &xit,
    const BlockVector<double> &xi,
    const double alpha,
    const double t
  );

  void distribute_residual (
    Vector<double> &residual,
    const vector<double> &local_res,
    const vector<unsigned int> &dofs_1,
    const unsigned int offset_1
  );

  void distribute_jacobian (
    SparseMatrix<double> &Jacobian,
    const FullMatrix<double> &local_Jac,
    const vector<unsigned int> &dofs_1,
    const vector<unsigned int> &dofs_2,
    const unsigned int offset_1,
    const unsigned int offset_2
  );

  void distribute_constraint_on_pressure (
    Vector<double> &residual,
    const double average_pressure
  );

  void distribute_constraint_on_pressure (
    SparseMatrix<double> &jacobian,
    const vector<double> &pressure_coefficient,
    const vector<unsigned int> &dofs,
    const unsigned int offset
  );

  void localize (
    Vector<double> &local_M_gamma3_inv_A_gamma,
    const Vector<double> &M_gamma3_inv_A_gamma,
    const vector<unsigned int> &dofs
  );

  void get_Agamma_values (
    const FEValues<dim,dim> &fe_v_s,
    const vector< unsigned int > &dofs,
    const Vector<double> &xi,
    Vector<double> &local_A_gamma
  );

  template <class FEVal>
  void get_Pe_F_and_DPeFT_dxi_values (
    const FEVal &fe_v_s,
    const vector< unsigned int > &dofs,
    const Vector<double> &xi,
    const bool update_jacobian,
    vector<Tensor<2,dim,double> > &Pe,
    vector<Tensor<2,dim,double> > &F,
    vector< vector<Tensor<2,dim,double> > > &DPe_dxi
  );

  void get_inverse_transpose (
    const vector < Tensor <2, dim> > &F,
    vector < Tensor <2, dim> > &local_invFT
  );

  void get_volume_flux_vector (const double t);

  void calculate_error () const;

  unsigned int n_dofs() const
  {
    return n_total_dofs;
  };

  void output_step (
    const double t,
    const BlockVector<double> &solution,
    const unsigned int step_number,
    const double h,
    const bool _output = false
  );

  template<class Type>
  inline void set_to_zero (Type &v) const;

  template<class Type>
  inline void set_to_zero (Table<2,Type> &v) const;

  template<class Type>
  inline void set_to_zero(vector<Type> &v) const;

  double norm(const vector<double> &v);

  void fsi_bm_postprocess();

  void fsi_bm_postprocess2();

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version);

  void restart_computations();

  void save_for_restart();

};

#endif
