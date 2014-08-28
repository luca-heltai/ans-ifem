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

#ifndef ifem_parameters_generalized_h
#define ifem_parameters_generalized_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;
using namespace dealii::Functions;
using namespace std;

//! This class collects all of the user-specified parameters, both
//! pertaining to physics of the problem (e.g., shear modulus, dynamic
//! viscosity), and to other numerical aspects of the simulation (e.g.,
//! names for the grid files, the specification of the boundary
//! conditions). This class is derived from the ParameterHandler class
//! in <code>deal.II</code>.

template <int dim>
class IFEMParametersGeneralized : public ParameterHandler
{
public:
  IFEMParametersGeneralized(int argc, char **argv);

  ~IFEMParametersGeneralized();

// Polynomial degree of the interpolation functions for the velocity
// of the fluid and the displacement of the solid. This parameters
// must be greater than one.

  unsigned int degree;

// Mass density of the fluid.

  double rho_f;

// Mass density of the immersed solid.

  double rho_s;

// Dynamic viscosity of the fluid.

  double eta_f;

//Dynamic viscosity of the immersed solid.

  double eta_s;

// Shear modulus of the immersed solid.

  double mu;

// Poisson's ratio of the immersed solid.

  double nu;

// Flag to indicate whether the solid is compressible or not

  bool solid_is_compressible;

// Time step.

  double dt;


// Final time.

  double T;

// Intial time.

  double t_i;


// Dimensional constant for the equation that sets the velocity of the
// solid provided by the time derivative of the displacement function
// equal to the velocity provided by the interpolation over the
// control volume.

  double Phi_B;


// Displacement of the immersed solid at the initial time.

  ParsedFunction<dim> W_0;


// Velocity over the control volume at the initial time.

  ParsedFunction<dim> u_0;


// Dirichlet boundary conditions for the control volume.

  ParsedFunction<dim> u_g;


// Body force field.

  ParsedFunction<dim> force;


// Mesh refinement level of the control volume.

  unsigned int ref_f;


// Mesh refinement level for the immersed domain.

  unsigned int ref_s;


// Maps of boundary value functions: 1st: a boundary indicator;
// 2nd: a boundary value function.

  map<unsigned char, const Function<dim> *> boundary_map;


// Maps of boundary value functions for homogeneous Dirichlet boundary
// values: 1st: a boundary indicator; 2nd: a zero boundary value
// function.

  map<unsigned char, const Function<dim> *> zero_boundary_map;

  //:

  bool solid_has_DBC;

  std::map<unsigned char, const Function<dim> *> boundary_map_solid;

  std::map<unsigned int, double> boundary_values_solid;
//:
// Vector of flags for distinguishing between velocity and pressure
// degrees of freedom.

  vector<bool> component_mask;


// Map storing the boundary conditions: 1st: a boundary degree of freedom;
// 2nd: the value of field corresponding to the given degree of freedom.

  map<unsigned int, double> boundary_values;


// Flag to indicate whether or not the Newton iteration scheme must
// update the Jacobian at each iteration.

  bool update_jacobian_continuously;


// Flag to indicate whether or not the time integration scheme must be
// semi-implicit.

  bool semi_implicit;


// Flag to indicate how to deal with the non-uniqueness of the
// pressure field.

  bool fix_pressure;


// Flag to indicate whether homogeneous Dirichlet boundary conditions
// are applied.

  bool all_DBC;


// When set to true, an update of the system Jacobian is
// performed at the beginning of each time step.

  bool update_jacobian_at_step_beginning;


// Name of the mesh file for the solid domain.

  string solid_mesh;


// Name of the mesh file for the fluid domain.

  string fluid_mesh;


// Name of the output file.

  string output_name;


// The interval of timesteps between storage of output.

  int output_interval;


// Flag to indicating the use the spread operator.

  bool use_spread;

// Flag to determine whether the solid and fluid have the same density or not

  bool same_density;

// Flag to determine whether the solid and fluid have the same viscosity or not

  bool same_viscosity;

// List of available consitutive models for the elastic
// stress of the immersed solid:
//
// INH_0: incompressible neo-Hookean with $P_{s}^{e} = \mu (F - F^{-T})$.
//
// INH_1: incompressible neo-Hookean with $P_{s}^{e} = \mu F$.
//
// CircumferentialFiberModel:
// $P_{s}^{e} = \mu F (e_{\theta} \otimes e_{\theta}) F^{-T}$.
//
//
// CNH_W1: compressible neo-Hookean with
// $ P^{e} = \mu [F - F^{-T}/ (\det(F)^{2.0 \beta} ) ]$, .
// $\beta = \nu/(1 - 2 \nu)."
//
//
// CNH_W2: compressible neo-Hookean with
// $P^{e} = \mu F - (\mu + \tau) F^{-T}/( det(F)^{2.0 \beta} )$,
// $\beta= \nu/(1 - 2 \nu)$,
// $\tau$ is initial isotropic stress (residual pressure).
//
//
// STVK: Saint Venant Kirchhoff material with
// $ P^{e} = F (2 \mu E + \lambda \tr(E) I)
// $ E = 1/2 (F^{T} F - I)$,
// $ \lambda = 2.0 \mu \nu/(1.0 - 2.0 \nu);"

  enum MaterialModel {INH_0 = 1, INH_1, CircumferentialFiberModel,
                      CNH_W1, CNH_W2, STVK
                     };


// Variable to identify the constitutive model for the immersed solid.

  unsigned int material_model;


// String to store the name of the finite element that will be used
// for the pressure field.

  string fe_p_name;

// Variable to store the center of the ring with circumferential fibers

  Point<dim> ring_center;

// Variables to store the strength(s) and location(s) of point source(s)
  unsigned int n_pt_source;
  vector<ParsedFunction<dim>*> pt_source_strength;
  vector<Point<dim> > pt_source_location;

// Variable to store the constants necessary to impose the penalty due to
// the compressibility of the solid

  double pressure_constant_c1;
  double pressure_constant_c2;

// Variable to store the residual pressure

  double tau;

// Variables to store the type of the quadrature rule to be used for integration
// of entities defined over the immersed solid domain.

  enum SolidQuadratureRule {QGauss=1, Qiter_Qtrapez, Qiter_Qmidpoint};
  unsigned int quad_s_type;
  unsigned int quad_s_degree;

// Variable to store whether the flow should be modeled as a time-dependent
// Stokes flow, i.e. the inertial terms in the momentum balance equation will
//   be neglected for such a flow

  bool stokes_flow_like;

// Variables to store the parameters needed to construct the grids for the
// control volume and solid for FSI problems which deal with a disk interacting
// with a viscous fluid.
  bool disk_falling_test;

  Point<dim> rectangle_bl;
  Point<dim> rectangle_tr;

  bool colorize_boundary;

  Point<dim> ball_center;
  double ball_radius;

// Variable to store whether the Turek-Hron FSI Benchmark test is being run
// or not.

  bool fsi_bm;

  bool cfd_test;
  bool csm_test;

  bool only_NS;
// Variable to store whether Dirichlet boundary conditions will be imposed on
// the solid for the benchmark cases
  bool use_dbc_solid;

// Variable to store whether we are running simulations involving the brain mesh.

  bool brain_mesh;

// Variable to store whether this is a restart or not.
  bool this_is_a_restart;


// Variable to store whether data should be saved for later restart or not.
  bool save_for_restart;


// Prefix of files required for restart.
  string file_info_for_restart;

};

#endif
