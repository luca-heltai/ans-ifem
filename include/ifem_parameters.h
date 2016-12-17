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

#ifndef ifem_parameters_h
#define ifem_parameters_h

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
class IFEMParameters :
  public ParameterHandler
{
public:
  IFEMParameters(int argc, char **argv);


//! Polynomial degree of the interpolation functions for the velocity
//! of the fluid and the displacement of the solid. This parameters
//! must be greater than one for the problem to be stable.

  unsigned int degree;


//! Mass density of the fluid and of the immersed solid.

  double rho;


//! Dynamic viscosity of the fluid and of the immersed solid.

  double eta;


//! Shear modulus of the neo-Hookean immersed solid.

  double mu;

//! Time step.

  double dt;


//! Final time.

  double T;


//! Dimensional constant for the equation that sets the velocity of the
//! solid provided by the time derivative of the displacement function
//! equal to the velocity provided by the interpolation over the
//! control volume.

  double Phi_B;


//! Displacement of the immersed solid at the initial time.

  ParsedFunction<dim> W_0;


//! Velocity over the control volume at the initial time.

  ParsedFunction<dim> u_0;


//! Dirichlet boundary conditions for the control volume.

  ParsedFunction<dim> u_g;


//! Body force field.

  ParsedFunction<dim> force;


//! Mesh refinement level of the fluid control volume.

  unsigned int ref_f;


//! Mesh refinement level for the immersed solid domain.

  unsigned int ref_s;


//! Maps of boundary value functions: 1st: a boundary indicator;
//! 2nd: a boundary value function.

  map<unsigned char, const Function<dim> *> boundary_map;


//! Maps of boundary value functions for homogeneous Dirichlet boundary
//! values: 1st: a boundary indicator; 2nd: a zero boundary value
//! function.

  map<unsigned char, const Function<dim> *> zero_boundary_map;


//! Vector of flags for distinguishing between velocity and pressure
//! degrees of freedom.

  vector<bool> component_mask;


//! Map storing the boundary conditions: 1st: a boundary degree of freedom;
//! 2nd: the value of field corresponding to the given degree of freedom.

  map<unsigned int, double> boundary_values;


//! Flag to indicate whether or not the Newton iteration scheme must
//! update the Jacobian at each iteration.

  bool update_jacobian_continuously;


//! Flag to indicate whether or not the time integration scheme must
//! be semi-implicit. If the time integration is semi_implicit, the
//! non linearity of the problem is reduced, because the immersed
//! structure is evaluated at the previous time step, and maintained
//! constant during the entire Newton iteration on the non-linear
//! parts of the Navier-Stokes equation.

  bool semi_implicit;


//! Flag to indicate how to deal with the non-uniqueness of the
//! pressure field. This flag is only used when the boundary
//! conditions on the velocity are all of Dirichlet type

  bool fix_pressure;


//! Flag to indicate whether only Dirichlet boundary conditions are
//! applied on the velocity field. In this case the pressure is known
//! only up to a constant, and we can decide how to resolve the non
//! uniqueness of the solution by either fixing one degree of freedom
//! of the pressure (setting @fix_pressure to true), or by filtering
//! the constant modes of the pressure (the default behaviour if
//! @fix_pressure is false).

  bool all_DBC;


//! When set to true, an update of the system Jacobian is
//! performed at the beginning of each time step.

  bool update_jacobian_at_step_beginning;


//! Name of the mesh file for the solid domain.

  string solid_mesh;


//! Name of the mesh file for the fluid domain.

  string fluid_mesh;


//! Base name of the output files. To this file name, a number
//! identifying the time step is added during the output procedure,
//! together with the appropriate extension (vtu, for binary vtk files).

  string output_name;


//! The interval of timesteps between storage of output. If the time
//! step is very small, then it might be desirable not to generate the
//! output files at each time interval.

  int output_interval;


//! Flag indicating the use of the spread operator.

  bool use_spread;


//! List of available consitutive models for the elastic
//! stress of the immersed solid
  enum MaterialModel
  {
    //! incompressible neo-Hookean with
    //! \f$P_{s}^{e} = \mu (F - F^{-T})\f$
    INH_0 = 1,

    //! incompressible neo-Hookean with
    //! \f$P_{s}^{e} = \mu F\f$
    INH_1,

    //! \f$P_{s}^{e} = \mu F (e_{\theta} \otimes e_{\theta}) F^{-T}\f$.
    CircumferentialFiberModel
  };


//! Variable to identify the constitutive model for the immersed solid.

  MaterialModel material_model;


//! String to store the name of the finite element that will be used
//! for the pressure field.

  string fe_p_name;

//! Variable to store the center of the ring with circumferential fibers

  Point<dim> ring_center;
};


#endif
