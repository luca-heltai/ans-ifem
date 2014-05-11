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

#include "ifem_parameters.h"
#include <fstream>

// Names imported into to the global namespace:
using namespace dealii;
using namespace dealii::Functions;
using namespace std;

// Class constructor: the name of the input file is
// <code>immersed_fem.prm</code>. If the file does not exist at run time, it
// is created, and the simulation parameters are given default values.

template <int dim>
IFEMParameters<dim>::IFEMParameters(int argc, char **argv) :
		W_0(dim),
		u_0(dim+1),
		u_g(dim+1),
		force(dim+1),
		component_mask(dim+1, true)
{

// Declaration of parameters for the ParsedFunction objects
// in the class.
  this->enter_subsection("W0");
  ParsedFunction<dim>::declare_parameters(*this, dim);
  this->leave_subsection();

  this->enter_subsection("u0");
  ParsedFunction<dim>::declare_parameters(*this, dim+1);
  this->leave_subsection();

  this->enter_subsection("ug");
  ParsedFunction<dim>::declare_parameters(*this, dim+1);
  this->declare_entry("Function expression", "if(y>.99, 1, 0); 0; 0",
	  Patterns::Anything());
  this->leave_subsection();

  this->enter_subsection("force");
  ParsedFunction<dim>::declare_parameters(*this, dim+1);
  this->leave_subsection();


// Declaration of the class parameters and assignment of default
// values.
  this->declare_entry (
    "Velocity finite element degree",
    "2",
    Patterns::Integer(2,10)
  );
  this->declare_entry ("Fluid refinement", "3", Patterns::Integer());
  this->declare_entry ("Solid refinement", "2", Patterns::Integer());
  this->declare_entry ("Delta t", ".1", Patterns::Double());
  this->declare_entry ("Final t", "1", Patterns::Double());
  this->declare_entry ("Update J cont", "false", Patterns::Bool());
  this->declare_entry (
    "Force J update at step beginning",
    "false",
    Patterns::Bool()
  );
  this->declare_entry ("Density", "1", Patterns::Double());
  this->declare_entry ("Viscosity", "1", Patterns::Double());
  this->declare_entry ("Elastic modulus", "1", Patterns::Double());
  this->declare_entry ("Phi_B", "1", Patterns::Double());
  this->declare_entry ("Semi-implicit scheme", "true", Patterns::Bool());
  this->declare_entry ("Fix one dof of p", "false", Patterns::Bool());
  this->declare_entry (
    "Solid mesh",
    "meshes/solid_square.inp",
    Patterns::Anything()
  );
  this->declare_entry (
    "Fluid mesh",
    "meshes/fluid_square.inp",
    Patterns::Anything()
  );
  this->declare_entry ("Output base name", "out/square", Patterns::Anything());
  this->declare_entry ("Dirichlet BC indicator", "0", Patterns::Integer(0,254));
  this->declare_entry ("All Dirichlet BC", "true", Patterns::Bool());
  this->declare_entry (
    "Interval (of time-steps) between output",
    "1",
    Patterns::Integer()
  );
  this->declare_entry ("Use spread operator","true", Patterns::Bool());
  this->declare_entry (
    "Solid constitutive model",
    "INH_0",
    Patterns::Selection ("INH_0|INH_1|CircumferentialFiberModel"),
    "Constitutive models available are: "
    "INH_0: incompressible neo-Hookean with "
    "P^{e} = mu (F - F^{-T}); "
    "INH_1: incompressible Neo-Hookean with P^{e} = mu F; "
    "CircumferentialFiberModel: incompressible with "
    "P^{e} = mu F (e_{\\theta} \\otimes e_{\\theta}) F^{-T}; "
    "this is suitable for annular solid comprising inextensible "
    "circumferential fibers"
  );
  this->declare_entry (
    "Finite element for pressure",
    "FE_DGP",
    Patterns::Selection("FE_DGP|FE_Q"),
    "Select between FE_Q (Lagrange finite element space of "
    "continuous, piecewise polynomials) or "
    "FE_DGP(Discontinuous finite elements based on Legendre "
    "polynomials) to approximate the pressure field"
  );
  this->enter_subsection (
    "Equilibrium Solution of Ring with Circumferential Fibers"
  );
  this->declare_entry ("Inner radius of the ring", "0.25", Patterns::Double());
  this->declare_entry ("Width of the ring", "0.0625", Patterns::Double());
  this->declare_entry (
    "Any edge length of the (square) control volume",
    "1.",
    Patterns::Double()
  );
  this->declare_entry (
    "x-coordinate of the center of the ring",
    "0.5",
    Patterns::Double()
  );
  this->declare_entry (
    "y-coordinate of the center of the ring",
    "0.5",
    Patterns::Double()
  );
  this->leave_subsection ();


// Specification of the parmeter file. If no parameter file is
// specified in input, use the default one, else read each additional
// argument.
  if(argc == 1) 
    this->read_input ("immersed_fem.prm");
  else
    for(int i=1; i<argc; ++i)
      this->read_input(argv[i]);


// Reading in the parameters.
  this->enter_subsection ("W0");
  W_0.parse_parameters (*this);
  this->leave_subsection ();

  this->enter_subsection ("u0");
  u_0.parse_parameters (*this);
  this->leave_subsection ();

  this->enter_subsection ("ug");
  u_g.parse_parameters (*this);
  this->leave_subsection ();

  this->enter_subsection ("force");
  force.parse_parameters (*this);
  this->leave_subsection ();

  ref_f = this->get_integer ("Fluid refinement");
  ref_s = this->get_integer ("Solid refinement");
  dt = this->get_double ("Delta t");
  T = this->get_double ("Final t");
  update_jacobian_continuously = this->get_bool ("Update J cont");
  update_jacobian_at_step_beginning = this->get_bool (
    "Force J update at step beginning"
  );

  rho = this->get_double ("Density");
  eta = this->get_double ("Viscosity");

  mu = this->get_double ("Elastic modulus");
  Phi_B = this->get_double ("Phi_B");

  semi_implicit = this->get_bool ("Semi-implicit scheme");
  fix_pressure = this->get_bool ("Fix one dof of p");

  solid_mesh = this->get ("Solid mesh");
  fluid_mesh = this->get ("Fluid mesh");
  output_name = this->get ("Output base name");

  unsigned char id = this->get_integer ("Dirichlet BC indicator");
  all_DBC = this->get_bool ("All Dirichlet BC");
  output_interval = this->get_integer (
    "Interval (of time-steps) between output"
  );
  use_spread =this->get_bool ("Use spread operator");

  if(this->get("Solid constitutive model") == string("INH_0"))
    material_model = INH_0;
  else if (this->get("Solid constitutive model") == string("INH_1"))
    material_model = INH_1;
  else if (this->get("Solid constitutive model")
	   == string("CircumferentialFiberModel"))
    material_model = CircumferentialFiberModel;
  else
    cout
      << " No matching constitutive model found! Using INH_0."
      << endl;

  component_mask[dim] = false;
  static ZeroFunction<dim> zero (dim+1);
  zero_boundary_map[id] = &zero;
  boundary_map[id] = &u_g;

  degree = this->get_integer ("Velocity finite element degree");

  fe_p_name = this->get ("Finite element for pressure");
  fe_p_name +="<dim>(" + Utilities::int_to_string(degree-1) + ")";

   this->enter_subsection (
					 "Equilibrium Solution of Ring with Circumferential Fibers"
						   );
   ring_center[0] = this->get_double ("x-coordinate of the center of the ring");
   ring_center[1] = this->get_double ("y-coordinate of the center of the ring");
   this->leave_subsection();


// The following lines help keeping track of what prm file goes

// with a specific output.  Therefore, they are here for

// convenience and not for any specific computational need.
  ofstream paramfile((output_name+"_param.prm").c_str());
  this->print_parameters(paramfile, this->Text);
}

template class IFEMParameters<2>;
template class IFEMParameters<3>;
