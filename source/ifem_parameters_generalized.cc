#include "ifem_parameters_generalized.h"
#include <iostream>
#include <fstream>

// Class constructor: the name of the input file is
// <code>immersed_fem.prm</code>. If the file does not exist at run time, it
// is created, and the simulation parameters are given default values.

template <int dim>
IFEMParametersGeneralized<dim>::IFEMParametersGeneralized(int argc, char **argv) :
  IFEMParameters<dim>(argc,argv),
  W_0(dim),
  u_0(dim+1),
  u_g(dim+1),
  force(dim+1),
  component_mask(dim+1, true)
  //,pt_source_strength(1)
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
  this->declare_entry ("Fluid refinement", "4", Patterns::Integer());
  this->declare_entry ("Solid refinement", "1", Patterns::Integer());
  this->declare_entry ("Delta t", ".1", Patterns::Double());
  this->declare_entry ("Final t", "1", Patterns::Double());
  this->declare_entry ("Initial t", "0.", Patterns::Double());
  this->declare_entry ("Update J cont", "false", Patterns::Bool());
  this->declare_entry (
    "Force J update at step beginning",
    "false",
    Patterns::Bool()
  );
  this->declare_entry ("Fluid density", "1", Patterns::Double());
  this->declare_entry ("Solid density", "1", Patterns::Double());
  this->declare_entry ("Fluid viscosity", "1", Patterns::Double());
  this->declare_entry ("Solid viscosity", "1", Patterns::Double());
  this->declare_entry ("Solid elastic modulus", "1", Patterns::Double());
  this->declare_entry ("Solid Poisson\'s ratio", "0.4", Patterns::Double());
  this->declare_entry ("Phi_B", "1", Patterns::Double());
  this->declare_entry ("Semi-implicit scheme", "true", Patterns::Bool());
  this->declare_entry ("Fix one dof of p", "false", Patterns::Bool());
  this->declare_entry (
    "Solid mesh",
    "mesh/solid_square.inp",
    Patterns::Anything()
  );
  this->declare_entry (
    "Fluid mesh",
    "mesh/fluid_square.inp",
    Patterns::Anything()
  );
  this->declare_entry ("Output base name", "out/square", Patterns::Anything());
  this->declare_entry ("Dirichlet BC indicator", "1", Patterns::Integer(0,254));
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
    Patterns::Selection ("INH_0|INH_1|CircumferentialFiberModel|CNH_W1|CNH_W2|STVK"),
    "Constitutive models available are: \n"
    "INH_0: incompressible neo-Hookean with \n\t"
    "P^{e} = mu (F - F^{-T}); \n "
    "INH_1: incompressible Neo-Hookean with \n\t"
    "P^{e} = mu F; \n"
    "CircumferentialFiberModel: incompressible with \n\t"
    "P^{e} = mu F (e_{\\theta} \\otimes e_{\\theta}) F^{-T};"
    "\n this is suitable for annular solid comprising inextensible "
    "circumferential fibers \n"
    "CNH_W1: compressible neo-Hookean with \n\t"
    "P^{e} = mu [ F - F^{-T}/( det(F)^{2.0 beta} ) ], \n\t"
    "beta= nu/(1 - 2 * nu); \n"
    "CNH_W2: compressible neo-Hookean with \n\t"
    "P^{e} = mu F - (mu + tau) F^{-T}/( det(F)^{2.0 beta} ),\n\t"
    "beta= nu/(1 - 2 * nu),\n\t"
    "tau is initial isotropic stress; \n"
    "STVK: Saint Venant Kirchhoff material with \n\t"
    "P^{e} = F (2 mu E + lambda tr(E) I),\n\t"
    "E = 1/2 (F^{T} F - I),\n\t"
    "lambda = 2.0 mu nu/(1.0 - 2.0 nu);"
  );

  this->declare_entry("Solid is compressible", "false", Patterns::Bool());

  this->declare_entry("Solid pressure constant, c1", "1.", Patterns::Double());
  this->declare_entry("Solid pressure constant, c2", "0.", Patterns::Double());
  this->declare_entry("Solid residual pressure", "0.", Patterns::Double());

  this->declare_entry ("Finite element for pressure",
                       "FE_DGP",
                       Patterns::Selection("FE_DGP|FE_Q"),
                       "Select between FE_Q (Lagrange finite element space of "
                       "continuous, piecewise polynomials) or "
                       "FE_DGP(Discontinuous finite elements based on Legendre "
                       "polynomials) to approximate the pressure field"
                      );

  this->declare_entry(
    "Solid quadrature rule",
    "QIter+QTrapez",
    Patterns::Selection ("QGauss|QIter+QTrapez|QIter+QMidpoint"),
    "Select one of the followings:\n"
    "QGauss: Gauss-Legendre quadrature of arbitrary order;\n"
    "QIter+QTrapez: Quadrature rule comprising copies of trapezoidal rule;\n"
    "QIter+QMidpoint: Quadrature rule comprising copies of midpoint rule."
  );

  this->declare_entry("Solid quadrature rule degree/copies",
                      "10",
                      Patterns::Integer());


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

  this->enter_subsection("Point source");
  this->declare_entry("Number of point sources present",
                      "0",
                      Patterns::Integer(),
                      "It is recommended that FE_Q be used for your simulation"
                      "when source strength is non-zero.");
  this->declare_entry("List of location(s) of point source(s):",
                      "(0.5, 0.5)",
                      Patterns::Anything(),
                      "Items of this list are separated using semi-colon.");
  //:
  this->enter_subsection("Strength of point source no.1");
  ParsedFunction<dim>::declare_parameters(*this, 1);
  this->leave_subsection();

  this->enter_subsection("Strength of point source no.2");
  ParsedFunction<dim>::declare_parameters(*this, 1);
  this->leave_subsection();

  this->enter_subsection("Strength of point source no.3");
  ParsedFunction<dim>::declare_parameters(*this, 1);
  this->leave_subsection();

  this->enter_subsection("Strength of point source no.4");
  ParsedFunction<dim>::declare_parameters(*this, 1);
  this->leave_subsection();
  //:
  this->leave_subsection();

  this->declare_entry("Time-dependent Stokes flow","false",Patterns::Bool());

  this->enter_subsection("Grid parameters for disk in viscous flow test");
  //:
  this->declare_entry("Use following grid parameters","false",Patterns::Bool());
  //:
  this->enter_subsection("Grid dimensions for the (rectangular) control volume");
  this->declare_entry("bottom left corner, x-coord", "0.", Patterns::Double());
  this->declare_entry("bottom left corner, y-coord", "0.", Patterns::Double());
  this->declare_entry("top right corner, x-coord", "1.", Patterns::Double());
  this->declare_entry("top right corner, y-coord", "1.", Patterns::Double());
  this->declare_entry("Colorize boundary", "false", Patterns::Bool());
  this->leave_subsection();
  //:
  this->enter_subsection("Grid dimensions for the disk");
  this->declare_entry("disk center, x-coord", "0.5", Patterns::Double());
  this->declare_entry("disk center, y-coord", "0.5", Patterns::Double());
  this->declare_entry("disk radius", "0.1", Patterns::Double());
  this->leave_subsection();
  //:
  this->leave_subsection();

  this->declare_entry("Turek-Hron FSI Benchmark test", "false", Patterns::Bool());
  this->declare_entry("Turek-Hron CFD Benchmark test", "false", Patterns::Bool());
  this->declare_entry("Turek-Hron CSM Benchmark test", "false", Patterns::Bool());
  this->declare_entry("Turek-Hron test-- Impose DBC for solid",
                      "false",
                      Patterns::Bool());
  this->declare_entry("Solve only NS component", "false", Patterns::Bool());


  this->enter_subsection("Grid parameters for brain mesh");
  //:
  this->declare_entry("Use brain mesh", "false",Patterns::Bool());
  this->declare_entry("Scaling factor", "0.0033333", Patterns::Double());
  this->declare_entry("Translation x-dirn", "28.0", Patterns::Double());
  this->declare_entry("Translation y-dirn", "22.0", Patterns::Double());
  //:
  this->leave_subsection();

  this->enter_subsection("For restart");
  this->declare_entry("Save data for a possible restart",
                      "true", Patterns::Bool());
  this->declare_entry("This is a restart","false", Patterns::Bool());
  this->declare_entry ("File prefix used for files needed for restart",
                       "-restart-",
                       Patterns::Anything());

  this->leave_subsection();


// Specification of the parameter file. If no parameter file is
// specified in input, use the default one, else read each additional
// argument.
  if (argc == 1)
    this->read_input ("immersed_fem.prm");
  else
    for (int i=1; i<argc; ++i)
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
  t_i = this->get_double ("Initial t");
  update_jacobian_continuously = this->get_bool ("Update J cont");
  update_jacobian_at_step_beginning = this->get_bool (
                                        "Force J update at step beginning"
                                      );

  rho_f = this->get_double ("Fluid density");
  rho_s = this->get_double ("Solid density");
  eta_f = this->get_double ("Fluid viscosity");
  eta_s = this->get_double ("Solid viscosity");

  same_density = (rho_f == rho_s);
  same_viscosity = (eta_f == eta_s);

  solid_is_compressible = this->get_bool ("Solid is compressible");

  mu = this->get_double ("Solid elastic modulus");
  nu = this ->get_double ("Solid Poisson\'s ratio");
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

  if (this->get("Solid constitutive model") == string("INH_0"))
    material_model = INH_0;
  else if (this->get("Solid constitutive model") == string("INH_1"))
    material_model = INH_1;
  else if (this->get("Solid constitutive model")
           == string("CircumferentialFiberModel"))
    material_model = CircumferentialFiberModel;
  else if (this->get("Solid constitutive model") == string("CNH_W1"))
    material_model = CNH_W1;
  else if (this->get("Solid constitutive model") == string("CNH_W2"))
    material_model = CNH_W2;
  else if (this->get("Solid constitutive model")
           == string("STVK"))
    material_model = STVK;
  else
    cout
        << " No matching constitutive model found! Using INH_0."
        << endl;


  pressure_constant_c1 = this->get_double("Solid pressure constant, c1");
  pressure_constant_c2 = this->get_double("Solid pressure constant, c2");

  tau = this->get_double("Solid residual pressure");

  if (this->get("Solid quadrature rule") == string("QGauss"))
    {
      quad_s_type = QGauss;
      cout << "Using QGauss \n";
    }
  else if (this->get("Solid quadrature rule") == string("QIter+QTrapez"))
    {
      quad_s_type = Qiter_Qtrapez;
      cout << "Using QI+QT \n";
    }
  else if (this->get("Solid quadrature rule")
           == string("QIter+QMidpoint"))
    {
      quad_s_type = Qiter_Qmidpoint;
      cout << "Using QI+QMid \n";
    }
  else
    cout
        << " No matching pattern found! Using QGauss."
        << endl;

  quad_s_degree = this->get_integer ("Solid quadrature rule degree/copies");
  cout<<" deg/copy="<<quad_s_degree<<endl;
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

  this->enter_subsection("Point source");
  //:
  n_pt_source = this->get_integer("Number of point sources present");
  cout<<"No. of pts ="<<n_pt_source<<endl;

  if (n_pt_source)
    {
      string str_loc_all = this->get("List of location(s) of point source(s):");
      vector<string> vec_loc = Utilities::split_string_list(str_loc_all,';');
      vector<string> vec_loc_coord;

      Assert(n_pt_source == vec_loc.size(),
             ExcDimensionMismatch(n_pt_source, vec_loc.size())
            );


      for (unsigned int i=0; i< n_pt_source; ++i)
        {
          vec_loc_coord = Utilities::split_string_list(vec_loc[i].substr(1, vec_loc[i].size()-2),',');
          Assert(vec_loc_coord.size() == dim,
                 ExcDimensionMismatch(vec_loc_coord.size(), dim)
                );

          if (dim == 2)
            {
              pt_source_location.push_back(Point<dim>(Utilities::string_to_double(vec_loc_coord[0]),
                                                      Utilities::string_to_double(vec_loc_coord[1])));
            }
          else
            pt_source_location.push_back(Point<dim>(Utilities::string_to_double(vec_loc_coord[0]),
                                                    Utilities::string_to_double(vec_loc_coord[1]),
                                                    Utilities::string_to_double(vec_loc_coord[2])));


          pt_source_strength.push_back(new ParsedFunction<dim>(1));
          cout<<"Parsing source fn ...."<<"Strength of point source no." + Utilities::int_to_string(i+1)<<endl;
          this->enter_subsection("Strength of point source no." + Utilities::int_to_string(i+1));
          pt_source_strength[i]->parse_parameters(*this);
          this->leave_subsection();
        }
    }
  this->leave_subsection();

  stokes_flow_like = this->get_bool("Time-dependent Stokes flow");

  this->enter_subsection("Grid parameters for disk in viscous flow test");
  //:
  disk_falling_test = this->get_bool("Use following grid parameters");
  //:
  this->enter_subsection("Grid dimensions for the (rectangular) control volume");
  rectangle_bl[0] = this->get_double("bottom left corner, x-coord");
  rectangle_bl[1] = this->get_double("bottom left corner, y-coord");

  rectangle_tr[0] = this->get_double("top right corner, x-coord");
  rectangle_tr[1] = this->get_double("top right corner, y-coord");
  colorize_boundary = this->get_bool("Colorize boundary");
  this->leave_subsection();

  this->enter_subsection("Grid dimensions for the disk");
  ball_center[0] = this->get_double ("disk center, x-coord");
  ball_center[1] = this->get_double ("disk center, y-coord");
  ball_radius = this->get_double ("disk radius");
  this->leave_subsection();
  //:
  this->leave_subsection();

  fsi_bm = this->get_bool ("Turek-Hron FSI Benchmark test");

  cfd_test = this->get_bool("Turek-Hron CFD Benchmark test");
  csm_test = this->get_bool("Turek-Hron CSM Benchmark test");

  use_dbc_solid = this->get_bool("Turek-Hron test-- Impose DBC for solid");

  only_NS = this->get_bool("Solve only NS component");

  static ZeroFunction<dim> zero_solid (dim);

  if (fsi_bm) //This just to do with the way in which the boundary ids are set up for the meshes for the FSI BM problem
    {
      boundary_map[80] = &zero;
      boundary_map[81] = &zero;
      boundary_map[1] = &zero;
      zero_boundary_map[80] = &zero;
      zero_boundary_map[81] = &zero;
      zero_boundary_map[1] = &zero;
      if (use_dbc_solid)
        boundary_map_solid[81] = &zero_solid;
    }
  if (use_dbc_solid)
    boundary_map_solid[0] = &zero_solid;

  this->enter_subsection("Grid parameters for brain mesh");
  //:
  brain_mesh = this->get_bool("Use brain mesh");
  //:
  this->leave_subsection();

  if (only_NS)
    {
      cout<<" NOTE: Only solving for NS component of the problem!"<<endl;
      same_density = true;
      rho_s = rho_f;
      same_viscosity = true;
      eta_s = eta_f;
      mu = 0.0;
      output_name += "_onlyNS";
    }

  this->enter_subsection("For restart");
  save_for_restart = this->get_bool ("Save data for a possible restart");
  this_is_a_restart = this->get_bool ("This is a restart");
  file_info_for_restart = this->get("File prefix used for files needed for restart");
  this->leave_subsection();


// The following lines help keeping track of what prm file goes

// with a specific output.  Therefore, they are here for

// convenience and not for any specific computational need.
  ofstream paramfile((output_name+"_param.prm").c_str());
  this->print_parameters(paramfile, this->Text);

  if (n_pt_source) pt_source_strength[0]->set_time(0.);
  std::cout << "\n==============================" << std::endl;
  std::cout << "Parameters\n"
            << "==========\n"
            << "Density fluid:\t\t\t"   <<  rho_f << "\n"
            << "Density solid:\t\t\t"   <<  rho_s << "\n"
            << "Viscosity fluid:\t\t"  <<  eta_f << "\n"
            << "Viscosity solid:\t\t"  <<  eta_s<<"\n"
            << "Compressible solid:\t\t"<<  (solid_is_compressible? "true":"false") << "\n"
            << "Shear mod. solid:\t\t" <<  mu <<"\n"
            << "Poisson's ratio:\t\t"  <<  nu <<"\n"
            << "Matl. type: \t\t\t"    << material_model<<"\n"
            << "Semi-implicit:\t\t\t"  <<  (semi_implicit? "true":"false") <<"\n"
            << "delta_t:\t\t\t"        <<  dt << "\n"
            << "Final time:\t\t\t"     <<  T << "\n"
            //<< "Pt. source strength (at t= 0):\t"<< (n_pt_source ?pt_source_strength[0]->value(Point<dim>(0.,0.):0) << "\n";
            << "Output files name:\t\t"  << output_name << "\n";

  if (solid_is_compressible)
    cout
        << "Constant c1:\t\t\t "<< pressure_constant_c1 <<"\n"
        << "Constant c2:\t\t\t "<< pressure_constant_c2 <<"\n";

  cout<< std::endl;

}


template <int dim>
IFEMParametersGeneralized<dim>::~IFEMParametersGeneralized()
{
  for (unsigned int i = 0; i < n_pt_source; ++i)
    delete pt_source_strength[i];
}

template class IFEMParametersGeneralized<2>;
template class IFEMParametersGeneralized<3>;
