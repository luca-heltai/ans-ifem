/* $Id: step-35.cc 21497 2010-07-14 13:50:13Z bangerth $ */
/* Version: $Name:  $ */
/*      */
/*    Copyright (C) 2007, 2008, 2009, 2010, 2011 by the deal.II authors */
/*    Author: Abner Salgado, Texas A&M University 2009 */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */


				 // @sect3{Include files}

				 // This code was developed starting from one of the
				 // examples provided by the deal.II FEM library.
				 // The example in question is step-33, which is
				 // a tutorial on Euler flow that makes use of the
				 // Sacado package.  This package, part of the Trilinos
				 // libraries, enables automatic differentiation.
#include <base/parameter_handler.h>
#include <base/point.h>
#include <base/function.h>
#include <base/tensor.h>
#include <base/parsed_function.h>
#include <base/quadrature_lib.h>
#include <base/multithread_info.h>
#include <base/thread_management.h>
#include <base/work_stream.h>
#include <base/parallel.h>
#include <base/utilities.h>
#include <base/conditional_ostream.h>

#include <lac/vector.h>
#include <lac/constraint_matrix.h>
#include <lac/sparse_matrix.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <lac/solver_gmres.h>
#include <lac/sparse_ilu.h>
#include <lac/sparse_direct.h>
#include <lac/vector_view.h>

#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/grid_refinement.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <grid/tria_boundary_lib.h>
#include <grid/grid_in.h>

#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <dofs/dof_renumbering.h>
#include <dofs/dof_constraints.h>

#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <fe/fe_values.h>
#include <fe/fe_tools.h>
#include <fe/fe_system.h>
#include <fe/mapping_q_eulerian.h>

#include <numerics/matrices.h>
#include <numerics/vectors.h> //Look in here for the VectorTool name space.
#include <numerics/fe_field_function.h>
#include <numerics/data_out.h>


// Elements of the C++ standard library
#include <iostream>
#include <fstream>
#include <string> // Already included by "utilities.h"
#include <vector> // Already included by "utilities.h"
#include <map>    // Already included by "utilities.h"
#include <cmath>
#include <typeinfo>


				 // Finally we import all the deal.II
				 // names to the global namespace
using namespace dealii;
using namespace dealii::Functions;
using namespace std;


				 /** This class
				 collects all of the user-specified parameters,
				 both pertaining to physics of problem (e.g., shear
				 modulus, dynamic viscosity),
				 and to other numerical aspects of the
				 simulation (e.g., names for the grid files,
				 the specification of the boundary conditions)
				 This class is derived from the \b ParameterHandler
				 class in deal.II */
template <int dim>
class ProblemParameters :
  public ParameterHandler 
{
  public:
    ProblemParameters();
				     /** Degree of approximation for
				     the finite element space of the velocity and
				     displacement fields. This parameters must be
				     greater than one. */
    unsigned int degree;

				     //! Mass density of the fluid and of the immersed solid.
    double rho; 
				     /** Dynamic viscosity of the fluid and of the
				     immersed solid. */
    double eta;
    
				     //! Shear modulus of the neo-Hookean immersed solid.
    double mu;

				     //! Time step.
    double dt;

				     //! Final time.
    double T;

				     //! Dimensional constant for the velocity of the solid Eq.
    double Phi_B;

				     /** Displacement field of the immersed solid
				     at the initial time. */
    ParsedFunction<dim> W_0;

				     /** Velocity field of the fluid at the
				     initial time. */
    ParsedFunction<dim> u_0;

				     //! Dirichlet boundary conditions (for the fluid).
    ParsedFunction<dim> u_g;
    
				     //! Body force field.
    ParsedFunction<dim> force;

				     //! Fluid mesh refinement level.
    unsigned int ref_f;

    				 //! Solid mesh refinement level.
    unsigned int ref_s;

				     /** Maps of boundary value functions.
				     1st: a boundary indicator;
				     2nd: a boundary value function. */
    map<unsigned char, const Function<dim> *> boundary_map;

    				     /** Maps of boundary value functions
    				     for homogeneous Dirichlet boundary values.
    				     1st: a boundary indicator;
    				     2nd: a zero boundary value function. */
    map<unsigned char, const Function<dim> *> zero_boundary_map;
 
 				     /** Vector of flags for distinguishing between
 				     velocity and pressure degrees of freedom. */
    vector<bool> component_mask;

 				     /** Map storing the boundary conditions.
 				     1st: a boundary degree of freedom;
				     2nd: the value of field corresponding to the given
				     degree of freedom. */
    map<unsigned int, double> boundary_values;

 				     /** Flag to indicate whether or not the Newton
 				     iteration scheme must update the Jacobian
 				     at each iteration. */
    bool update_jacobian_continuously;

 				     /** Flag to indicate whether or not the time
 				     integration scheme must be semi-implicit. */
    bool semi_implicit;

 				     /** Flag to indicate how to deal with the
 				     non-uniqueness of the pressure field. */
    bool fix_pressure;
				    /** Flag to indicate whether Dirichlet boundary conditions
				  need to be applied on the all boundaries of the control volume*/
    bool all_DBC;
    
 				     /** When set to true, an update of the system Jacobian is
 				     performed at the beginning of each time step. */
    bool update_jacobian_at_step_beginning;
    
 				     //! Name of the mesh file for the solid domain.
    string solid_mesh;

 				     //! Name of the mesh file for the fluid domain.
    string fluid_mesh;

 				     //! Name of the output file.
    string output_name;
};

 				     /** Class constructor: the name of the input file
 				     is "immersed_fem.prm".  If the file does not exist at
 				     run time, it is created, and the simulation parameters
 				     are given default values.
 				     */
template <int dim>
ProblemParameters<dim>::ProblemParameters() :
		W_0(dim),
		u_0(dim+1),
		u_g(dim+1),
		force(dim+1),
		component_mask(dim+1, true)
{
 				     /** Declaration of parameters for the problem. */
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

  this->declare_entry("Velocity finite element degree", "2", Patterns::Integer(2,10));
  this->declare_entry("Fluid refinement", "4", Patterns::Integer());
  this->declare_entry("Solid refinement", "1", Patterns::Integer());
  this->declare_entry("Delta t", ".1", Patterns::Double());
  this->declare_entry("Final t", "1", Patterns::Double());
  this->declare_entry("Update J cont", "false", Patterns::Bool());
  this->declare_entry("Force J update at step beginning","false",
                      Patterns::Bool());
  this->declare_entry("Density", "1", Patterns::Double());
  this->declare_entry("Viscosity", "1", Patterns::Double());
  this->declare_entry("Elastic modulus", "1", Patterns::Double());
  this->declare_entry("Phi_B", "1", Patterns::Double());
  this->declare_entry("Semi-implicit scheme", "true", Patterns::Bool());
  this->declare_entry("Fix one dof of p", "false", Patterns::Bool());
  this->declare_entry("Solid mesh", "mesh/solid.inp", Patterns::Anything());
  this->declare_entry("Fluid mesh", "mesh/fluid.inp", Patterns::Anything());
  this->declare_entry("Output base name", "out/square", Patterns::Anything());
  this->declare_entry("Dirichlet BC indicator", "1", Patterns::Integer(0,254));
   this->declare_entry("All Dirichlet BC", "true", Patterns::Bool());

 				     /** Specification of the parmeter file. */  
  this->read_input("immersed_fem.prm");
 
 				     /** Reading in the parameters. */  
  this->enter_subsection("W0");
  W_0.parse_parameters(*this);
  this->leave_subsection();
  
  this->enter_subsection("u0");
  u_0.parse_parameters(*this);
  this->leave_subsection();
  
  this->enter_subsection("ug");
  u_g.parse_parameters(*this);
  this->leave_subsection();

  this->enter_subsection("force");
  force.parse_parameters(*this);
  this->leave_subsection();

  ref_f = this->get_integer("Fluid refinement");
  ref_s = this->get_integer("Solid refinement");
  dt = this->get_double("Delta t");
  T = this->get_double("Final t");
  update_jacobian_continuously = this->get_bool("Update J cont");
  update_jacobian_at_step_beginning = this->get_bool("Force J update at step beginning");

  rho = this->get_double("Density");
  eta = this->get_double("Viscosity");

  mu = this->get_double("Elastic modulus");
  Phi_B = this->get_double("Phi_B");

  semi_implicit = this->get_bool("Semi-implicit scheme");
  fix_pressure = this->get_bool("Fix one dof of p");
  
  solid_mesh = this->get("Solid mesh");
  fluid_mesh = this->get("Fluid mesh");
  output_name = this->get("Output base name");

  unsigned char id = this->get_integer("Dirichlet BC indicator");
   all_DBC = this->get_bool("All Dirichlet BC");

  component_mask[dim] = false;
  static ZeroFunction<dim> zero(dim+1);
  zero_boundary_map[id] = &zero;
  boundary_map[id] = &u_g;
  
  degree = this->get_integer("Velocity finite element degree");

}

				/** It defines simulations objects. The only method
				in the public interface is <b>run()</b>, which is 
				invoked to carry out the simulation. */
template <int dim>
class ImmersedFEM
{
  public:
                /** No default constructor is defined.
                Simulation objects must be initialized
				by assigning the simulation parameters, which are
				elements of objects of type <b>ProblemParameters</b>. */
    ImmersedFEM(ProblemParameters<dim> &par);
    ~ImmersedFEM();

    void run ();
    
  private:
				/** The parameters of the problem. */
    ProblemParameters<dim> &par;

				/** Vector of boundary indicators.
				    The type of this vector matches the return
				    type of the function
				    Triangulation< dim, spacedim >::get_boundary_indicators	()	
				*/
    std::vector<unsigned char>     boundary_indicators;

				/** This is the triangulation over the control volume (fluid
				domain). Following <b>deal.II</b> conventions,
				a triangulation pertains to a manifold of dimension
				<i>dim</i> embedded in a space of dimension <i>spacedim</i>.
				In this case, only a single dimensional parameter is specified
				so that the dimension of the manifold and of the containing
				space are the same. */
    Triangulation<dim>			tria_f;
    
				/** Triangulations of the immersed domain
				(solid domain). Following <b>deal.II</b> conventions,
				a triangulation pertains to a manifold of dimension
				<i>dim</i> embedded in a space of dimension <i>spacedim</i>.
				While in this case the two dimension parameters
				are set equal to each other, it is possible to
				formulate problems in which the immersed domain is
				a manifold of dimension lower than that of the containing
				space. */
    Triangulation<dim, dim>		tria_s;

				/** This is FESystems for the control volume. It consists
				of two fields: velocity (a vector field of dimension <i>dim</i>) 
				and pressure (a scalar field).  The meaning of the parameter
				<i>dim</i> is as for the <b>Triangulation<dim> tria_f</b> 
				element of the class. */
    FESystem<dim>		fe_f;
    
				/** This is FESystems for the immersed domain. It consists
				of a single field: displacement (a vector field of dimension <i>dim</i>).  The meaning of the parameter
				<i>dim</i> is as for the <b>Triangulation<dim> tria_s</b> 
				element of the class. */
    FESystem<dim, dim>	fe_s;

				/** The dof_handlers for the FESystem defined for the
				control volume. */
    DoFHandler<dim>			dh_f;
    DoFHandler<dim, dim>	dh_s;

				// The triangulation defined for the immersed domain defines the
				// reference configuration of the immersed domain, where
				// reference (and later current) configuration must be understood
				// in the sense of continuum mechanics.  As the immersed domain
				// moves through the fluid, it is important to be able to
				// conveniently describe quantities defined over the immersed
				// domain according to an Eulerian view.  It is therefore
				// convenient to define a MappingQEulerian object that will
				// support such a description.  To be sure, the statement below
				// only defines a point to the mapping object in question.
    MappingQEulerian<dim, Vector<double>, dim> * mapping;
    
				// The quadrature objects for the fluid and solid domains.
    QGauss<dim>	quad_f;
    QGauss<dim>	quad_s;

				     // Constraints and sparsity pattern
    ConstraintMatrix		constraints_f;
    ConstraintMatrix		constraints_s;
    BlockSparsityPattern	sparsity;
    
				// We write the system as 
				//    F(xi_n, xi_n_1, t ) = 0,
				// where xi is a list of three fields, two of which are vector
				// fields.  Specifically, we have xi = (u, p, W).
				// xi_n and x_n_1 are the values of xi at the iterations
				// n and n + 1, respectively.
				// To solve this problem, we will therefore need to compute the
				// Jacobian of the function F.  Accordingly an object of type
				// BlockSparseMatrix is defined.
    BlockSparseMatrix<double> JF;
				// Sometimes it is not necessary to update the Jacobian matrix
				// during a Newton iteration.  However, we still need to update
				// the value of the system's residual.  Since the function that
				// computes the system's residual is the same as the function
				// that computes the Jacobian, when updating the redual but not
				// the Jacobian, we pass the function in question a dummy object
				// of the same type as JF.
    BlockSparseMatrix<double> dummy_JF;

				     // This vector contains all the state of the system at
				     // current time step: velocity, pressure, and displacement
				     // of the solid.
    BlockVector<double> current_xi;
				     // This vector contains the state of the system at
				     // previous time step: velocity, pressure, and displacement
				     // of the solid.
    BlockVector<double> previous_xi;
				     // Approximation of the time derivative of the state of the
				     // system.
    BlockVector<double> current_xit;
				     // BlockVector to store the current value of the residual.
    BlockVector<double> current_res;
				     // BlockVector to receive the state update during Newton
				     // iterations.
    BlockVector<double> newton_update;
				     // Vector to compute the average pressure in the system.
    Vector<double> pressure_average;
				     // Vector to represent a uniform unit pressure.
    Vector<double> unit_pressure;


				     // We now define a vector that can be used as a temporary
				     // storage of the data in current_xi.  This object is
				     // defined simply for convenience.  It is defined as a
				     // private member of the class to avoid that the object is
				     // allocated and deallocated when used, so to gain in
				     // efficiency.
    BlockVector<double> utility_xi;


				     // Number of degrees of freedom
				     // for each component of the
				     // system
    unsigned int n_dofs_u, n_dofs_p, n_dofs_up, n_dofs_W, n_total_dofs;
    
				     // This is the matrix that will be inverted when
				     // solving the problem.
    SparseDirectUMFPACK JF_inv;
    
    double scaling;

    double previous_time;
   
   /*-----------------------------------------*/
				    //The first dof pertaining to pressure field
   unsigned int constraining_dof;
				  //A container to store the dofs corresponding to the pressure field 
   set<unsigned int> pressure_dofs;
   

/*------------------------------------------*/
   /*------------------------------------------*/
					 //	A vector to store the contribution of the elastic stress
					 //of the solid 
   Vector <double> A_gamma; 
   Vector <double> M_gamma3_inv_A_gamma;
   
					 //The inverse of the mass matrix for the fluid domain
   SparseDirectUMFPACK M_gamma3_inv; 
					 //Mass matrix for the fluid domain
   SparseMatrix<double> M_gamma3; 
   
/*------------------------------------------*/

				  //Area of the control volume
   double area;

    std::ofstream global_info_file;


				     // The following functions define the inner workings of the
				     // class.
				     
				     // As typical, we need to define the appropriate
				     // triangulations and distribute the degree of freedom over
				     // them.  In this particular implementation, it is
				     // understood that two grids need to be provided, one for
				     // the container domain and the other for the immersed
				     // domain.  In both cases, the UCD format is used and the
				     // naming convention is as follows:
				     // "fluid_[dim]d.inp" for the container domain, and
				     // "solid_[dim]d.inp" for the immersed domain.
				     // In either case, [dim] and [dim] must match the
				     // the values of the parameters dim and dim,
				     // respectively.
    void create_triangulation_and_dofs ();

    void apply_constraints(vector<double> &local_res,
			   FullMatrix<double> &local_jacobian,
			   const Vector<double> &local_up,
			   const vector<unsigned int> &dofs);
//     
   void compute_current_bc(const double time);
   
   void apply_current_bc(BlockVector<double> &vec,
						 const double time);

				     // Build the sparsity of the
				     // system. 
    void assemble_sparsity(Mapping<dim, dim> &mapping);

    /*void assemble_zero_mean_value_constraints();*/
    void  get_area_and_first_pressure_dof();
    
    void residual_and_or_Jacobian(BlockVector<double> &residual,
				    BlockSparseMatrix<double> &Jacobian,
				    const BlockVector<double> &xit,
				    const BlockVector<double> &xi,
				    const double alpha,
				    const double t);

    
				     // Compute the Jacobian of the
				     // residual for time derivative
				     // xit and state xi. The final
				     // system is viewed as
				     // F(xit,xi,t)=0. The Jacobian
				     // that is assembled is
				     // alpha*dF/dxit+dF/dxi
//     void assemble_Jacobian(BlockVector<double> &residual,
// 			   BlockSparseMatrix<double> &Jacobian,
// 			   const BlockVector<double> &xit,
// 			   const BlockVector<double> &xi,
// 			   const double alpha,
// 			   const double t);


void distribute_residual(Vector<double> &residual,
						const std::vector<double> &local_res,
						const std::vector<unsigned int> &dofs_1,
						const unsigned int offset_1);
    
void distribute_jacobian(SparseMatrix<double> &Jacobian,
						const FullMatrix<double> &local_Jac,
						const std::vector<unsigned int> &dofs_1,
						const std::vector<unsigned int> &dofs_2,
						const unsigned int offset_1,
						const unsigned int offset_2);
 /*--------------------------------*/  
  void distribute_constraint_on_pressure(Vector<double> &residual,
										 const double average_pressure);
 
   void distribute_constraint_on_pressure( SparseMatrix<double> &jacobian,
										  const vector<double> &pressure_coefficient,
										  const vector<unsigned int> &dofs,
										  const unsigned int offset);
/*----------------------------------*/
   /*------------------------------------*/
   
   void localize (Vector<double> &local_M_gamma3_inv_A_gamma,
				  const Vector<double> &M_gamma3_inv_A_gamma,
				  const vector<unsigned int> &dofs);
   
   void get_Agamma_values( const FEValues<dim,dim> &fe_v_s,
						   const std::vector< unsigned int > &dofs,
						   const Vector<double> &xi,
						  Vector<double> &local_A_gamma);
						
void get_PFT_and_PFT_Dxi_values(
                const FESystem<dim,dim> &fe_s,
                const FEValues<dim,dim> &fe_v_s,
                const std::vector< unsigned int > &dofs,
                const Vector<double> &xi, 
                const bool update_jacobian, 
                std::vector<Tensor<2,dim,double> > &PFT,
                std::vector< std::vector<Tensor<2,dim,double> > > & PFT_Dxi);
/*----------------------------*/    
				     /** Interface to OdeArgument */
    
    unsigned int n_dofs() const {return n_total_dofs;};

    void output_step(const double t,
			     const Vector<double> &solution,
			     const unsigned int step_number,
			     const double h);

    template<class Type>
    inline void set_to_zero (Type &v) const;

    template<class Type>
    inline void set_to_zero (Table<2,Type> &v) const;

    template<class Type>
    inline void set_to_zero(std::vector<Type> &v) const;
    
    double norm(const std::vector<double> &v);


};


template <int dim>
ImmersedFEM<dim>::ImmersedFEM(ProblemParameters<dim> &par)
		:
		par(par),
		fe_f(FE_Q<dim>(par.degree), dim, FE_Q<dim>(par.degree-1), 1),/* Changed FE_DGP to FE_Q*/
		fe_s(FE_Q<dim, dim>(par.degree), dim),
		dh_f(tria_f),
		dh_s(tria_s),
		quad_f(par.degree+2),
		quad_s(par.degree+2)
{
  if(par.degree <= 1)
    std::cout << " WARNING: The chosen pair of finite element spaces is not stable."
	      << std::endl
	      << " The obtained results will be nonsense"
	      << std::endl;
  
  create_triangulation_and_dofs ();

  global_info_file.open((par.output_name+"_global.gpl").c_str());
}


template <int dim>
ImmersedFEM<dim>::~ImmersedFEM()
{
  delete mapping;
  global_info_file.close();
}

template <int dim>
void
ImmersedFEM<dim>::compute_current_bc(const double t)
{
   par.u_g.set_time(t);
   VectorTools::interpolate_boundary_values(MappingQ1<dim>(),//SR: Added this to overcome an error in newer versions of dealii
											dh_f,
											par.boundary_map,
											par.boundary_values,
											par.component_mask);
   
   // Find the first pressure dof
   if(par.fix_pressure == true)
   {
	  /*  vector<unsigned int> dofs(fe_f.dofs_per_cell);
       dh_f.begin_active()->get_dof_indices(dofs);
       unsigned int id=0;
	   
       while(true)
	   {
	   if(fe_f.system_to_component_index(id).first == dim) break;
	   else ++id;
	   }*/
	  par.boundary_values[constraining_dof] = 0;
   }
}

template <int dim>
void
ImmersedFEM<dim>::apply_current_bc(BlockVector<double> &vec,
						const double t)
{
  compute_current_bc(t);
  map<unsigned int, double>::iterator
    it    = par.boundary_values.begin(),
    itend = par.boundary_values.end();
    // NOTE: modify by removing use of constraints_f but adding an assert on the size of vec.
  if(vec.size() != 0)
    for(;it != itend; ++it)
      vec.block(0)(it->first) = it->second;
  else
    for(;it != itend; ++it)
      constraints_f.set_inhomogeneity(it->first, it->second);
}



				     // Functions dealing with the creation of the triangulation
				     // and distribution of the degrees of freedom.
				     // Two grids need to be provided, one for
				     // the container domain and the other for the immersed
				     // domain.  In both cases, the UCD format is used and the
				     // naming convention is as follows:
				     // "fluid_[dim]d.inp" for the container domain, and
				     // "solid_[dim]d.inp" for the immersed domain.
				     // In either case, [dim] and [dim] must match the
				     // the values of the parameters dim and dim,
				     // respectively.
template <int dim>
void
ImmersedFEM<dim>::
create_triangulation_and_dofs ()
{
				     // As specified in the documentation for the "GridIn" class
				     // the triangulation corresponding to a grid needs to be
				     // empty at this time.
  GridIn<dim> grid_in_f;
  grid_in_f.attach_triangulation (tria_f);

  {
    std::ifstream file (par.fluid_mesh.c_str());
    Assert (file, ExcFileNotOpen (par.fluid_mesh.c_str()));
    //  A grid in ucd format is expected. 
    grid_in_f.read_ucd (file);
  }
  
  GridIn<dim, dim> grid_in_s;
  grid_in_s.attach_triangulation (tria_s);

  {
    std::ifstream file (par.solid_mesh.c_str());
    Assert (file, ExcFileNotOpen (par.solid_mesh.c_str()));
    //  A grid in ucd format is expected. 
    grid_in_s.read_ucd (file);
  }

  std::cout << "Number of fluid refines = " << par.ref_f
	    << std::endl;
  tria_f.refine_global (par.ref_f);
  std::cout << "Number of active cells: " << tria_f.n_active_cells()
	    << std::endl;
 
  std::cout << "Number of solid refines = " << par.ref_s
	    << std::endl;
  tria_s.refine_global (par.ref_s);
  std::cout << "Number of active cells: " << tria_s.n_active_cells()
	    << std::endl;
				     // Initialization of the boundary_indicators vector
  boundary_indicators = tria_f.get_boundary_indicators();
  
				     // Distribution of the degrees of freedom.
				     // Both for the solid and fluid domains, the dofs are
				     // renumbered first globally and then by component.
  dh_f.distribute_dofs (fe_f);
  DoFRenumbering::boost::Cuthill_McKee (dh_f);

				   // Now renumber also velocity and
				   // pressure wise, and store the
				   // number of dofs.  This is due to the fact that
				   // we are storing the information pertaining to the various
				   // components of the system in block matrix.
  std::vector<unsigned int> block_component (dim+1,0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise (dh_f, block_component);

  std::vector<unsigned int> dofs_per_block (2);
  DoFTools::count_dofs_per_block (dh_f, dofs_per_block, block_component);

				     // Accounting of the number of degrees of freedom for the
				     // fluid domain on a block by block basis.
  n_dofs_u  = dofs_per_block[0];
  n_dofs_p  = dofs_per_block[1];
  n_dofs_up = dh_f.n_dofs();

				   // Simply distribute dofs on the
				   // solid displacement.
  dh_s.distribute_dofs (fe_s);
  DoFRenumbering::boost::Cuthill_McKee (dh_s);

  n_dofs_W = dh_s.n_dofs();
  n_total_dofs = n_dofs_up+n_dofs_W;

  std::cout << "dim (V_h) = " << n_dofs_u      << endl
	    << "dim (Q_h) = " << n_dofs_p      << endl
	    << "dim (Z_h) = " << dh_s.n_dofs() << endl
	    << "Total: "      << n_total_dofs        << endl;

  vector<unsigned int> all_dofs(2);
  all_dofs[0] = n_dofs_up;
  all_dofs[1] = n_dofs_W;

				     // Re-initialization of the BlockVectors containing the
				     // values of the degrees of freedom and of the residual.
     current_xi.reinit(all_dofs);
    previous_xi.reinit(all_dofs);
    current_xit.reinit(all_dofs);
    current_res.reinit(all_dofs);
  newton_update.reinit(all_dofs);
     utility_xi.reinit(all_dofs);

				     // Re-initialization of the average and unit pressure
				     // Vectors.
  pressure_average.reinit(n_dofs_up);
  unit_pressure.reinit(n_dofs_up);

  {
    constraints_f.clear ();
    constraints_s.clear ();
				     // Enforce hanging node constraints.
    DoFTools::make_hanging_node_constraints (dh_f, constraints_f);

    DoFTools::make_hanging_node_constraints (dh_s, constraints_s);
    
				     // The strategy used to solve the problem is such that
				     // one must assemble the Jacobian of the residual vector
				     // using zero boundary values for the velocity.
				     // The specification of the actual boundary values is
				     // done later by the "apply_current_bc" function. 
	 VectorTools::interpolate_boundary_values(MappingQ1<dim>(),//SR: Added this to overcome an error in newer versions of dealii
											 dh_f,
											 par.zero_boundary_map,
											 constraints_f,
											 par.component_mask);
  }
				     // The function assemble_zero_mean_value_constraints
				     // deals with the enforcement of the zero-average of the
				     // pressure.  In this particular example, the zero-pressure
				     // average constraint is not built into the constraint
				     // matrix.  The enforcement of the constraint in question
				     // is placed before closing the constraint matrix simply
				     // for "conceptual" reasons.  If the enforcement
				     // of the zero-average were built into the constraint
				     // matrix this constraint would have to be built before
				     // closing the constraint matrix.
  /* Instead of assemble_zero_mean_value_constraints(); we use a different function
   that calculates the area of the control volume as well as the first dof of the pressure*/
  get_area_and_first_pressure_dof();
   
  constraints_f.close();
  constraints_s.close();
  
  ConstraintMatrix cc; 
  cc.close(); // This matrix plays no part in the formulation.
              // It is defined here only in the eventuality that
              // the "VectorTools::project" function is used in the construction
              // in initializing the vectors previous_xi.block(0) and
              // unit_pressure.
              
              // Construction of the initial conditions.
  if(fe_f.has_support_points()) // The function fe_f.has_support_points()
                                // returns "true" only is every component of
                                // the FESystem fe_f has support points.
     {
       VectorTools::interpolate(dh_f, par.u_0, previous_xi.block(0));
       VectorTools::interpolate(dh_f, ComponentSelectFunction<dim>(dim, 1., dim+1), unit_pressure);
     }
   else
    {
      VectorTools::project(dh_f, cc, quad_f, par.u_0,
			   previous_xi.block(0));
				       // Construct unit pressure
      VectorTools::project(dh_f, cc, quad_f,
			   ComponentSelectFunction<dim>(dim, 1., dim+1), unit_pressure);
    }

  if(fe_s.has_support_points()) // The function fe_f.has_support_points()
                                // returns "true" only is every component of
                                // the FESystem fe_f has support points. 
    VectorTools::interpolate(dh_s, par.W_0, previous_xi.block(1));
  else
    VectorTools::project(dh_s, cc, quad_s, par.W_0,
			 previous_xi.block(1));
  
  mapping = new MappingQEulerian<dim, Vector<double>, dim>
	    (par.degree, previous_xi.block(1), dh_s);
    
  {

    BlockCompressedSimpleSparsityPattern csp (2,2);

    csp.block(0,0).reinit (n_dofs_up, n_dofs_up);
    csp.block(0,1).reinit (n_dofs_up, n_dofs_W );
    csp.block(1,0).reinit (n_dofs_W , n_dofs_up);
    csp.block(1,1).reinit (n_dofs_W , n_dofs_W );
    
    // As stated in the documentation, now we *must* call the function
    // csp.collect_sizes.() since have changed the size of the sub-objects of
    // the object csp.
    csp.collect_sizes();

    Table< 2, DoFTools::Coupling > coupling(dim+1,dim+1);
    for(unsigned int i=0; i<dim; ++i) 
      {
    // velocity is coupled with pressure.  
	coupling(i,dim) = DoFTools::always;
	// pressure is coupled with velocity.
	coupling(dim,i) = DoFTools::always;
	for(unsigned int j=0; j<dim; ++j)
	  // The velocity components are coupled with themselves and each other.
	  coupling(i,j) = DoFTools::always;
      }
    // The pressure is coupled with itself.
    coupling(dim, dim) = DoFTools::always;
    // Why not use coupling.fill(DoFTools::always) ?
    
	 /*---------------------------------------------------*/
	 set<unsigned int>::iterator it = pressure_dofs.begin();
	 constraining_dof = *it;
	 for(++it; it != pressure_dofs.end(); ++it)
	 {
		csp.block(0,0).add(constraining_dof, *it);//Just to create the proper sparsity pattern
	 }  
	 /*----------------------------------------------------*/
	 
    DoFTools::make_sparsity_pattern (dh_f, coupling,
				     csp.block(0,0),
				     constraints_f,
				     true);
    DoFTools::make_sparsity_pattern (dh_s, csp.block(1,1));

    sparsity.copy_from (csp);
    assemble_sparsity(*mapping);
  }
  JF.reinit(sparsity);

				   // Boundary conditions at t = 0.
  apply_current_bc(previous_xi, 0);
   /*-------------------------------------------------------------------------*/
      
   A_gamma.reinit(n_dofs_W);
   
   M_gamma3_inv_A_gamma.reinit(n_dofs_W);
   
				  //Creating the mass matrix for the solid domain and storing 
				  //its inverse
   ConstantFunction<dim> phi_b_func (par.Phi_B, dim);
   
   M_gamma3.reinit (sparsity.block(1,1));

				  //Using the dealii in-built functionality to create the mass 
				  //matrix
   MatrixCreator::create_mass_matrix (dh_s, quad_s, M_gamma3, &phi_b_func); 
   
   M_gamma3_inv.initialize (M_gamma3);

   /*-------------------------------------------------------------------------*/

}


template <int dim>
void
ImmersedFEM<dim>::
assemble_sparsity (Mapping<dim, dim> &immersed_mapping) 
{
  static Vector<double> tmp;
  tmp.reinit(n_dofs_up, true);
  FEFieldFunction<dim, DoFHandler<dim>, Vector<double> >
    up_field (dh_f, tmp);
  
  std::vector< typename DoFHandler<dim>::active_cell_iterator > cells;
  std::vector< std::vector< Point< dim > > > qpoints;
  std::vector< std::vector< unsigned int> > maps;
  std::vector< unsigned int > dofs_f(fe_f.dofs_per_cell);
  std::vector< unsigned int > dofs_s(fe_s.dofs_per_cell);

  typename DoFHandler<dim,dim>::active_cell_iterator
    cell = dh_s.begin_active(),
    endc = dh_s.end();

  FEValues<dim,dim> fe_v(immersed_mapping, fe_s, quad_s,
				  update_quadrature_points);

  CompressedSimpleSparsityPattern sp1(n_dofs_up, n_dofs_W);
  CompressedSimpleSparsityPattern sp2(n_dofs_W , n_dofs_up);
  
  for(; cell != endc; ++cell) 
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(dofs_s);
      up_field.compute_point_locations (fe_v.get_quadrature_points(),
					cells, qpoints, maps);
      for(unsigned int c=0; c<cells.size(); ++c) 
	{
	  cells[c]->get_dof_indices(dofs_f);
	  for(unsigned int i=0; i<dofs_f.size(); ++i)
	    for(unsigned int j=0; j<dofs_s.size(); ++j) 
	      {
		sp1.add(dofs_f[i],dofs_s[j]);
		sp2.add(dofs_s[j],dofs_f[i]);
	      }
	}
    }
  
  sparsity.block(0,1).copy_from(sp1);
  sparsity.block(1,0).copy_from(sp2);
}


template <int dim>
void
ImmersedFEM<dim>::
 get_area_and_first_pressure_dof()
{
   area = 0.0;
   typename DoFHandler<dim,dim>::active_cell_iterator
   cell = dh_f.begin_active(),
   endc = dh_f.end();
      
   FEValues<dim,dim> fe_v(fe_f, quad_f,
						  update_values |
						  update_JxW_values);
  
   std::vector<unsigned int> dofs_f(fe_f.dofs_per_cell);

   //Calculate the area of the control volume   
   for(; cell != endc; ++cell) 
   {
      fe_v.reinit(cell);
      cell->get_dof_indices(dofs_f);
	
      for(unsigned int i=0; i < fe_f.dofs_per_cell; ++i)
	  {
		 unsigned int comp_i = fe_f.system_to_component_index(i).first;
		 if(comp_i == dim)  pressure_dofs.insert(dofs_f[i]);

	  }
	  for(unsigned int q=0; q<quad_f.size(); ++q)
		 area += fe_v.JxW(q);
   }
   
   //Get the first dof pertaining to pressure
   constraining_dof = *(pressure_dofs.begin());

}
/*
template <int dim>
void
ImmersedFEM<dim>::
assemble_zero_mean_value_constraints () 
{
  typename DoFHandler<dim,dim>::active_cell_iterator
    cell = dh_f.begin_active(),
    endc = dh_f.end();

  pressure_average = 0;

  FEValues<dim,dim> fe_v(fe_f, quad_f,
			 update_values |
			 update_JxW_values);
  
  std::vector<unsigned int> dofs_f(fe_f.dofs_per_cell);
  Vector<double> local_C(fe_f.dofs_per_cell);
  
  unsigned int comp_i = 0;
  double area = 0;
  
  for(; cell != endc; ++cell) 
    {
      local_C = 0;
      fe_v.reinit(cell);
      cell->get_dof_indices(dofs_f);
				       // Compute the coefficients for
				       // the integral of the pressure
      for(unsigned int i=0; i<fe_f.dofs_per_cell; ++i)
	{
	  comp_i = fe_f.system_to_component_index(i).first;
	  if(comp_i == dim) 
	    {
	      for(unsigned int q=0; q<quad_f.size(); ++q)
		local_C(i) += fe_v.shape_value(i,q)*fe_v.JxW(q);

	    }
	}
      
      for(unsigned int q=0; q<quad_f.size(); ++q)
	area += fe_v.JxW(q);
      
      for(unsigned int i=0;i<fe_f.dofs_per_cell;++i)
	pressure_average(dofs_f[i]) += local_C(i);
    }
  pressure_average /= area;
}  
*/

// template <int dim>
// void
// ImmersedFEM<dim>::residual(const double t,
// 					BlockVector<double> &residual,
// 					const BlockVector<double> &xit_,
// 					const BlockVector<double> &xi_) 
// {
//   static BlockSparseMatrix<double> dummy;
//   assemble_residual_template<double>(residual, dummy, xit_, xi_, 0, t);
// }


// template <int dim>
// void
// ImmersedFEM<dim>::assemble_Jacobian(BlockVector<double> &residual,
// 						 BlockSparseMatrix<double> &Jacobian,
// 						 const BlockVector<double> &xit_,
// 						 const BlockVector<double> &xi_, 
// 						 const double alpha,
// 						 const double t) 
// {
//   assemble_residual_template<SacDouble>(residual, Jacobian,
// 					xit_, xi_, alpha, t);
// }
// 
// template <int dim>
// int ImmersedFEM<dim>::residual(const double t, 
// 					    Vector<double> &dst,  
// 					    const Vector<double> &src_yy,
// 					    const Vector<double> &src_yp) 
// {
//   xi = src_yy;
//   xit = src_yp;
//   
//   residual(t, current_res, xit, xi);
//   dst = current_res;
//   
//   cout << "t = " << t << ", R: " << dst.l2_norm() << std::endl;
//   
//   return 0;
// }

// template <int dim>
// int ImmersedFEM<dim>::setup_jacobian_prec(const double t, 
// 						       const Vector<double> &src_yy,
// 						       const Vector<double> &src_yp,
// 						       double alpha) 
// {
//   xi = src_yy;
//   xit = src_yp;
//   assemble_Jacobian(current_res, JF, xit, xi, alpha, t);
//   
//   try 
//     {
//       JF_inv.initialize(JF);
//     }
//   catch(...)
//     {
//       return -1;
//     }
//   cout << "J Assembled. " << std::endl;
//   return 0;
// }
  
  
template <int dim>
void ImmersedFEM<dim>::residual_and_or_Jacobian(BlockVector<double> &residual,
							  BlockSparseMatrix<double> &jacobian,
							  const BlockVector<double> &xit,
							  const BlockVector<double> &xi, 
							  const double alpha,
							  const double t) 
{
    bool update_jacobian = !jacobian.empty();
    
    if(mapping != NULL) delete mapping;

					// In a semi-implicit scheme, the position of the immersed
					// body coincides with the position of the body at the
					// previous time step.
    if(par.semi_implicit == true) 
     {
       if(std::fabs(previous_time - t) > 1e-12) // If at the very beginning of the current time step. (Not needed when using the implicit Euler Method written in run().)
        {
          previous_time = t;
          previous_xi = xi;
        }
        mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree, previous_xi.block(1), dh_s);
     }
    else
     mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree, xi.block(1), dh_s);

					// For the purpose of applying the boundary conditions, we
					// set a scaling factor to be equal to the diameter of the
					// first cell in the triangulation.
    scaling = dh_f.begin_active()->diameter();

					// Initialization of the residual.
    residual = 0;
					// If the calculation of the residual is requested, then
					// the Jacobian is initialized.
    if(update_jacobian)
     {
       jacobian.clear();
       assemble_sparsity(*mapping);
       jacobian.reinit(sparsity);
     }
  

					// Evaluation of the current values of the external force
					// and of the boundary conditions.
  par.force.set_time(t);
  compute_current_bc(t);

				   // Computation of the maximum number of degrees of freedom
				   // one could have on a "fluid-solid interaction cell".
				   // RATIONALE: the coupling of the fluid and solid domains is
				   // computed by finding each of the fluid cells that interact
				   // with a given solid cell. In each "interaction instance" we
				   // will be dealing with a total number of degrees of freedom
				   // that is the sum of the dofs of the "current" solid cell
				   // and the dofs of the "current" fluid cell in the list of
				   // fluid cells interacting with the solid cell in question.
  unsigned int n_local_dofs = fe_f.dofs_per_cell + fe_s.dofs_per_cell;

  // Storage for the local dofs in the fluid and in the solid, respectively.
  std::vector< unsigned int > dofs_f(fe_f.dofs_per_cell);
  std::vector< unsigned int > dofs_s(fe_s.dofs_per_cell);

  // FEValues for the fluid.
  FEValues<dim> fe_f_v(fe_f,
                       quad_f,
                       update_values |
                       update_gradients |
                       update_JxW_values |
                       update_quadrature_points);

  // Number of quadrature points on fluid and solid cells, respectively.
  const unsigned int nqpf = quad_f.size();
  const unsigned int nqps = quad_s.size();

  // The local residual vector.  The largest possible size this vector is
  // n_local_dofs.
  std::vector<double> local_res(n_local_dofs);
  std::vector<Vector<double> > local_force(nqpf, Vector<double>(dim+1));
  FullMatrix<double> local_jacobian;
  if(update_jacobian) local_jacobian.reinit(n_local_dofs, n_local_dofs); 

// -------------------------------------
// Since we want to solve a system of equations of the form
// F(xi', xi, t) = 0, we need to manage the information in
// xi' as though it were independent of the information in
// xi. We do so by defining a vector of local degrees of
// freedom that has a length equal to twice the total number
// of local degrees of freedom.  This information is stored
// in the Type vector "local_x".
// * The first fe_f.dofs_per_cell elements of local_x contain
//   the elements of xi' corresponding to the current fluid
//   cell.
// * The subsequent fe_s.dofs_per_cell elements of local_x
//   contain the elements of xi' corresponding to the
//   current solid cell.
// * The subsequent fe_f.dofs_per_cell elements of local_x
//   contain the elements of xi corresponding to the
//   current fluid cell.
// * The subsequent fe_s.dofs_per_cell elements of local_x
//   contain the elements of xi corresponding to the
//   current fluid cell.
//   std::vector<double> local_x(2*n_local_dofs);
// 
// Definition of the local dependent variables for the fluid.
  std::vector<Vector<double> > local_upt(nqpf, Vector<double>(dim+1));
  std::vector<Vector<double> > local_up (nqpf, Vector<double>(dim+1));
  std::vector< std::vector< Tensor<1,dim> > > local_grad_up(nqpf, std::vector< Tensor<1,dim> >(dim+1));

  unsigned int comp_i = 0, comp_j = 0;
   
   /*-----------------------------------------------------------------------*/
   double local_average_pressure = 0.0;
   vector<double> local_pressure_coefficient(n_local_dofs);
   /*------------------------------------------------------------------------*/
  
// ------------------------------------------------------------
// OPERATORS DEFINED OVER THE ENTIRE DOMAIN: BEGIN 
// ------------------------------------------------------------
// ============================================================
// We now determine the contribution to the residual due to the
// fluid.  This is the standard Navier-Stokes component of the 
// problem.  As such, the contributions are to the equation in
// V' and to the equation in Q'.
// ============================================================
  
  // These iterators point to the first and last active cell of
  // the fluid domain.
  typename DoFHandler<dim>::active_cell_iterator
    cell = dh_f.begin_active(),
    endc = dh_f.end();

  // Cycle over the cells of the fluid domain.
  for(; cell != endc; ++cell) 
    {
      cell->get_dof_indices(dofs_f);
      
      // Re-initialization of the FEValues.
      fe_f_v.reinit(cell);
      
      // Values of the partial derivative of the velocity relative to time
      // at the quadrature points on the current fluid cell.
      // Strictly speaking, this vector also includes values of the partial
      // derivative of the pressure with respect to time.
      fe_f_v.get_function_values(xit.block(0), local_upt);
      // Values of the velocity at the quadrature points on the current fluid
      // cell. Strictly speaking, this vector also includes values of pressure.
      fe_f_v.get_function_values(xi.block(0), local_up);
      // Values of the gradient of the velocity at the quadrature points of
      // the current fluid cell.
      fe_f_v.get_function_gradients(xi.block(0), local_grad_up);
      // Values of the body force at the quadrature points of the current fluid
      // cell.
      par.force.vector_value_list(fe_f_v.get_quadrature_points(), local_force);

      // Initialization of the local residual and local jacobian.
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);
	   
	   /*------------------------------------------------*/
	   local_average_pressure = 0.0;
	   set_to_zero(local_pressure_coefficient);
	   /*------------------------------------------------*/
	   
      for(unsigned int i=0; i<fe_f.dofs_per_cell;++i) 
        {
          comp_i = fe_f.system_to_component_index(i).first;
          for(unsigned int q=0; q< nqpf; ++q) 
				   // ------------------------------------ 
				   // Contribution to the equation in V'.
				   // ------------------------------------
          if(comp_i < dim) 
           {
				   // rho (ut - b).v - p div v
             local_res[i] += par.rho
                           * ( local_upt[q](comp_i)
                              -
                               local_force[q](comp_i) )
                           * fe_f_v.shape_value(i,q)
                           * fe_f_v.JxW(q)
                           - local_up[q](dim)
                           * fe_f_v.shape_grad(i,q)[comp_i]
                           * fe_f_v.JxW(q);
             if(update_jacobian) 
             {
               for(unsigned int j=0; j<fe_f.dofs_per_cell;++j)
                 {
                   comp_j = fe_f.system_to_component_index(j).first;
                   if( comp_i == comp_j )
                     local_jacobian(i,j) += par.rho
                                          * alpha
                                          * fe_f_v.shape_value(i,q)
                                          * fe_f_v.shape_value(j,q)
                                          * fe_f_v.JxW(q);
                   if( comp_j == dim )
                     local_jacobian(i,j) -= fe_f_v.shape_grad(i,q)[comp_i]
                                          * fe_f_v.shape_value(j,q)
                                          * fe_f_v.JxW(q);
                 }
             }
				 // T_{f} . grad_v + rho (grad_u u) . v
				 // where T_{f} = 2 \eta (1/2) (L + L^{T})
          for(unsigned int d=0; d<dim; ++d)
            {
              local_res[i] += par.eta
                            * ( local_grad_up[q][comp_i][d]
                               +
                                local_grad_up[q][d][comp_i] )
                            * fe_f_v.shape_grad(i,q)[d]
                            * fe_f_v.JxW(q)
                            + par.rho
                            * local_grad_up[q][comp_i][d]
                            * local_up[q](d) 
                            * fe_f_v.shape_value(i,q)
                            * fe_f_v.JxW(q);
            }
          if( update_jacobian )
           {
             for(unsigned int j=0; j<fe_f.dofs_per_cell;++j)
              {
                comp_j = fe_f.system_to_component_index(j).first;
                if( comp_j == comp_i )
                 for( unsigned int d = 0; d < dim; ++d )
                   local_jacobian(i,j) += par.eta 
                                        * fe_f_v.shape_grad(i,q)[d]
                                        * fe_f_v.shape_grad(j,q)[d]
                                        * fe_f_v.JxW(q)
                                        + par.rho
                                        * fe_f_v.shape_value(i,q)
                                        * local_up[q](d)
                                        * fe_f_v.shape_grad(j,q)[d]
                                        * fe_f_v.JxW(q);
                if(comp_j < dim)
                  local_jacobian(i,j) += par.eta
                                       * fe_f_v.shape_grad(i,q)[comp_j]
                                       * fe_f_v.shape_grad(j,q)[comp_i]
                                       * fe_f_v.JxW(q)
                                       + par.rho
                                       * local_grad_up[q][comp_i][comp_j]
                                       * fe_f_v.shape_value(i,q)
                                       * fe_f_v.shape_value(j,q)
                                       * fe_f_v.JxW(q);
              }
           }
           }
          else
           {
				 // ------------------------------------ 
				 // Contribution to the equation in Q'.
				 // ------------------------------------
				 // - q div u
             for(unsigned int d=0; d<dim;++d)
               local_res[i] -= local_grad_up[q][d][d]
                             * fe_f_v.shape_value(i,q)
                             * fe_f_v.JxW(q);
             if( update_jacobian )
              for(unsigned int j=0; j<fe_f.dofs_per_cell;++j)
               {
                 comp_j = fe_f.system_to_component_index(j).first;
                 if( comp_j < dim )
                  local_jacobian(i,j) -= fe_f_v.shape_value(i,q)
                                       * fe_f_v.shape_grad(j,q)[comp_j]
                                       * fe_f_v.JxW(q);
               }
			  /*----------------------------------------------------*/
			  if (par.all_DBC && !par.fix_pressure)
			  {
				 local_average_pressure += xi.block(0)(dofs_f[i])
										  *fe_f_v.shape_value(i,q)
										  *fe_f_v.JxW(q);
				 if(update_jacobian)
					local_pressure_coefficient[i] += fe_f_v.shape_value(i,q)*fe_f_v.JxW(q)
											 *scaling/area;
			  }
			 /*-----------------------------------------------------*/ 
           }
        }
				 // Apply boundary conditions.
      apply_constraints(local_res,
                        local_jacobian,
                        xi.block(0),
                        dofs_f);
				 // Now the contribution to the residual due to the current cell
				 // is assembled into the global system's residual.
      distribute_residual(residual.block(0), local_res, dofs_f, 0);
      if( update_jacobian ) distribute_jacobian(JF.block(0,0),
                                                local_jacobian,
                                                dofs_f,
                                                dofs_f,
                                                0,
                                                0);
	   
	   if(par.all_DBC && !par.fix_pressure) 
	   {
		  distribute_constraint_on_pressure(residual.block(0),
											local_average_pressure);
		  
		  if (update_jacobian) distribute_constraint_on_pressure(jacobian.block(0,0),
																 local_pressure_coefficient,
																 dofs_f,
																 0);
	   }
    }

// ------------------------------------------------------------
// OPERATORS DEFINED OVER ENTIRE DOMAIN: END 
// ------------------------------------------------------------

// ------------------------------------------------------------
// OPERATORS DEFINED OVER THE IMMERSED DOMAIN: BEGIN 
// ------------------------------------------------------------
// All but one of the operators defined over the immersed domain pertain
// to the interaction between solid and fluid.  The one contribution that does
// not pertain to this interaction is assembled last.  With this in mind, we
// distinguish two orders or organization.
//
// FIRST: we have a cycle over the cells of the immersed domain.  For
//        each cell of the immersed domain we determine the cells in the
//        fluid domain interacting with the cell in question.  Then we cycle
//        over each of the fluid cell.
// SECOND: The operators defined over the immersed domain contribute to
//         all three of the equations forming the problem.  We group
//         the operators in question by equation.  Specifically, we 
//         first deal with the terms that contribute to the equation in
//         V', then we deal with the terms that contribute to Q',
//         and finally we deal with the terms that contribute to Y'.
// NOTE: In the equation in Y' there is contribution that does not arise
//       from the interaction of solid and fluid. 

// Setting up the infrastructure to store the information
// needed to characterize the fluid-solid interaction.

  // This is a field containing the values of velocity and pressure.
  FEFieldFunction<dim, DoFHandler<dim>, Vector<double> >
    up_field (dh_f, xi.block(0));

  // Containers needed to store the information pertaining to the interaction
  // of the current solid cell with the corresponding set of fluid cells that
  // happen to contain the quadrature points of the solid cell in question.
  std::vector< typename DoFHandler<dim>::active_cell_iterator > fluid_cells;
  std::vector< std::vector< Point< dim > > > fluid_qpoints;
  std::vector< std::vector< unsigned int> > fluid_maps;

  // Local storage of the
  //  * velocity in the solid: Wt;
  //  * displacement in the solid: W;
  //  * P F^{T}, which is the work conjugate of the velocity gradient when
  //    measured over the reference configuration: PFT
  //  * Frechet derivative of PFT with respect to degrees of freedom in a solid
  //    cell: PFT_Dxi.
  std::vector<Vector<double> > local_Wt(nqps, Vector<double>(dim));
  std::vector<Vector<double> > local_W (nqps, Vector<double>(dim));
  std::vector<Tensor<2,dim,double> > PFT(nqps, Tensor<2,dim,double>());
  std::vector< std::vector<Tensor<2,dim,double> > > PFT_Dxi;
  if( update_jacobian )
  {
	 //PFT.resize(nqps, Tensor<2,dim,double>());
	 PFT_Dxi.resize(nqps, std::vector< Tensor<2,dim,double> >(fe_s.dofs_per_cell));
  }
   /*-------------------------------------------------------------------------*/
   A_gamma = 0.0;
   Vector<double> local_A_gamma (fe_s.dofs_per_cell);
  
   Vector<double> local_M_gamma3_inv_A_gamma(fe_s.dofs_per_cell);
   /* ------------------------------------------------------------------------*/

  // This information is used in finding what fluid cell contain the solid
  // domain at the current time.
  FEValues<dim,dim> fe_v_s_mapped(*mapping, fe_s, quad_s,
                                  update_quadrature_points);

  // FEValues needed to carry out integrations over the solid domain.
  FEValues<dim,dim> fe_v_s(fe_s,
                           quad_s,
                           update_quadrature_points |
                           update_values |
                           update_gradients |
                           update_JxW_values);

  
  // Iterators pointing to the beginning and end cells of the active
  // triangulation for the solid domain.
  typename DoFHandler<dim,dim>::active_cell_iterator
    cell_s = dh_s.begin_active(),
    endc_s = dh_s.end();
   
   /*-------------------------------------------------------------------------*/
				  //First we cycle over the cells of the solid domain to evaluate
				  //A_gamma and M_gamma3_inv_A_gamma
   
   //Begin the loop over cells to calculate A_gamma
   if(par.mu != 0 )
   {
	  for (; cell_s != endc_s; ++cell_s)
	  {
		 fe_v_s.reinit(cell_s);
		 cell_s->get_dof_indices(dofs_s);
     
		 get_Agamma_values(fe_v_s, dofs_s, xi.block(1),local_A_gamma);
		 
		 A_gamma.add (dofs_s, local_A_gamma);
	  }//end cell_s-loop over solid cells 
   
	  M_gamma3_inv_A_gamma = A_gamma;//Use of 'M_gamma3_inv_A_gamma' is optional
	  M_gamma3_inv.solve (M_gamma3_inv_A_gamma);
   }
   /*--------------------------------------------------------------------------*/
   

  // -----------------------------------------------
  // Cycle over the cells of the solid domain: BEGIN
  // -----------------------------------------------
  for(cell_s = dh_s.begin_active(); cell_s != endc_s; ++cell_s)
    {
      fe_v_s_mapped.reinit(cell_s);
      fe_v_s.reinit(cell_s);
      cell_s->get_dof_indices(dofs_s);

// 						  // Localization of the current independent variables
// 						  // pertaining to the solid.
      fe_v_s.get_function_values(xit.block(1), local_Wt);
      fe_v_s.get_function_values( xi.block(1), local_W);
      if(par.mu != 0 )
	  {
		 localize (local_M_gamma3_inv_A_gamma, M_gamma3_inv_A_gamma, dofs_s);
		// if( update_jacobian)
		 get_PFT_and_PFT_Dxi_values(fe_s, fe_v_s, dofs_s, xi.block(1), update_jacobian, PFT, PFT_Dxi);
	  }

				 // Coupling between fluid and solid.
				 // Identification of the fluid cells containing the
				 // quadrature points on the current solid cell.
      up_field.compute_point_locations(fe_v_s_mapped.get_quadrature_points(),
                                       fluid_cells,
                                       fluid_qpoints,
                                       fluid_maps);

      local_force.resize(nqps, Vector<double>(dim+1));
      par.force.vector_value_list(fe_v_s_mapped.get_quadrature_points(),
                                  local_force);
      
      
      // Cycle over all of the fluid cells that happen to contain some of the 
      // the quadrature points of the current solid cell.
      for(unsigned int c=0; c<fluid_cells.size(); ++c) 
	{
	  fluid_cells[c]->get_dof_indices(dofs_f);

				 // Local FEValues of the
				 // fluid
      Quadrature<dim> local_quad(fluid_qpoints[c]);
      FEValues<dim> local_fe_f_v(fe_f, 
                                 local_quad,
                                 update_values |
                                 update_gradients |
                                 update_hessians);
      local_fe_f_v.reinit(fluid_cells[c]);

				 // Construction of the values at the quadrature points
				 // of the current solid cell of the velocity of the fluid.
      local_up.resize (local_quad.size(), Vector<double>(dim+1));
      local_fe_f_v.get_function_values(xi.block(0), local_up);

// ============================================================
// Equation in V': Here we assemble all of the terms in the
//                 equation in V' that are defined over B.
// ------------------------------------------------------------
// Equation in Q': Here we assemble all of the terms in the 
//                 equation in Q' that are defined over B. 
//                 There are two such terms, both pertaining to
//                 the solution of the compressible case.
//                 These terms concern the removal of the
//                 incompressibility condition over B and the
//                 construction of a penalty term specifying 
//                 the value of the pressure over B.
// ------------------------------------------------------------
// Equation in Y': Here we assemble all of the terms in the
//                 equation in Y' that involve the velocity u.
// ============================================================
// NOTE: Since the FEValues pertaining to the equations in V'
//       and Q' are grouped together, the terms contributing
//       to the equations in V' and Q', all of these terms
//       are assembled in a single cycle over the dofs of the
//       current fluid cell.
// ============================================================


// ****************************************************
// Equation in V' and equation in Q'
// initialization of residual 
// ****************************************************
if(par.mu != 0){
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);
// ****************************************************
// Equation in V', Equation in Q'
// begin cycle over fluid dofs
// ****************************************************	  
      for(unsigned int i=0; i<fe_f.dofs_per_cell;++i)
        {
          comp_i = fe_f.system_to_component_index(i).first;

          if(comp_i < dim) // Equation in V'
           for(unsigned int q=0; q<local_quad.size(); ++q)
             {
				 // Quadrature point on the *mapped* solid (Bt)
               unsigned int &qs = fluid_maps[c][q];
				
				 // Contribution due to the elastic component
				 // of the stress response function in the solid.
				// P F^{T} . grad_x v
              /* local_res[i] += (PFT[qs][comp_i] * local_fe_f_v.shape_grad(i,q))
                             * fe_v_s.JxW(qs);
               if( update_jacobian ) // Recall that the Hessian is symmetric.
			   {
                  for( unsigned int j = 0; j < fe_s.dofs_per_cell; ++j )
                    {
					   unsigned int wj = j + fe_f.dofs_per_cell;
					   unsigned int comp_j =
                                       fe_s.system_to_component_index(j).first;
					  
						local_jacobian(i,wj) 
						   += ( PFT_Dxi[qs][j][comp_i]
                           *
                            local_fe_f_v.shape_grad(i,q) )
						   * fe_v_s.JxW(qs);
						  if( !par.semi_implicit )
							 local_jacobian(i,wj)
							  += ( PFT[qs][comp_i]
								  *
								  local_fe_f_v.shape_hessian(i,q)[comp_j])
								 * fe_v_s.shape_value(j,qs)
								 * fe_v_s.JxW(qs);
					   }
                }*/
				/*------------------------------------------------------------*/
				for( unsigned int j = 0; j < fe_s.dofs_per_cell; ++j )//Spread
				{
				   local_res[i] += 
				   local_fe_f_v.shape_value(i, q)
				   *fe_v_s.shape_value(j, qs)
				   * local_M_gamma3_inv_A_gamma(j)
				   * fe_v_s.JxW(qs);
				   
				   if( update_jacobian ) 
				   {
					  unsigned int wj = j + fe_f.dofs_per_cell;
					  unsigned int comp_j =
					  fe_s.system_to_component_index(j).first;
					  
					  local_jacobian(i,wj) 
					  += ( PFT_Dxi[qs][j][comp_i]
						  *
						  local_fe_f_v.shape_grad(i,q) )
					  * fe_v_s.JxW(qs);
					  
					  if( !par.semi_implicit )
						 local_jacobian(i,wj)
						 += ( PFT[qs][comp_i]
							 *
							 local_fe_f_v.shape_hessian(i,q)[comp_j])
						 * fe_v_s.shape_value(j,qs)
						 * fe_v_s.JxW(qs);
				   }
				}//Spread
			   
				/*------------------------------------------------------------*/
             }
        }

// 
// ****************************************************
// Equation in V' add to global residual
// ****************************************************
      apply_constraints(local_res,
                        local_jacobian,
                        xi.block(0),
                        dofs_f);
      distribute_residual(residual.block(0),
                          local_res,
                          dofs_f,
                          0);
      if( update_jacobian ) distribute_jacobian(JF.block(0,1),
                                                local_jacobian,
                                                dofs_f,
                                                dofs_s,
                                                0,
                                                fe_f.dofs_per_cell);
}                             
// ****************************************************
// Equation in V': COMPLETED
// Equation in Y': NOT YET COMPLETED
// ****************************************************

// ****************************************************
// Equation in Y'
// initialization of residual 
// ****************************************************
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);
// ****************************************************
// Equation in Y'
// begin cycle over solid dofs
// ****************************************************	  
      for(unsigned int i=0; i<fe_s.dofs_per_cell;++i) 
        {
          unsigned int wi = i + fe_f.dofs_per_cell;
          comp_i = fe_s.system_to_component_index(i).first;
          for(unsigned int q=0; q<local_quad.size(); ++q)
            {
              unsigned int &qs = fluid_maps[c][q];
				 // - u . y
              local_res[wi] -= par.Phi_B 
                             * local_up[q](comp_i)
                             * fe_v_s.shape_value(i,qs)
                             * fe_v_s.JxW(qs);
              if( update_jacobian )
               for(unsigned int j = 0; j < fe_f.dofs_per_cell; ++j)
                 {
                   comp_j = fe_f.system_to_component_index(j).first;
                   if( comp_i == comp_j )
                    {
                      local_jacobian(wi,j) -= par.Phi_B 
                                            * fe_v_s.shape_value(i,qs)
                                            * local_fe_f_v.shape_value(j,q)
                                            * fe_v_s.JxW(qs);
                      if( !par.semi_implicit )
                      for(unsigned int k = 0; k < fe_s.dofs_per_cell; ++k)
                        {
                          unsigned int wk = k + fe_f.dofs_per_cell;
                          unsigned int comp_k = fe_s.system_to_component_index(k).first;
                          local_jacobian(wi,wk)
                             -= par.Phi_B 
                              * fe_v_s.shape_value(i,qs)
                              * fe_v_s.shape_value(k,qs)
                              * local_fe_f_v.shape_grad(j,q)[comp_k]
                              * xi.block(0)(dofs_f[j])
                              * fe_v_s.JxW(qs);
                        }
                    }
                 }
            }
        }

// ****************************************************
// Equation in Y'
// add to global residual
// ****************************************************
      apply_constraints(local_res,
                        local_jacobian,
                        xi.block(0),
                        dofs_f);
      distribute_residual(residual.block(1),
                          local_res,
                          dofs_s,
                          fe_f.dofs_per_cell);
      if( update_jacobian )
       {
         distribute_jacobian(JF.block(1,0),
                             local_jacobian,
                             dofs_s,
                             dofs_f,
                             fe_f.dofs_per_cell,
                             0);
         if( !par.semi_implicit ) distribute_jacobian(JF.block(1,1),
                                                      local_jacobian,
                                                      dofs_s,
                                                      dofs_s,
                                                      fe_f.dofs_per_cell,
                                                      fe_f.dofs_per_cell);
       }

	  
// ****************************************************
// Equation in V': COMPLETED
// Equation in Q': COMPLETED
// Equation in Y': COMPLETED
// ****************************************************	  
    }
//       
// // ------------------------------------------------------------
// // Equation in Y': Here we assemble all of the terms in the
// //                 equation in Y' that involve w_dot.
//       // As always, we start by initializing the contribution to the
//       // local residual.
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);
      
      // begin cycle over solid dofs
      for(unsigned int i=0; i<fe_s.dofs_per_cell; ++i) 
    {
      comp_i = fe_s.system_to_component_index(i).first;
      unsigned int wi = i + fe_f.dofs_per_cell;
      for(unsigned int qs=0; qs<nqps; ++qs) 
        {
				// Wt . Y
          local_res[wi] += par.Phi_B
                         * local_Wt[qs](comp_i)
                         * fe_v_s.shape_value(i,qs)
                         * fe_v_s.JxW(qs);
          if( update_jacobian )
           for(unsigned int j=0; j<fe_s.dofs_per_cell; ++j)
             {
               comp_j = fe_s.system_to_component_index(j).first;
               unsigned int wj = j + fe_f.dofs_per_cell;
               if( comp_i == comp_j )
                local_jacobian(wi,wj) += par.Phi_B
                                       * alpha
                                       * fe_v_s.shape_value(i,qs)
                                       * fe_v_s.shape_value(j,qs)
                                       * fe_v_s.JxW(qs);
             }

	    }
	}
      // As always, now assemble the contribution just computed into the
      // global residual.
      distribute_residual(residual.block(1),
                          local_res,
                          dofs_s,
                          fe_f.dofs_per_cell);
      if( update_jacobian ) distribute_jacobian(JF.block(1,1),
                                                local_jacobian,
                                                dofs_s,
                                                dofs_s,
                                                fe_f.dofs_per_cell,
                                                fe_f.dofs_per_cell);
      
 } // Cycle over the cells of the solid domain: END

// ------------------------------------------------------------
// OPERATORS DEFINED OVER THE IMMERSED DOMAIN: END 
// ------------------------------------------------------------
 
}
    

// template <int dim>
// template<class Type>
// void
// ImmersedFEM<dim>::localize_fields(std::vector<std::vector<Type> >&local_ut,
// 						std::vector<std::vector<Type> >&local_u,
// 						std::vector<Table<2,Type> >&local_grad_u,
// 						std::vector<Type >&local_p,
// 						const FEValues<dim> &fe_f_v,
// 						const std::vector<Type> &local_x) const 
// {
//   set_to_zero(local_ut);
//   set_to_zero(local_u);
//   set_to_zero(local_grad_u);
//   set_to_zero(local_p);
// 
//   const unsigned int n_local_dofs = fe_f.dofs_per_cell+fe_s.dofs_per_cell;
// 
//   const Quadrature<dim> &quad = fe_f_v.get_quadrature();
// 				   // Definition of the dependent
// 				   // variables local_u and local_p as
// 				   // a function of the the
// 				   // independent variables
// 				   // local_xin1.
//   unsigned int comp_i = 0;
//   for(unsigned int i=0; i<fe_f.dofs_per_cell; ++i)
//     {
//       comp_i = fe_f.system_to_component_index(i).first;
//       if(comp_i < dim)
// 	for(unsigned int q=0; q<quad.size(); ++q) 
// 	  {
// 	    local_ut[q][comp_i] += local_x[i]*fe_f_v.shape_value(i,q);
// 	    local_u[q][comp_i] += local_x[i+n_local_dofs]*fe_f_v.shape_value(i,q);
// 
// 	    for(unsigned int j=0; j<dim;++j)
// 	      local_grad_u[q](comp_i,j) +=
// 		fe_f_v.shape_grad(i,q)[j]*local_x[i+n_local_dofs];
// 	  }
//       else
// 	{
// 	  for(unsigned int q=0; q<quad.size(); ++q) 
// 	    local_p[q] += fe_f_v.shape_value(i,q)*local_x[i+n_local_dofs];
// 	}
//     }
// }
// 
// 
// template <int dim>
// template<class Type>
// void
// ImmersedFEM<dim>::localize_fields(std::vector<std::vector<Type> >&local_Wt,
// 						std::vector<std::vector<Type> >&local_W,
// 						std::vector<Table<2,Type> >&local_F,
// 						std::vector<Table<2,Type> >&local_F_dot,
// 						std::vector<Table<2,Type> >&local_F_inv,
// 						std::vector<Type>&local_J,
// 						const FEValues<dim,dim> &fe_v_s,
// 						const std::vector<Type> &local_x) const 
// {
//   
// 				       // Initialization of local_Wt, local_W, local_F
//   set_to_zero(local_Wt);
//   set_to_zero(local_W);
//   set_to_zero(local_F);
//   set_to_zero(local_F_dot);
//   set_to_zero(local_F_inv);
//   set_to_zero(local_J);  
//   
//   const unsigned int n_local_dofs = fe_f.dofs_per_cell+fe_s.dofs_per_cell;
// 
//   const Quadrature<dim> &quad = fe_v_s.get_quadrature();
// 				   // Definition of the dependent variables local_u and local_p
// 				   // as a function of the the
// 				   // independent variables
// 				   // local_xin1.
//   unsigned int comp_i=0;
//   
//   for(unsigned int q=0; q<quad.size(); ++q) 
//     {
//       for(unsigned int i=0; i<fe_s.dofs_per_cell; ++i)
// 	{
// 	  comp_i = fe_s.system_to_component_index(i).first;
// 	  unsigned int wi=i+fe_f.dofs_per_cell;
// 	  
// 	  local_Wt[q][comp_i] += local_x[wi]*fe_v_s.shape_value(i,q);
// 	  local_W[q][comp_i] += local_x[wi+n_local_dofs]*fe_v_s.shape_value(i,q);
// 	  
// 	  for(unsigned int j=0; j<dim;++j) 
// 	    {
// 	      local_F[q](comp_i,j) +=
// 		fe_v_s.shape_grad(i,q)[j]*local_x[wi+n_local_dofs];
// 	      local_F_dot[q](comp_i,j) +=
// 		fe_v_s.shape_grad(i,q)[j]*local_x[wi];
// 	    }
// 	}
//       for(unsigned int d=0; d<dim; ++d)
// 	local_F[q](d,d) += 1;
//       Table<2, Type> & F = local_F[q];
//       local_F_inv[q] = inverse(F);
//       local_J[q] = jacobian(F);
//     }
// }

template <int dim>
void
ImmersedFEM<dim>::run()
{
					// Initialization of the time step counter and
					// of the corresponding values of time.
   unsigned int time_step = 1;
   double t = par.dt;

					// Initialization of the current state of the system:
					// at the beginning, the current state is set equal to the
					// initial state, which is stored in previous_xi. 
					// state.
   current_xi = previous_xi;
   previous_time = 0;

					// We initialize the variable  "update_Jacobian"
					// so to have a meaningful first update of the solution.
   bool update_Jacobian = true;

					// The function "n_dofs()" returns the value of the variable
					// "n_total_dofs" which corresponds to the sum of the
					// degrees of freedom in the fluid domain and the degrees of
					// freedom in the solid domain
					// While this code employs a "home-made" implicit Euler
					// time integration scheme, we set things up so to
					// facilitate the use of other time integrators, e.g., DAE.
					// Hence, we define a Vector<double> object that contains 
					// the same information as the previous_xi, but that can be
					// passed to integrators what do not accept objects of type
					// BockVector as input.
   Vector<double> solution(n_dofs());
   solution = previous_xi;
					// Write the initial conditions in the output file.
   output_step(0.0, solution, 0, par.dt);

					// The overall cycle over time begins here.
   for(; t<=par.T; t+= par.dt, ++time_step) 
     {
					// Initialization of two counters for the monitoring of
					// progress of the nonlinear solver.
       unsigned int nonlin_iter = 0;
       unsigned int outer_nonlin_iter = 0;

					// The nonlinear solver iteration cycle begins here.
       while(true)
       {
					// We view our system of equations to be of the following
					// form: F(xi', xi, t) = 0, xi(t0) = xi0.
					// Denoting the current time step by n, the vector
					// xi'(t_{n}) is assumed to be a linear combination of
					// xi(t_{i}), with i = n - m ... n, with m <= n.
					// For simplicity, here we implement an implicit Euler
					// method, according to which
					// xi'(t_{n}) = [xi(t_{n}) - xi(t_{n-1})]/dt, where dt
					// is the size of the time step.
        current_xit.sadd(0, 1./par.dt, current_xi, -1./par.dt, previous_xi);
//        utility_xi = current_xi;

        if(update_Jacobian == true)
        {
					// Determine the residual and the Jacobian of the residual.
         residual_and_or_Jacobian (current_res, JF, current_xit, current_xi, 1./par.dt, t);
					// Compute the inverse of the Jacobian.
         JF_inv.initialize(JF);
					// Reset the "update_Jacobian" variable to the value
					// specified in the parameter file.
         update_Jacobian = par.update_jacobian_continuously;
        }
       else
        {
					// Determine the residual but do not update the Jacobian.
          residual_and_or_Jacobian (current_res, dummy_JF, current_xit, current_xi, 0, t);
        }
					// Is the residual equal to zero?
       const double res_norm = current_res.l2_norm();
       if( res_norm < 1e-10 )
        {
					// Since the residual is equal zero, make a note and
					// advance to the next step.
         std::printf(" Step %03d, Res:  %-16.3e (converged in %d iterations)\n\n", time_step, res_norm, nonlin_iter);
         break;
        }
       else
        {
					// Since the residual is not equal zero, make a note of
					// it and compute an update.
         cout << nonlin_iter << ": " << res_norm << endl;
					// To compute the update to the current xi, we first change
					// the sign of the  current value of the residual ...
         current_res *= -1;
					// ... then we compute the update.  The latter is the
					// returned by the method "solve" of the object "JF_inv".
					// The latter is of class "SparseDirectUMFPACK" and
					// therefore the value of the (negative) of the current
					// residual must be supplied in a container of type
					// Vector<double>.  Hence, we first transfer the information
					// in "current_res" into a Vector<double> called "tmp" and
					// then we carry out the computation of the update.
         static Vector<double> tmp(current_res.size());
         tmp = current_res;
         JF_inv.solve(tmp);
					// Now that we have the updated of the solution into an
					// object of type Vector<double>, we repackage its content
					// into an object of type BlockVector.
         newton_update = tmp;
					// Finally, we determine the value of the updated solution.
         current_xi.add(1., newton_update);
					// We are here because an the solution needed to be updated.
					// The update was computed using whatever Jacobian was
					// available.  If, on entering this section of the loop,
					// the value of the residual was very poor and if the 
					// solutions method indicated in the parameter file did not
					// call for a continuous update of the Jacobian, now we
					// make sure that the Jacobian is updated before computing
					// the next solution update.
         if(res_norm > 1e-2) update_Jacobian = true;
        }
					// We are here because the solution needed an update.
					// So, start counting how many iterations are needed to
					// converge.  If convergen is not achieved in 15 iterations
					// update the Jacobian and try again.  The maximum number of
					// 15-iteration cycles is set to three.  The counter for
					// the cycle is "outer_nonlin_iter".
        ++nonlin_iter;
        if(nonlin_iter == 15)
        {
          update_Jacobian = true;
          nonlin_iter = 0;
          outer_nonlin_iter++;
          std::printf("   %-16.3e (not converged in 15 iterations. Step %d)\n\n", res_norm, outer_nonlin_iter);
        }

					// If converge is not in our destiny, accept defeat, with
					// as much grace as it can be mustered, and go home.
        AssertThrow (outer_nonlin_iter <= 3, ExcMessage ("No convergence in nonlinear solver"));
       }
					// We have computed a new solution.  So, we update the state
					// of the system and move to the next time step.
       previous_xi = current_xi;
       solution = current_xi;
       output_step(t, solution, time_step, par.dt);
       update_Jacobian = par.update_jacobian_continuously;
       if(par.update_jacobian_at_step_beginning) update_Jacobian = true;
       
     } // End of the cycle over time.

} // End of "run()"



template <int dim>
void ImmersedFEM<dim>::output_step (const double t,
						  const Vector<double> &solution,
						  const unsigned int step,
						  const double h)
{
  cout << "Time " << t
       << ", Step " << step
       << ", dt = " << h << endl;
  utility_xi = solution;

  global_info_file << t << " ";
  {
    std::vector<std::string> joint_solution_names (dim, "v");
    joint_solution_names.push_back ("p");
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dh_f);
    std::vector< DataComponentInterpretation::DataComponentInterpretation >
      component_interpretation (dim+1,
				DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim]
      = DataComponentInterpretation::component_is_scalar;
    
    data_out.add_data_vector (utility_xi.block(0),
			      joint_solution_names,
			      DataOut<dim>::type_dof_data,
			      component_interpretation);
    data_out.build_patches (par.degree);
    std::ofstream output ((par.output_name + "-fluid-" +
			   Utilities::int_to_string (step, 5) +
			   ".vtu").c_str());
    data_out.write_vtu (output);
  }
  {
    
    std::vector<std::string> joint_solution_names (dim, "W");
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dh_s);
    std::vector< DataComponentInterpretation::DataComponentInterpretation >
      component_interpretation (dim,
				DataComponentInterpretation::component_is_part_of_vector);
    
    data_out.add_data_vector (utility_xi.block(1),
			      joint_solution_names,
			      DataOut<dim>::type_dof_data,
			      component_interpretation);
    data_out.build_patches (*mapping);
    std::ofstream output ((par.output_name + "-solid-" +
			   Utilities::int_to_string (step, 5) +
			   ".vtu").c_str());
    data_out.write_vtu (output);
  }
  {
				     // Assemble in and out flux
    typename DoFHandler<dim,dim>::active_cell_iterator
       cell = dh_f.begin_active(),
       endc = dh_f.end();
    QGauss<dim-1> face_quad(par.degree+2);
    FEFaceValues<dim,dim> fe_v(fe_f, face_quad,
			       update_values |
			       update_JxW_values |
			       update_normal_vectors);
    std::vector<Vector<double> > local_vp(face_quad.size(),
					  Vector<double>(dim+1));
    double flux=0;
    for(; cell != endc; ++cell) 
      for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	if(cell->face(f)->at_boundary()) 
	  {
	    fe_v.reinit(cell, f);
	    fe_v.get_function_values(utility_xi.block(0), local_vp);
	    const std::vector<Point<dim> > &normals = fe_v.get_normal_vectors();
	    for(unsigned int q=0; q<face_quad.size(); ++q) 
	      {
		Point<dim> vq;
		for(unsigned int d=0;d<dim;++d)
		  vq[d] = local_vp[q](d);
		
		flux += (vq*normals[q])*fe_v.JxW(q);
	      }
	  }
    global_info_file << flux << " ";
  }
  {
				     // Compute area of the solid, and
				     // location of its center of mass
    typename DoFHandler<dim,dim>::active_cell_iterator
       cell = dh_s.begin_active(),
       endc = dh_s.end();
    FEValues<dim,dim> fe_v(*mapping, fe_s, quad_s,
			   update_JxW_values |
			   update_quadrature_points);
    std::vector<Vector<double> > local_X(quad_s.size(),
					 Vector<double>(dim+1));
    double area=0;
    Point<dim> center;
    for(; cell != endc; ++cell) 
      {
	fe_v.reinit(cell);
	const std::vector<Point<dim> > &qpoints = fe_v.get_quadrature_points();
	for(unsigned int q=0; q<quad_s.size(); ++q) 
	  {
	    area += fe_v.JxW(q);
	    center += fe_v.JxW(q)*qpoints[q];
	  }
      }
    center /= area;
    global_info_file << area << " ";
    global_info_file << center << std::endl;
  }
}

// template <int dim>
// template<class Type>
// void ImmersedFEM<dim>::localize
// (
//   std::vector<Type> &local_x,
//   const Vector<double> &global_xt,
//   const Vector<double> &global_x,
//   const unsigned int inner_offset,
//   const unsigned int outer_offset,
//   const std::vector< unsigned int> &dofs)
// 
// {
//   bool update_Jacobian = is_sacado(local_x[0]);
// 
//   Assert(local_x.size() == 2*outer_offset,
// 	 ExcDimensionMismatch(local_x.size(), 2*outer_offset));
//   Assert(dofs.size() + inner_offset <= outer_offset,
// 	 ExcMessage("Wrong dimensions..."));
//   
//   for (unsigned int i=0; i<dofs.size(); ++i){
//     unsigned int wi = i + inner_offset;
//     
//     local_x[wi]			= global_xt(dofs[i]);
//     local_x[wi+outer_offset]	= global_x(dofs[i]);
//     
//     if(update_Jacobian == true) {
//       std::vector<SacDouble> &local_x_sac = reinterpret_cast<std::vector<SacDouble>&>(local_x);
//       local_x_sac[wi].diff(wi, 2*outer_offset);
//       local_x_sac[wi + outer_offset].diff(wi + outer_offset, 2*outer_offset);
//     }
//   }
// }


// template <int dim>
// template <class Type>
// void ImmersedFEM<dim>::set_to_zero (Type &v) const
// {
//   v = 0;
// }
// 
// template <int dim>
// template <class Type>
// void ImmersedFEM<dim>::set_to_zero (std::vector<Type> &v) const
// {
//   for(unsigned int i = 0; i < v.size(); ++i)
//     set_to_zero(v[i]);
// }
// 
// template <int dim>
// template <class Type>
// void ImmersedFEM<dim>::set_to_zero (Table<2, Type> &v) const
// {
//   for(unsigned int i=0; i<v.size()[0]; ++i)
//     for(unsigned int j=0; j<v.size()[1]; ++j)
//       set_to_zero(v(i,j));
// }
// 
// template <int dim>
// template <class Type>
// Table<2,Type> ImmersedFEM<dim>::piola
// (const Table<2,Type> &F) const 
// {
//   // Piola-Kirchhoff response function.
//   Table<2,Type> P(dim,dim);
//   
//   if(par.zero_residual_stress == true) 
//     {
//       Table<2,Type> Finv = inverse(F);
//       
// 				       // Classical incompressible Neo-Hookean
//       for(unsigned int i=0; i<dim; ++i)
// 	for(unsigned int j=0; j<dim; ++j)
// 	  P(i,j) = par.mu*(F(i,j) - Finv(j,i));
//     }
//   else 
//     {
//       for(unsigned int i=0; i<dim; ++i)
// 	for(unsigned int j=0; j<dim; ++j)
// 	  P(i,j) = par.mu*(F(i,j));
//     }
//   
//   return P;
// }

template <int dim>
void ImmersedFEM<dim>::get_Agamma_values( const FEValues<dim,dim> &fe_v_s,
										  const std::vector< unsigned int > &dofs,
										  const Vector<double> &xi, 
										  Vector<double> &local_A_gamma)
{
   set_to_zero(local_A_gamma);

   unsigned int qsize = fe_v_s.get_quadrature().size();
   
   std::vector< std::vector< Tensor<1,dim> > > H(qsize, std::vector< Tensor<1,dim> >(dim));
   fe_v_s.get_function_gradients(xi, H);
   
   for( unsigned int qs = 0; qs < qsize; ++qs )
   {
	  for (unsigned int k = 0; k < dofs.size(); ++k)
	  {
		 unsigned int comp_k = fe_s.system_to_component_index(k).first;
		 
		 //Agamma = P:Grad_y, where P = mu*F= mu*(H+I)
		 local_A_gamma (k) +=
		 par.mu* (H[qs][comp_k]*fe_v_s.shape_grad(k, qs)
				  + fe_v_s.shape_grad(k, qs)[comp_k])
		 *fe_v_s.JxW(qs);
	  }
	  
   }//end loop over quadrature points
   
}

template <int dim>
void ImmersedFEM<dim>::get_PFT_and_PFT_Dxi_values(
                const FESystem<dim,dim> &fe_s,
                const FEValues<dim,dim> &fe_v_s,
                const std::vector< unsigned int > &dofs,
                const Vector<double> &xi, 
                const bool update_jacobian, 
                std::vector<Tensor<2,dim,double> > &PFT,
                std::vector< std::vector<Tensor<2,dim,double> > > & PFT_Dxi)
{
  std::vector< std::vector< Tensor<1,dim> > > H(PFT.size(), std::vector< Tensor<1,dim> >(dim));
  fe_v_s.get_function_gradients(xi, H);
  
  for( unsigned int qs = 0; qs < PFT.size(); ++qs )
    {
      for( unsigned int i = 0; i < dim; ++i )
        for( unsigned int j = 0; j < dim; ++j )
          {
			 PFT[qs][i][j]  = H[qs][i][j] + H[qs][j][i] + (H[qs][i] * H[qs][j]) + (i==j? 1.0: 0.0);//The last term has been added since the constitutive relation is now P=mu*F instead of mu*(F-F^{-T})
            PFT[qs][i][j] *= par.mu;

            if( update_jacobian )
             {
               for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
                 {
                   PFT_Dxi[qs][k][i][j] = 0;
                   unsigned int comp_k = fe_s.system_to_component_index(k).first;
                   if( i == comp_k )
                   PFT_Dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)[j]
                                         + fe_v_s.shape_grad(k,qs) * H[qs][j];
                   if( j == comp_k )
                   PFT_Dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)[i]
                                         + fe_v_s.shape_grad(k,qs) * H[qs][i];
                   PFT_Dxi[qs][k][i][j] *= par.mu;
                 }
             }
          }
    }
  
}

// template <int dim>
// template <class Type>
// Table<2,Type> ImmersedFEM<dim>::deviatoric
// (const Table<2,Type> &P,
//  const Table<2,Type> &F,
//  const Table<2,Type> &Finv) const 
// {
//   Table<2,Type> Pd(dim,dim);
//   Type Ps = spherical(P, F);
//   
//   for(unsigned int i=0; i<dim;++i)
//     for(unsigned int j=0; j<dim;++j)
//       Pd(i,j) = (P(i,j) - Ps*Finv(j,i));
//   return Pd;
// }

// template <int dim>
// template <class Type>
// Type ImmersedFEM<dim>::spherical
// (const Table<2,Type> &P,
//  const Table<2,Type> &F) const 
// {
//   Type Ps = 0;
//   for(unsigned int i=0; i<dim;++i)
//     for(unsigned int j=0; j<dim;++j)
//       Ps += 1./(dim)*P(i,j)*F(i,j);
//       //
//       Ps = 0;
//   return Ps;
// }


// template <int dim>
// template <class Type>
// Type ImmersedFEM<dim>::jacobian
// (const Table<2,Type> &F) const 
// {
//   Type J = 0;
//   switch(dim) 
//     {
//       case 2:
// 	    J = F(0,0)*F(1,1)-F(0,1)*F(1,0);
// 	    break;
//       case 3:
// 	    J = F(0,0) * F(1,1) * F(2,2) -
// 		F(0,0) * F(1,2) * F(2,1) +
// 		F(1,0) * F(2,1) * F(0,2) -
// 		F(1,0) * F(0,1) * F(2,2) +
// 		F(2,0) * F(0,1) * F(1,2) -
// 		F(2,0) * F(1,1) * F(0,2);
// 	    
// 	    break;
//       default:
// 	    Assert(false, ExcInternalError());
// 	    break;
//     }
//   return J;
// }
// 
// 
// template <int dim>
// template <class Type>
// Table<2,Type> ImmersedFEM<dim>::inverse
// (const Table<2,Type> &F) const 
// {
//   Table<2,Type> Finv (dim,dim);
//   Type Jinv = 1./jacobian(F);
//   switch(dim) 
//     {
//       case 2:
// 	    Finv[0][0] = Jinv*(F[1][1]);
// 	    Finv[0][1] = Jinv*(-F[0][1]);
// 	    Finv[1][0] = Jinv*(-F[1][0]);
// 	    Finv[1][1] = Jinv*(F[0][0]);
// 
// 	    break;
//       case 3:
// 	    Finv[0][0] = Jinv*(F[1][1] * F[2][2] - F[1][2] * F[2][1]);
// 	    Finv[0][1] = Jinv*(-F[0][1] * F[2][2] + F[0][2] * F[2][1]);
// 	    Finv[0][2] = Jinv*(F[0][1] * F[1][2] - F[0][2] * F[1][1]);
// 	    Finv[1][0] = Jinv*(F[2][0] * F[1][2] - F[1][0] * F[2][2]);
// 	    Finv[1][1] = Jinv*(-F[2][0] * F[0][2] + F[0][0] * F[2][2]);
// 	    Finv[1][2] = Jinv*(F[1][0] * F[0][2] - F[0][0] * F[1][2]);
// 	    Finv[2][0] = Jinv*(-F[2][0] * F[1][1] + F[1][0] * F[2][1]);
// 	    Finv[2][1] = Jinv*(F[2][0] * F[0][1] - F[0][0] * F[2][1]);
// 	    Finv[2][2] = Jinv*(-F[1][0] * F[0][1] + F[0][0] * F[1][1]);
// 
// 	    break;
//       default:
// 	    Assert(false, ExcInternalError());
// 	    break;
//     }
//   return Finv;
// }


  

template <int dim>
void ImmersedFEM<dim>::distribute_residual(Vector<double> &residual,
						const std::vector<double> &local_res,
						const std::vector<unsigned int> &dofs_1,
						const unsigned int offset_1)
{
  for(unsigned int i=0, wi=offset_1; i<dofs_1.size();++i,++wi)
    {
      residual(dofs_1[i]) += local_res[wi];
    }
}

template <int dim>
void ImmersedFEM<dim>::distribute_jacobian(SparseMatrix<double> &Jacobian,
						const FullMatrix<double> &local_Jac,
						const std::vector<unsigned int> &dofs_1,
						const std::vector<unsigned int> &dofs_2,
						const unsigned int offset_1,
						const unsigned int offset_2)
{

  for(unsigned int i=0, wi=offset_1; i<dofs_1.size();++i,++wi)
    {
      for(unsigned int j=0, wj=offset_2; j<dofs_2.size();++j,++wj) Jacobian.add(dofs_1[i],dofs_2[j],local_Jac(wi,wj));
    }
}

// template <int dim>
// template <class Type>
// void ImmersedFEM<dim>::distribute(BlockVector<double> &residual,
// 						BlockSparseMatrix<double> &Jacobian,
// 						const std::vector<Type> &local_res_typed,
// 						const std::vector<unsigned int> &dofs_f,
// 						const std::vector<unsigned int> &dofs_s,
// 						const double alpha) const
// {
//   const unsigned int n_local_dofs = fe_f.dofs_per_cell + fe_s.dofs_per_cell;
// 
// 				   // Fluid - Fluid
//   distribute(residual.block(0), Jacobian.block(0,0),
// 	     local_res_typed, dofs_f, 0, alpha);
// 
// 				   // Solid - Solid
//   distribute(residual.block(1), Jacobian.block(1,1),
// 	     local_res_typed, dofs_s, fe_f.dofs_per_cell, alpha);
//   
//   if(is_sacado(local_res_typed[0]) )
//     {
//       const std::vector<SacDouble> &local_res
// 	= reinterpret_cast<const std::vector<SacDouble> &>(local_res_typed);
//       
//       for(unsigned int i=0; i<fe_f.dofs_per_cell;++i)
// 	for(unsigned int j=0, wj=fe_f.dofs_per_cell; j<fe_s.dofs_per_cell;++j,++wj) 
// 	  {
// 	    
// 	    if(local_res[i].dx() != NULL)
// 	      Jacobian.block(0,1).add(dofs_f[i],dofs_s[j],
// 				      alpha*local_res[i].fastAccessDx(wj) +
// 				      local_res[i].fastAccessDx(wj+n_local_dofs));
// 	    
// 	    if(local_res[wj].dx() != NULL)
// 	      Jacobian.block(1,0).add(dofs_s[j],dofs_f[i],
// 				      alpha*local_res[wj].fastAccessDx(i) +
// 				      local_res[wj].fastAccessDx(i+n_local_dofs));
// 	  }
//     }
// }


// template <int dim>
// void ImmersedFEM<dim>::distribute_extra(BlockSparseMatrix<double> &Jacobian,
// 						      const std::vector<SacDouble> &local_res_extra,
// 						      const std::vector<unsigned int> &dofs_s,
// 						      const double) const
// {
//   const unsigned int n_local_dofs = fe_f.dofs_per_cell + fe_s.dofs_per_cell;
//   for(unsigned int i=0, wi=fe_f.dofs_per_cell; i<fe_s.dofs_per_cell;++i,wi++)
//     for(unsigned int j=0, wj=fe_f.dofs_per_cell; j<fe_s.dofs_per_cell;++j,++wj) 
//       {
// 	
// 	if(local_res_extra[wi].dx() != NULL)
// 	  Jacobian.block(1,1).add(dofs_s[i],dofs_s[j],
// 				  local_res_extra[wi].fastAccessDx(wj+n_local_dofs));
//       }
// }


// template <int dim>
// Vector<double> & ImmersedFEM<dim>::differential_components() 
// {
//   static Vector<double> v;
//   if(v.size() == 0) 
//     {
//       v.reinit(n_dofs(), true);
//       v = 1;
//       for(unsigned int i=0; i<n_dofs_p; ++i)
// 	v(i+n_dofs_u) = 0;
//     }
//   return v;
// }


// template <int dim>
// int ImmersedFEM<dim>::jacobian_prec(double,
// 						 Vector<double>&dst,
// 						 const Vector<double> & ,
// 						 const Vector<double> & ,
// 						 const Vector<double> & src,
// 						 const double)
// {
//   JF_inv.vmult(dst, src);
//   return 0;
// }


template <int dim>
void ImmersedFEM<dim>::apply_constraints(vector<double> &local_res,
						      FullMatrix<double> &local_jacobian,
						      const Vector<double> &value_of_dofs,
						      const vector<unsigned int> &dofs)
{
  
  for(unsigned int i=0; i<dofs.size(); ++i) 
    {
      map<unsigned int,double>::iterator it = par.boundary_values.find(dofs[i]);
      if(it != par.boundary_values.end() )
       {
         local_res[i] = scaling * ( value_of_dofs(dofs[i]) - it->second );
         if( !local_jacobian.empty() )
          {
           for(unsigned int j=0; j<local_jacobian.n(); ++j) local_jacobian(i,j) = 0;
           local_jacobian(i,i) = scaling;
          }
       }
	   /*------------------------------------------------------------------------------*/
	   if(par.all_DBC && !par.fix_pressure)
	   {
		  if(dofs[i] == constraining_dof)
		  {
			 local_res[i] = 0; 
			 if( !local_jacobian.empty() )
				local_jacobian.add_row(i, -1, i);
		  }
	   }
	   /*------------------------------------------------------------------------------*/
	   
    }
}

/*-----------------------------------------------*/
template <int dim> 
void ImmersedFEM<dim>::
distribute_constraint_on_pressure(Vector<double> &residual,
						       const double average_pressure)
{
   residual(constraining_dof) += average_pressure*scaling;
}

template <int dim> 
void ImmersedFEM<dim>::
distribute_constraint_on_pressure( SparseMatrix<double> &jacobian,
							  const vector<double> &pressure_coefficient,
							  const vector<unsigned int> &dofs,
							  const unsigned int offset)
{
   for(unsigned int i=0, wi=offset; i<dofs.size();++i,++wi)
	  jacobian.add(constraining_dof, dofs[i], pressure_coefficient[wi]);
   
}
/*---------------------------------------------------*/
/*---------------------------------------------------*/
template <int dim> 
void ImmersedFEM<dim>::
localize (Vector<double> &local_M_gamma3_inv_A_gamma,
		  const Vector<double> &M_gamma3_inv_A_gamma,
		  const vector<unsigned int> &dofs)
{
   for (unsigned int i = 0; i < dofs.size(); ++i)
   {
	  local_M_gamma3_inv_A_gamma (i) = M_gamma3_inv_A_gamma(dofs[i]);
   }
}
/*---------------------------------------------------*/

// template <int dim>
// int ImmersedFEM<dim>::jacobian(double,
// 					    Vector<double>&dst,
// 					    const Vector<double> & ,
// 					    const Vector<double> & ,
// 					    const Vector<double> & src,
// 					    const double)
// {
//   xi = src;
//   JF.vmult(current_xi, xi);
//   dst = current_xi;
//   return 0;
// }

template <int dim>
template <class Type>
void ImmersedFEM<dim>::set_to_zero (Type &v) const
{
  v = 0;
}

template <int dim>
template <class Type>
void ImmersedFEM<dim>::set_to_zero (std::vector<Type> &v) const
{
  for(unsigned int i = 0; i < v.size(); ++i)
    set_to_zero(v[i]);
}

template <int dim>
template <class Type>
void ImmersedFEM<dim>::set_to_zero (Table<2, Type> &v) const
{
  for(unsigned int i=0; i<v.size()[0]; ++i)
    for(unsigned int j=0; j<v.size()[1]; ++j)
      set_to_zero(v(i,j));
}

template <int dim>
double ImmersedFEM<dim>::norm(const std::vector<double> &v)
{
  double norm = 0;
  for( unsigned int i = 0; i < v.size(); ++i) norm += v[i]*v[i];
  return norm = sqrt(norm);
}



				 // @sect3{ The main function }

				 // The main function looks very much like in
				 // all the other tutorial programs, so there
				 // is little to comment on here:
int main()
{
  try
    {
      ProblemParameters<2> par;
      ImmersedFEM<2> test (par);
      test.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  std::cout << "----------------------------------------------------"
	    << std::endl
	    << "Apparently everything went fine!"
	    << std::endl
	    << "Don't forget to brush your teeth :-)"
	    << std::endl << std::endl;
  return 0;
}
