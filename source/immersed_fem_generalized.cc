#include "immersed_fem_generalized.h"

void move_file (const string &old_name,
				const string &new_name)
{
   const int error = system (("mv " + old_name + " " + new_name).c_str());
   
   AssertThrow (error == 0, ExcMessage(string ("Can't move files: ")
									   +
									   old_name + " -> " + new_name));
}




// Constructor:
//    Initializes the FEM system of the control volume;
//    Initializes the FEM system of the immersed domain;
//    Initializes, corresponding dof handlers, and the quadrature rule;
//    It runs the <code>create_triangulation_and_dofs</code> function.

template <int dim>
ImmersedFEMGeneralized<dim>::ImmersedFEMGeneralized (IFEMParametersGeneralized<dim> &par)
		:
		par (par),
		fe_f (
		  FE_Q<dim>(par.degree),
		  dim,
		  *FETools::get_fe_from_name<dim>(par.fe_p_name),
		  1
		),
		fe_s (FE_Q<dim, dim>(par.degree), dim),
		dh_f (tria_f),
		dh_s (tria_s),
		quad_f (par.degree+2)
{
  if(par.degree <= 1)
    cout
      << " WARNING: The chosen pair of finite element spaces is not  stable."
      << endl
      << " The obtained results will be nonsense."
      << endl;

  if( Utilities::match_at_string_start(par.fe_p_name, string("FE_DGP")))
    dgp_for_p = true;
  else dgp_for_p = false;

   switch (par.quad_s_type) 
   {
	  case IFEMParametersGeneralized<dim>::QGauss :
		 quad_s.initialize(
						   QGauss<dim>(par.quad_s_degree).get_points(),
						   QGauss<dim>(par.quad_s_degree).get_weights()
						   );
		 break;
	  case IFEMParametersGeneralized<dim>::Qiter_Qtrapez :
		 quad_s.initialize(
						   QIterated<dim>
						   (QTrapez<1>(), par.quad_s_degree).get_points(),
						   QIterated<dim>
						   (QTrapez<1>(), par.quad_s_degree).get_weights()
						   );
		 break;
	  case IFEMParametersGeneralized<dim>::Qiter_Qmidpoint :
		 quad_s.initialize(
						   QIterated<dim>
						   (QMidpoint<1>(), par.quad_s_degree).get_points(),
						   QIterated<dim>
						   (QMidpoint<1>(), par.quad_s_degree).get_weights()
						   );
		 break;
	  default:
		 break;
   }

   if(par.this_is_a_restart)
   {
	  global_info_file.open((par.output_name+"_global.gpl").c_str(), ios::app);
	  
	  if(par.fsi_bm)
		 fsi_bm_out_file.open((par.output_name+"_fsi_bm.out").c_str(), ios::app);
   }
   else
   {
	  global_info_file.open((par.output_name+"_global.gpl").c_str());
   
	  if(par.fsi_bm)
		 fsi_bm_out_file.open((par.output_name+"_fsi_bm.out").c_str());
   }
   
   create_triangulation_and_dofs ();

	  
}

// Distructor: deletion of pointers created with <code>new</code> and
// closing of the record keeping file.

template <int dim>
ImmersedFEMGeneralized<dim>::~ImmersedFEMGeneralized ()
{
   if (par.save_for_restart)
   	  save_for_restart();
   
  delete mapping;
  global_info_file.close();
   
   if(par.fsi_bm)
	  fsi_bm_out_file.close();
	  
}

// Determination of the current value of time dependent boundary
// values.

template <int dim>
void
ImmersedFEMGeneralized<dim>::compute_current_bc (const double t)
{
  par.u_g.set_time(t);
  VectorTools::interpolate_boundary_values (
    StaticMappingQ1<dim>::mapping,
    dh_f,
    par.boundary_map,
    par.boundary_values,
    par.component_mask
  );

// Set to zero the value of the first dof associated to
// the pressure field.
  if( (par.solid_is_compressible == false) && (par.fix_pressure == true)) 
	 par.boundary_values[constraining_dof] = 0;
}

// Application of time dependent boundary conditions.

template <int dim>
void
ImmersedFEMGeneralized<dim>::apply_current_bc
(
  BlockVector<double> &vec,
  const double t
)
{
  compute_current_bc(t);
  map<unsigned int, double>::iterator it    = par.boundary_values.begin(),
				      itend = par.boundary_values.end();
  if(vec.size() != 0)
    for(; it != itend; ++it)
      vec.block(0)(it->first) = it->second;
  else
    for(; it != itend; ++it)
      constraints_f.set_inhomogeneity(it->first, it->second);
   
   if(par.use_dbc_solid)
	  for(it=par.boundary_values_solid.begin(); it != par.boundary_values_solid.end(); ++it)
		 vec.block(1)(it->first) = it->second;
	  
}



// Defines the triangulations for both the control volume and the
// immersed domain.  It distributes degrees of freedom over said
// triangulations. Both grids are assumed to be available in UCD
// format. The naming convention is as follows:
// <code>fluid_[dim]d.inp</code> for the control volume and
// <code>solid_[dim]d.inp</code> for the immersed domain. This function also
// sets up the constraint matrices for the enforcement of Dirichlet
// boundary conditions. In addition, it sets up the framework for
// enforcing the initial conditions.

template <int dim>
void
ImmersedFEMGeneralized<dim>::create_triangulation_and_dofs ()
{
   
  if(par.material_model == IFEMParametersGeneralized<dim>::CircumferentialFiberModel)
    {
// This is used only by the solution of the problem with the immersed
// domain consisting of a circular cylinder.  We only implemented this
// in two dimensions.
      Assert(dim == 2, ExcNotImplemented());
      
      ExactSolutionRingWithFibers<dim> ring(par);
	   
// Construct the square domain for the control volume using the parameter file.
	   GridGenerator::hyper_cube (tria_f, 0., ring.l);  

// Construct the hyper shell using the parameter file.      
      GridGenerator::hyper_shell(tria_s, ring.center,
				 ring.R, ring.R+ring.w);
      
      static const HyperShellBoundary<dim> shell_boundary(ring.center);
      tria_s.set_boundary(0, shell_boundary);
    }
 else if (par.disk_falling_test)
 {
	Assert(dim == 2, ExcNotImplemented());
		
	// Construct a rectangular domain for the control volume using the parameter file.
	GridGenerator::hyper_rectangle (tria_f, 
									par.rectangle_bl,
									par.rectangle_tr,
									par.colorize_boundary
									);  
	
	// Construct a hyper ball to represent the solid using the parameter file.      
	GridGenerator::hyper_ball(tria_s,
							  par.ball_center,
							  par.ball_radius);
	
   static const HyperBallBoundary<dim> ball_boundary( par.ball_center, 
													 par.ball_radius);
									
	tria_s.set_boundary(0, ball_boundary);
 }
  else
    {
	   // As specified in the documentation for the "GridIn" class the
	   // triangulation corresponding to a grid needs to be empty at
	   // this time.
	   GridIn<dim> grid_in_f;
	   grid_in_f.attach_triangulation (tria_f);
	   
	   {
		  ifstream file (par.fluid_mesh.c_str());
		  Assert (file, ExcFileNotOpen (par.fluid_mesh.c_str()));
		  
		  
		  // A grid in ucd format is expected.
		  grid_in_f.read_ucd (file);
	   }
	   
      GridIn<dim, dim> grid_in_s;
      grid_in_s.attach_triangulation (tria_s);

      ifstream file (par.solid_mesh.c_str());
      Assert (file, ExcFileNotOpen (par.solid_mesh.c_str()));
      
// A grid in ucd format is expected.
      grid_in_s.read_ucd (file);
    }

 if (par.fsi_bm)
 {
	Point<dim> center_circ(0.2, 0.2);
	double radius_circ = 0.05;
	static const HyperBallBoundary<dim> boundary_cyl(center_circ,radius_circ);
	
	tria_f.set_boundary(80, boundary_cyl);
//	tria_f.set_boundary(81, boundary_cyl);
	
//	tria_s.set_boundary(81, boundary_cyl);
 }
   if(par.brain_mesh)
   {
	  par.enter_subsection("Grid parameters for brain mesh");
	  double scale_factor= par.get_double("Scaling factor");
	  double shift_x =par.get_double("Translation x-dirn");
	  double shift_y =par.get_double("Translation y-dirn");
	  par.leave_subsection();
	  
	  cout<<"Brain mesh shift"<<shift_x<<","<<shift_y<<", scale="<<scale_factor<<endl;
	  //Transformations for the brain mesh only:
	  GridTools::shift(Point<dim>(shift_x,shift_y), tria_s);
	  GridTools::scale(scale_factor, tria_s);
	  
   }
   
  cout
    << "Number of fluid refines = "
    << par.ref_f
    << endl;
  tria_f.refine_global (par.ref_f);
  cout
    << "Number of active fluid cells: "
    << tria_f.n_active_cells ()
    << endl;
  cout
    << "Minimal fluid cell diam = "
    << GridTools::minimal_cell_diameter(tria_f)
    << endl;
  cout
   << "Maximal fluid cell diam = "
   << GridTools::maximal_cell_diameter(tria_f)
   << endl;//SR 
  cout
    << "Number of solid refines = "
    << par.ref_s
    << endl;
  tria_s.refine_global (par.ref_s);
  cout
    << "Number of active solid cells: "
    << tria_s.n_active_cells ()
    << endl;
  cout
    << "Minimal solid cell diam = "
    << GridTools::minimal_cell_diameter(tria_s)
    << endl;
  cout
    << "Maximal solid cell diam = "
    << GridTools::maximal_cell_diameter(tria_s)
    << endl; 

// Initialization of the boundary_indicators vector.
  boundary_indicators = tria_f.get_boundary_indicators ();


// Distribution of the degrees of freedom. Both for the solid
// and fluid domains, the dofs are renumbered first globally
// and then by component.
  dh_f.distribute_dofs (fe_f);
  DoFRenumbering::boost::Cuthill_McKee (dh_f);


// Consistently with the fact that the various components of
// the system are stored in a block matrix, now renumber
// velocity and pressure component wise.
  vector<unsigned int> block_component (dim+1,0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise (dh_f, block_component);

  vector<unsigned int> dofs_per_block (2);
  DoFTools::count_dofs_per_block (dh_f, dofs_per_block, block_component);


// Accounting of the number of degrees of freedom for the fluid
//  domain on a block by block basis.
  n_dofs_u  = dofs_per_block[0];
  n_dofs_p  = dofs_per_block[1];
  n_dofs_up = dh_f.n_dofs ();


// Simply distribute dofs on the solid displacement.
  dh_s.distribute_dofs (fe_s);
  DoFRenumbering::boost::Cuthill_McKee (dh_s);


// Determine the total number of dofs.
  n_dofs_W = dh_s.n_dofs ();
  n_total_dofs = n_dofs_up+n_dofs_W;

  cout
    << "dim (V_h) = "
    << n_dofs_u
    << endl
    << "dim (Q_h) = "
    << n_dofs_p
    << endl
    << "dim (Z_h) = "
    << dh_s.n_dofs ()
    << endl
    << "Total: "
    << n_total_dofs
    << endl;
   
  vector<unsigned int> all_dofs (2);
  all_dofs[0] = n_dofs_up;
  all_dofs[1] = n_dofs_W;


// Re-initialization of the BlockVectors containing the values of the
// degrees of freedom and of the residual.
  current_xi.reinit (all_dofs);
  previous_xi.reinit (all_dofs);
  current_xit.reinit (all_dofs);
  current_res.reinit (all_dofs);
  newton_update.reinit (all_dofs);

// Re-initialization of the average and unit pressure vectors.
  pressure_average.reinit (n_dofs_up);
  unit_pressure.reinit (n_dofs_up);

// Re-initialization of temporary vectors.
  tmp_vec_n_total_dofs.reinit(n_total_dofs);
  tmp_vec_n_dofs_up.reinit(n_dofs_up);
   tmp_vec_n_dofs_W.reinit(n_dofs_W);

// We now deal with contraint matrices.
  {
    constraints_f.clear ();
    constraints_s.clear ();

// Enforce hanging node constraints.
    DoFTools::make_hanging_node_constraints (dh_f, constraints_f);
    DoFTools::make_hanging_node_constraints (dh_s, constraints_s);


// To solve the problem we first assemble the Jacobian of the residual
// using zero boundary values for the velocity. The specification of
// the actual boundary values is done later by the <code>apply_current_bc</code>
// function.
    VectorTools::interpolate_boundary_values (
      StaticMappingQ1<dim>::mapping,
      dh_f,
      par.zero_boundary_map,
      constraints_f,
      par.component_mask);
  }
   
   //: Currently the immersed solid can have homogeneous Dirichlet boundary conditions for the CFD or FSI BM tests
   VectorTools::interpolate_boundary_values (
											 StaticMappingQ1<dim>::mapping,
											 dh_s,
											 par.boundary_map_solid,
											 par.boundary_values_solid
											 );
   


// Determine the area (in 2D) of the control volume and find the first
// dof pertaining to the pressure.
  get_area_and_first_pressure_dof ();

  constraints_f.close ();
  constraints_s.close ();


// The following matrix plays no part in the formulation. It is
// defined here only to use the VectorTools::project function in
// initializing the vectors previous_xi.block(0) and unit_pressure.
  ConstraintMatrix cc;
  cc.close();



// Construction of the initial conditions.
   if (par.this_is_a_restart)
	  restart_computations();
   else
   {
	  previous_time = 0.0;
	  current_time = 0.0; 
	  time_step = 0;
	  dt = par.dt;	  
	  
	  if(fe_f.has_support_points())
	  {
		 VectorTools::interpolate (dh_f, par.u_0, previous_xi.block(0));
		 VectorTools::interpolate (
								   dh_f,
								   ComponentSelectFunction<dim>(dim, 1., dim+1),
								   unit_pressure
								   );
	  }
	  else
	  {
		 VectorTools::project (dh_f, cc, quad_f, par.u_0, previous_xi.block(0));
		 VectorTools::project (
							   dh_f,
							   cc,
							   quad_f,
							   ComponentSelectFunction<dim>(dim, 1., dim+1),
							   unit_pressure
							   );
	  }

	  if(fe_s.has_support_points())
		 VectorTools::interpolate (dh_s, par.W_0, previous_xi.block(1));
	  else
		 VectorTools::project (dh_s, cc, quad_s, par.W_0, previous_xi.block(1));
	  
   }
   // Initialization of the current state of the system.
   current_xi = previous_xi;
   
   mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree,
															 previous_xi.block(1),
															 dh_s);
   if (!par.this_is_a_restart)
   {
   // Write the initial conditions in the output file.
	  output_step (0.0, previous_xi, time_step, par.dt, true);
	  
	  if (par.fsi_bm) fsi_bm_postprocess2();
   }
   
// We now deal with the sparsity patterns.
  {

    BlockCompressedSimpleSparsityPattern csp (2,2);

    csp.block(0,0).reinit (n_dofs_up, n_dofs_up);
    csp.block(0,1).reinit (n_dofs_up, n_dofs_W );
    csp.block(1,0).reinit (n_dofs_W , n_dofs_up);
    csp.block(1,1).reinit (n_dofs_W , n_dofs_W );


// As stated in the documentation, now we <i>must</i> call the function
// <code>csp.collect_sizes.()</code> since have changed the size
// of the sub-objects of the object <code>csp</code>.
    csp.collect_sizes();

    Table< 2, DoFTools::Coupling > coupling(dim+1,dim+1);
    for(unsigned int i=0; i<dim; ++i)
      {

// Velocity is coupled with pressure.
	coupling(i,dim) = DoFTools::always;

// Pressure is coupled with velocity.
	coupling(dim,i) = DoFTools::always;
	for(unsigned int j=0; j<dim; ++j)

// The velocity components are coupled with themselves and each other.
	  coupling(i,j) = DoFTools::always;
      }

// The pressure is coupled with itself.
    coupling(dim, dim) = DoFTools::always;


// Find the first pressure dof.  Then tell all the pressure dofs that
// they are related to the first pressure dof.
	 if (par.all_DBC && !par.solid_is_compressible)
	 {
		set<unsigned int>::iterator it = pressure_dofs.begin();
		for(++it; it != pressure_dofs.end(); ++it)
		{
		   csp.block(0,0).add(constraining_dof, *it);
		}
	 }

    DoFTools::make_sparsity_pattern (dh_f,
				     coupling,
				     csp.block(0,0),
				     constraints_f,
				     true);
    DoFTools::make_sparsity_pattern (dh_s, csp.block(1,1));

    sparsity.copy_from (csp);
    assemble_sparsity(*mapping);
  }

// Here is the Jacobian matrix.
  JF.reinit(sparsity);


// Boundary conditions at t = 0 (Note: If this is a restart then nothing needs to be done.)
   if (!par.this_is_a_restart)
	  apply_current_bc(previous_xi, previous_time);


   if (par.use_spread || par.fsi_bm)
   {
// Resizing other containers concerning the elastic response of the
// immersed domain.
	  A_gamma.reinit(n_dofs_W);
	  M_gamma3_inv_A_gamma.reinit(n_dofs_W);


// Creating the mass matrix for the solid domain and storing its
// inverse.
	  ConstantFunction<dim> phi_b_func (par.Phi_B, dim);
	  M_gamma3.reinit (sparsity.block(1,1));


// Using the <code>deal.II</code> in-built functionality to
// create the mass matrix.
	  MatrixCreator::create_mass_matrix (dh_s, quad_s, M_gamma3, &phi_b_func);
	  M_gamma3_inv.initialize (M_gamma3);
   }

   //: Determine the volume flux vector at the initial instant of time
   if(par.n_pt_source)
   {
	  volume_flux.reinit(n_dofs_up);
	  get_volume_flux_vector (par.t_i);
   }

}


// Relatively standard way to determine the sparsity pattern of each
// block of the global Jacobian.

template <int dim>
void
ImmersedFEMGeneralized<dim>::assemble_sparsity (Mapping<dim, dim> &immersed_mapping)
{
  FEFieldFunction<dim, DoFHandler<dim>, Vector<double> > up_field (dh_f, tmp_vec_n_dofs_up);

  vector< typename DoFHandler<dim>::active_cell_iterator > cells;
  vector< vector< Point< dim > > > qpoints;
  vector< vector< unsigned int> > maps;
  vector< unsigned int > dofs_f(fe_f.dofs_per_cell);
  vector< unsigned int > dofs_s(fe_s.dofs_per_cell);

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

// Determination of the volume (area in 2D) of the control volume and
// identification of the first dof associated with the pressure field.

template <int dim>
void
ImmersedFEMGeneralized<dim>::get_area_and_first_pressure_dof ()
{
  area = 0.0;
  typename DoFHandler<dim,dim>::active_cell_iterator
    cell = dh_f.begin_active (),
    endc = dh_f.end ();

  FEValues<dim,dim> fe_v (fe_f,
			  quad_f,
			  update_values |
			  update_JxW_values);

  vector<unsigned int> dofs_f(fe_f.dofs_per_cell);


// Calculate the area of the control volume.
  for(; cell != endc; ++cell)
    {
      fe_v.reinit (cell);
      cell->get_dof_indices (dofs_f);

      for(unsigned int i=0; i < fe_f.dofs_per_cell; ++i)
        {
	  unsigned int comp_i = fe_f.system_to_component_index(i).first;
	  if(comp_i == dim)
            {
	      pressure_dofs.insert(dofs_f[i]);
	      if (dgp_for_p) break;
            }
        }

      for(unsigned int q=0; q<quad_f.size(); ++q) area += fe_v.JxW(q);

    }


// Get the first dof pertaining to pressure.
  constraining_dof = *(pressure_dofs.begin());

}


// Assemblage of the various operators in the formulation along with
// their contribution to the system Jacobian.

template <int dim>
void
ImmersedFEMGeneralized<dim>::residual_and_or_Jacobian
(
  BlockVector<double> &residual,
  BlockSparseMatrix<double> &jacobian,
  const BlockVector<double> &xit,
  const BlockVector<double> &xi,
  const double alpha,
  const double t
)
{
   
// Determine whether or not the calculation of the Jacobian is needed.
  bool update_jacobian = !jacobian.empty();


// Reset the mapping to NULL.
  if(mapping != NULL) delete mapping;


// In a semi-implicit scheme, the position of the immersed body
// coincides with the position of the body at the previous time step.
  if(par.semi_implicit == true)
      mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree,
								previous_xi.block(1),
								dh_s);
  else
    mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree,
							      xi.block(1),
							      dh_s);


// In applying the boundary conditions, we set a scaling factor equal
// to the diameter of the smallest cell in the triangulation of the fluid .
   scaling = GridTools::minimal_cell_diameter(tria_f);

// Initialization of the residual.
  residual = 0;

// If the Jacobian is needed, then it is initialized here.
  if(update_jacobian)
    {
      jacobian.clear();
      assemble_sparsity(*mapping);
      jacobian.reinit(sparsity);
    }

   
//Add the contribution of the source term to the residual vector
   if (par.n_pt_source)
   {
	  get_volume_flux_vector (t);
	  residual.block(0) += volume_flux;
   }

   
// Evaluation of the current values of the external force and of the
// boundary conditions.
  par.force.set_time(t);
  compute_current_bc(t);


// Computation of the maximum number of degrees of freedom one could
// have on a fluid-solid interaction cell.  <b>Rationale</b> the coupling
// of the fluid and solid domains is computed by finding each of the
// fluid cells that interact with a given solid cell. In each
// interaction instance we will be dealing with a total number of
// degrees of freedom that is the sum of the dofs of the current
// solid cell and the dofs of the current fluid cell in the list of
// fluid cells interacting with the solid cell in question.
  unsigned int n_local_dofs = fe_f.dofs_per_cell + fe_s.dofs_per_cell;


// Storage for the local dofs in the fluid and in the solid.
  vector< unsigned int > dofs_f(fe_f.dofs_per_cell);
  vector< unsigned int > dofs_s(fe_s.dofs_per_cell);


// <code>FEValues</code> for the fluid.
  FEValues<dim> fe_f_v (fe_f,
			quad_f,
			update_values |
			update_gradients |
			update_JxW_values |
			update_quadrature_points);


// Number of quadrature points on fluid and solid cells.
  const unsigned int nqpf = quad_f.size();
  const unsigned int nqps = quad_s.size();


// The local residual vector: the largest possible size of this
// vector is <code>n_local_dofs</code>.
  vector<double> local_res(n_local_dofs);
  vector<Vector<double> > local_force(nqpf, Vector<double>(dim+1));
  FullMatrix<double> local_jacobian;
  if(update_jacobian) local_jacobian.reinit(n_local_dofs, n_local_dofs);


// Since we want to solve a system of equations of the form
// $f(\xi', \xi, t) = 0$,
// we need to manage the information in $\xi'$ as though it were
// independent of the information in $\xi$. We do so by defining a
// vector of local degrees of freedom that has a length equal
// to twice the total number of local degrees of freedom.
// This information is stored in the vector <code>local_x</code>.
// <ul>
// <li> The first <code>fe_f.dofs_per_cell</code> elements of
//      <code>local_x</code> contain the elements of $\xi'$
//      corresponding to the current fluid cell.
// <li> The subsequent <code>fe_s.dofs_per_cell</code> elements of
//      <code>local_x</code> contain the elements of $\xi'$ corresponding to the
//      current solid cell.
// <li> The subsequent <code>fe_f.dofs_per_cell</code> elements of
//      <code>local_x</code> contain the elements of $\xi$ corresponding to the
//      current fluid cell.
// <li> The subsequent <code>fe_s.dofs_per_cell</code> elements of
//      <code>local_x</code>.
// <ul>

// Definition of the local dependent variables for the fluid.
  vector<Vector<double> > local_upt(nqpf, Vector<double>(dim+1));
  vector<Vector<double> > local_up (nqpf, Vector<double>(dim+1));
  vector< vector< Tensor<1,dim> > > local_grad_up(
    nqpf,
    vector< Tensor<1,dim> >(dim+1)
  );
  vector< vector< Tensor<1,dim> > > local_grad_upt;
   vector< vector< Tensor<2,dim> > > local_hessian_up;
  unsigned int comp_i = 0, comp_j = 0;

// Initialization of the constants used for compensation of the Lagrange multiplier over the region occupied by the compressible solid:
   double c1 = par.pressure_constant_c1;
   double c2 = par.pressure_constant_c2;
   int sgn_c1 = -1;

 // The mean normal stress for the compressible solid
   double ps = 0.;
   
// Initialization of the local contribution to the pressure
// average.
  double local_average_pressure = 0.0;
  vector<double> local_pressure_coefficient(n_local_dofs);


// ------------------------------------------------------------
// OPERATORS DEFINED OVER THE ENTIRE DOMAIN: BEGIN
// ------------------------------------------------------------

// We now determine the contribution to the residual due to the
// fluid.  This is the standard Navier-Stokes component of the
// problem.  As such, the contributions are to the equation in
// $V'$ and to the equation in $Q'$.


// These iterators point to the first and last active cell of
// the fluid domain.
  typename DoFHandler<dim>::active_cell_iterator
    cell = dh_f.begin_active(),
    endc = dh_f.end();


// Cycle over the cells of the fluid domain.
  for(; cell != endc; ++cell)
    {
      cell->get_dof_indices(dofs_f);


// Re-initialization of the <code>FEValues</code>.
      fe_f_v.reinit(cell);


// Values of the partial derivative of the velocity relative to time
// at the quadrature points on the current fluid cell.  Strictly
// speaking, this vector also includes values of the partial
// derivative of the pressure with respect to time.
      fe_f_v.get_function_values(xit.block(0), local_upt);


// Values of the velocity at the quadrature points on the current
// fluid cell. Strictly speaking, this vector also includes values of
// pressure.
      fe_f_v.get_function_values(xi.block(0), local_up);


// Values of the gradient of the velocity at the quadrature points of
// the current fluid cell.
      fe_f_v.get_function_gradients(xi.block(0), local_grad_up);


// Values of the body force at the quadrature points of the current
// fluid cell.
      par.force.vector_value_list(fe_f_v.get_quadrature_points(), local_force);
	   if (par.csm_test) set_to_zero (local_force); ///:


// Initialization of the local residual and local Jacobian.
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);


// Initialization of the local pressure contribution.
      local_average_pressure = 0.0;
      set_to_zero(local_pressure_coefficient);

      for(unsigned int i=0; i<fe_f.dofs_per_cell; ++i)
	  {
		 comp_i = fe_f.system_to_component_index(i).first;
		 for(unsigned int q=0; q< nqpf; ++q)
			
			// -------------------------------------
			// Contribution to the equation in $V'$.
			// -------------------------------------
			if(comp_i < dim)
			{
			   
			   // $\rho_f [(\partial u/\partial t) - b ] \cdot v - p (\nabla \cdot v)$
			   local_res[i] += par.rho_f
			   * ( local_upt[q](comp_i)
				  -   local_force[q](comp_i) )
			   * fe_f_v.shape_value(i,q)
			   * fe_f_v.JxW(q)
			   - local_up[q](dim)
			   * fe_f_v.shape_grad(i,q)[comp_i]
			   * fe_f_v.JxW(q);
			   if(update_jacobian)
			   {
				  for(unsigned int j=0; j<fe_f.dofs_per_cell; ++j)
				  {
					 comp_j = fe_f.system_to_component_index(j).first;
					 if( comp_i == comp_j )
						local_jacobian(i,j) += par.rho_f
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
			   
			   // $\eta_{f} [\nabla_{x} u + (\nabla_{x} u)^{T}] \cdot \nabla v + \rho_{f} (\nabla_{x} u) u \cdot v$.
			   for(unsigned int d=0; d<dim; ++d)
			   {
				  local_res[i] += par.eta_f
				  * ( local_grad_up[q][comp_i][d]
					 +
					 local_grad_up[q][d][comp_i] )
				  * fe_f_v.shape_grad(i,q)[d]
				  * fe_f_v.JxW(q);
				  
				  if(!par.stokes_flow_like)
					 local_res[i] += par.rho_f
						* local_grad_up[q][comp_i][d]
						* local_up[q](d)
						* fe_f_v.shape_value(i,q)
						* fe_f_v.JxW(q);
			   }
			   if( update_jacobian )
			   {
				  for(unsigned int j=0; j<fe_f.dofs_per_cell; ++j)
				  {
					 comp_j = fe_f.system_to_component_index(j).first;
					 if( comp_j == comp_i )
						for( unsigned int d = 0; d < dim; ++d )
						{
						   local_jacobian(i,j)  += par.eta_f
						   * fe_f_v.shape_grad(i,q)[d]
						   * fe_f_v.shape_grad(j,q)[d]
						   * fe_f_v.JxW(q);
						   
						   if(!par.stokes_flow_like)
							  local_jacobian(i,j)  += par.rho_f
												   * fe_f_v.shape_value(i,q)
												   * local_up[q](d)
												   * fe_f_v.shape_grad(j,q)[d]
												   * fe_f_v.JxW(q);
						}
					 if(comp_j < dim)
					 {
						local_jacobian(i,j)   += par.eta_f
						* fe_f_v.shape_grad(i,q)[comp_j]
						* fe_f_v.shape_grad(j,q)[comp_i]
						* fe_f_v.JxW(q);
						
						if(!par.stokes_flow_like)
						   local_jacobian(i,j)  += par.rho_f
												* local_grad_up[q][comp_i][comp_j]
											 * fe_f_v.shape_value(i,q)
											 * fe_f_v.shape_value(j,q)
											 * fe_f_v.JxW(q);
					 }
				  }
			   }
			}
			else
			{
			   
			   // ------------------------------------
			   // Contribution to the equation in Q'.
			   // ------------------------------------
			   
			   // $-q (\nabla_{x} \cdot u)$
			   for(unsigned int d=0; d<dim; ++d)
				  local_res[i] -= local_grad_up[q][d][d]
				  * fe_f_v.shape_value(i,q)
				  * fe_f_v.JxW(q);
			   if( update_jacobian )
				  for(unsigned int j=0; j<fe_f.dofs_per_cell; ++j)
				  {
					 comp_j = fe_f.system_to_component_index(j).first;
					 if( comp_j < dim )
						local_jacobian(i,j) -= fe_f_v.shape_value(i,q)
						* fe_f_v.shape_grad(j,q)[comp_j]
						* fe_f_v.JxW(q);
				  }
			   
			   if (par.all_DBC && !par.fix_pressure)
			   {
				  if(
					 !dgp_for_p
					 ||
					 (dgp_for_p && (fe_f.system_to_component_index(i).second==0))
					 )
				  {
					 local_average_pressure += xi.block(0)(dofs_f[i])
					 *fe_f_v.shape_value(i,q)
					 *fe_f_v.JxW(q);
					 if(update_jacobian)
					 {
						local_pressure_coefficient[i] += fe_f_v.shape_value(i,q)
						*fe_f_v.JxW(q);
					 }
				  }
			   }
			}
	  }

// Apply boundary conditions.
      apply_constraints (local_res,
			 local_jacobian,
			 xi.block(0),
			 dofs_f,
			 0);


// Now the contribution to the residual due to the current cell
// is assembled into the global system's residual.
      distribute_residual(residual.block(0), local_res, dofs_f, 0);
      if(update_jacobian)
	distribute_jacobian (JF.block(0,0),
			     local_jacobian,
			     dofs_f,
			     dofs_f,
			     0,
			     0);

      if(par.all_DBC && !par.fix_pressure && !par.solid_is_compressible)
        {
	  distribute_constraint_on_pressure (residual.block(0),
					     local_average_pressure);

	  if(update_jacobian)
	    distribute_constraint_on_pressure (jacobian.block(0,0),
					       local_pressure_coefficient,
					       dofs_f,
					       0);
        }
    }

  //: SR--- For NS component only, we now just return :)
   if (par.only_NS)
   {
	  return;
	  cout<<" We have returned right?"<<endl;
   }
// -----------------------------------------
// OPERATORS DEFINED OVER ENTIRE DOMAIN: END
// -----------------------------------------


// -------------------------------------------------
// OPERATORS DEFINED OVER THE IMMERSED DOMAIN: BEGIN
// -------------------------------------------------
  
// We distinguish two orders of organization:
//  <ol>
// <li> we have a cycle
// over the cells of the immersed domain.  For each cell of the
// immersed domain we determine the cells in the fluid domain
// interacting with the cell in question.  Then we cycle over each of
// the fluid cell.
//  
// <li> The operators defined over the immersed
// domain contribute to all three of the equations forming the
// problem.  We group the operators in question by equation.
// Specifically, we first deal with the terms that contribute to the
// equation in $V'$, then we deal with the terms that contribute to $Q'$,
// and finally we deal with the terms that contribute to $Y'$.
// </ol>
// <b>Note:</b> In the equation in $Y'$ there is contribution that does
// not arise from the interaction of solid and fluid.



// Representation of the velocity and pressure in the control volume
// as a field.
  FEFieldFunction<dim, DoFHandler<dim>, Vector<double> >
    up_field (dh_f, xi.block(0));


// Containers to store the information on the interaction of the
// current solid cell with the corresponding set of fluid cells that
// happen to contain the quadrature points of the solid cell in
// question.
  vector< typename DoFHandler<dim>::active_cell_iterator > fluid_cells;
  vector< vector< Point< dim > > > fluid_qpoints;
  vector< vector< unsigned int> > fluid_maps;


// Local storage of the
// <ul>
//  <li> velocity in the solid ($\partial w/\partial t$): <code>local_Wt</code>;
//  <li> displacement in the solid ($w$): <code>local_W</code>;
//  <li> first Piola-Kirchhoff stress: <code>Pe</code>;
//  <li> deformation gradient ($F$): <code>F</code>;
//  <li> the determinant of the deformation gradient ($J$): <code>J</code>;  
//  <li> inverse transpose of the deformation gradient ($F^{-T}$): <code>invFT</code>;
//  <li> $P_{s}^{e} F^{T}$, which is the work conjugate of the velocity
//       gradient when measured over the deformed configuration:
//       <code>PeFT</code>;
//  <li> Frechet derivative of $P_{s}^{e} F^{T}$ with respect to degrees of
//    freedom in a solid cell: <code>DPeFT_dxi</code>.
// </ul>
  vector<Vector<double> > local_Wt(nqps, Vector<double>(dim));
  vector<Vector<double> > local_W (nqps, Vector<double>(dim));
  vector<Tensor<2,dim,double> > Pe(nqps, Tensor<2,dim,double>());
  vector<Tensor<2,dim,double> > F(nqps, Tensor<2,dim,double>());
  vector<double> local_J(nqps); 
  vector<Tensor<2,dim,double> > local_invFT(nqps, Tensor<2,dim,double>());
  Tensor<2,dim,double> PeFT;
  vector< vector<Tensor<2,dim,double> > > DPeFT_dxi;
  if(update_jacobian)
    {
      DPeFT_dxi.resize(nqps, vector< Tensor<2,dim,double> >
		       (fe_s.dofs_per_cell, Tensor<2,dim,double>()));
    }
   
   //SR: If the solid is compressible then we also need to store the following:
   // <ul>
   // <li> divergence of the velocity
   // <li> the mean elastic stress in the solis
   // </ul>
   vector<double> local_div_u;


// Initialization of the elastic operator of the immersed
// domain.
  A_gamma = 0.0;

// Definition of the local contributions to $A_{\gamma}$ and the product of
// the inverse of the mass matrix of the immersed domain with $A_{\gamma}$.
   Vector<double> local_A_gamma;
   Vector<double> local_M_gamma3_inv_A_gamma;
   if (par.use_spread)
   {
	  local_A_gamma.reinit(fe_s.dofs_per_cell);
	  local_M_gamma3_inv_A_gamma.reinit(fe_s.dofs_per_cell);
   }

// This information is used in finding what fluid cell contain the
// solid domain at the current time.
  FEValues<dim,dim> fe_v_s_mapped (*mapping,
				   fe_s,
				   quad_s,
				   update_quadrature_points);


// <code>FEValues</code> to carry out integrations over the solid domain.
  FEValues<dim,dim> fe_v_s(fe_s,
			   quad_s,
			   update_quadrature_points |
			   update_values |
			   update_gradients |
			   update_JxW_values);


// Iterators pointing to the beginning and end cells
// of the active triangulation for the solid domain.
  typename DoFHandler<dim,dim>::active_cell_iterator
    cell_s = dh_s.begin_active(),
    endc_s = dh_s.end();

if (par.use_spread)
{
// Now we cycle over the cells of the solid domain to evaluate $A_{\gamma}$
// and $M_{\gamma 3}^{-1} A_{\gamma}$.
  for(; cell_s != endc_s; ++cell_s)
    {
      fe_v_s.reinit (cell_s);
      cell_s->get_dof_indices (dofs_s);
      get_Agamma_values (fe_v_s, dofs_s, xi.block(1), local_A_gamma);
      A_gamma.add (dofs_s, local_A_gamma);
    }

  M_gamma3_inv_A_gamma = A_gamma;
  M_gamma3_inv.solve (M_gamma3_inv_A_gamma);
}

// -----------------------------------------------
// Cycle over the cells of the solid domain: BEGIN
// -----------------------------------------------
  for(cell_s = dh_s.begin_active(); cell_s != endc_s; ++cell_s)
    {
      fe_v_s_mapped.reinit(cell_s);
      fe_v_s.reinit(cell_s);
      cell_s->get_dof_indices(dofs_s);


// Localization of the current independent variables for the immersed
// domain.
      fe_v_s.get_function_values (xit.block(1), local_Wt);
      fe_v_s.get_function_values ( xi.block(1), local_W);
	  if (par.use_spread)
		 localize (local_M_gamma3_inv_A_gamma, M_gamma3_inv_A_gamma, dofs_s);
      get_Pe_F_and_DPeFT_dxi_values (fe_v_s,
				     dofs_s,
				     xi.block(1),
				     update_jacobian,
				     Pe,
				     F,
				     DPeFT_dxi);
	   
	   get_inverse_transpose(F, local_invFT);

	   
// Calculation of the determinant of the deformation gradient at the quadrature 
// points of the solid.
	   for(unsigned int qt = 0; qt < nqps; ++qt)
		  local_J [qt] = determinant(F[qt]);

// Coupling between fluid and solid.  Identification of the fluid
// cells containing the quadrature points on the current solid cell.
      up_field.compute_point_locations (fe_v_s_mapped.get_quadrature_points(),
					fluid_cells,
					fluid_qpoints,
					fluid_maps);

	   set_to_zero(local_force);
      local_force.resize (nqps, Vector<double>(dim+1));
      par.force.vector_value_list (fe_v_s_mapped.get_quadrature_points(),
				   local_force);

// Cycle over all of the fluid cells that happen to contain some of
// the the quadrature points of the current solid cell.
      for(unsigned int c=0; c<fluid_cells.size(); ++c)
	  {
		 fluid_cells[c]->get_dof_indices (dofs_f);
		 
		 
		 // Local <code>FEValues</code> of the fluid
		 Quadrature<dim> local_quad (fluid_qpoints[c]);
		 FEValues<dim> local_fe_f_v (fe_f,
									 local_quad,
									 update_values |
									 update_gradients |
									 update_hessians);
		 local_fe_f_v.reinit(fluid_cells[c]);
		 
		 
		 // Construction of the values at the quadrature points of the current
		 // solid cell of the velocity of the fluid.
		 set_to_zero(local_up);
		 local_up.resize (local_quad.size(), Vector<double>(dim+1));
		 local_fe_f_v.get_function_values (xi.block(0), local_up);
		 
		 set_to_zero(local_upt);
		 local_upt.resize (local_quad.size(), Vector<double>(dim+1));
		 local_fe_f_v.get_function_values (xit.block(0), local_upt);
		 
		
		 // Construction of the values at the quadrature points of the current
		 // solid cell of the gradient of velocity of the fluid.
		 set_to_zero(local_grad_up);
		 local_grad_up.resize (local_quad.size(), 
							   vector< Tensor<1,dim> >(dim+1)
							   );
		 local_fe_f_v.get_function_gradients (xi.block(0), local_grad_up);
		 
		 if(!par.semi_implicit)
		 {
			set_to_zero(local_grad_upt);
			local_grad_upt.resize (local_quad.size(),
								   vector< Tensor<1,dim> > (dim+1)
								   );
			local_fe_f_v.get_function_gradients (xit.block(0), local_grad_upt);
			
			
			set_to_zero(local_hessian_up);
			local_hessian_up.resize (local_quad.size(),
								   vector< Tensor<2,dim> > (dim+1)
								   );
			local_fe_f_v.get_function_hessians (xi.block(0), local_hessian_up);
			
			
		 }
		 
		 
		 // Construction of the values at the quadrature points of the current
		 // solid cell of the divergence of velocity of the fluid.
		 // Note that this is required only when the solid is compressible
		 set_to_zero(local_div_u);
		 local_div_u.resize(local_quad.size());
		 for(unsigned int qt = 0; qt < local_quad.size(); ++qt)
			for(unsigned int k= 0; k < dim; ++k)
			   local_div_u[qt] += local_grad_up[qt][k][k];
		 
		 // A bit of nomenclature:
		 // <dl>
		 // <dt>Equation in $V'$</dt>
		 //     <dd> Assemblage of the terms in the equation in $V'$ that
		 //          are defined over $B$.</dd>
		 
		 // <dt>Equation in $Y'$</dt> 
		 //     <dd> Assemblage of the terms in the equation in $Y'$ that involve
		 //          the velocity $u$. </dd>
		 // </dl>
		 
		 
		 
		 
		 // Equation in $V'$: initialization of residual.
		 set_to_zero(local_res);
		 if(update_jacobian) set_to_zero(local_jacobian);
		 
		 // Equation in $V'$: begin cycle over the quadrature points of the 
		 // solid cell that happen to be in this fluid cell
		 for(unsigned int q=0; q<local_quad.size(); ++q)
		 {
			// Quadrature point on the <i>mapped</i> solid ($B_{t}$).
			unsigned int &qs = fluid_maps[c][q];
			
		 
			if((!par.semi_implicit) || (!par.use_spread) || par.solid_is_compressible)
			   contract (PeFT, Pe[qs], 2, F[qs], 2);
			
			
			//Calculation of the mean elastic stress of a compressible solid
			if(par.solid_is_compressible) 
			{
			   ps = - (trace (PeFT)
					   / determinant(F[qs])
					   +
					   2.0
					   * par.eta_s
					   * local_div_u[q]
					   )/dim ;
			}
			
			//Begin cycle over the dofs of the fluid cell
			for(unsigned int i=0; i<fe_f.dofs_per_cell; ++i)
			{
			   comp_i = fe_f.system_to_component_index(i).first;
			   if(comp_i < dim)
			   {
				  // Contribution due to the elastic component of the stress response
				  // function in the solid:  $P_{s}^{e} F^{T} \cdot \nabla_{x} v$.  
				  if (!par.use_spread)
				  {
					 local_res[i] += (PeFT[comp_i]
									  * local_fe_f_v.shape_grad(i,q))
					 * fe_v_s.JxW(qs);
					 if(update_jacobian)
						// Recall that the Hessian is symmetric.
					 {
						for( unsigned int j = 0; j < fe_s.dofs_per_cell; ++j )
						{
						   unsigned int wj = j + fe_f.dofs_per_cell;
						   unsigned int comp_j = fe_s.system_to_component_index(j).first;
						   
						   local_jacobian(i,wj) += ( DPeFT_dxi[qs][j][comp_i]
													* local_fe_f_v.shape_grad(i,q) )
						   * fe_v_s.JxW(qs);
						   if( !par.semi_implicit )
							  local_jacobian(i,wj) += ( PeFT[comp_i]
													   * local_fe_f_v.shape_hessian(i,q)[comp_j])
							  * fe_v_s.shape_value(j,qs)
							  * fe_v_s.JxW(qs);
						}
					 }
				  }
				  else
				  {
					 for( unsigned int j = 0; j < fe_s.dofs_per_cell; ++j )
						// The spread operator
					 {
						unsigned int comp_j = fe_s.system_to_component_index(j).first;
						if (comp_i == comp_j)
						   local_res[i] += par.Phi_B
						   * local_fe_f_v.shape_value(i,q)
						   * fe_v_s.shape_value(j, qs)
						   * local_M_gamma3_inv_A_gamma(j)
						   * fe_v_s.JxW(qs);
						
						if(update_jacobian)
						{
						   unsigned int wj = j + fe_f.dofs_per_cell;
						   
						   local_jacobian(i,wj) += ( DPeFT_dxi[qs][j][comp_i]
													* local_fe_f_v.shape_grad(i,q) )
						   * fe_v_s.JxW(qs);
						   if( !par.semi_implicit )
							  local_jacobian(i,wj) += ( PeFT[comp_i]
													   *
													   local_fe_f_v.shape_hessian(i,q)[comp_j])
							  * fe_v_s.shape_value(j,qs)
							  * fe_v_s.JxW(qs);
						}
					 }
				  }
				  
				  
				  // If solid is incompressible and its density and/or its 
				  // viscosity is different from that of the fluid then the 
				  // contribution due to these difference over B must be 
				  // accounted for. On the other hand, additional contributions  
				  // must be considered when the solid is compressible.
				  //xr if( !par.same_density || par.solid_is_compressible) //:Terms 5 & 6
				  //xr{
					 // $ [( \rho_{s} - J \rho_f) (\partial u/\partial t) - b ) +  \rho_{s} (\nabla_{x} u ) \partial w/\partial t - \rho_{f} (\nabla_{x} u) u ] \cdot v 
					 local_res[i] +=  (par.rho_s
									   *(local_upt[q](comp_i) 
										   - local_force[qs](comp_i))
									   - local_J[qs]
									   * par.rho_f
									   * (local_upt[q](comp_i) 
										  - local_force[qs](comp_i)
										  * (par.csm_test ? 0.0: 1.0))
									   )
									   *local_fe_f_v.shape_value(i,q)
									   *fe_v_s.JxW(qs);
									   
					 for(unsigned int k=0; k<dim; ++k)
						local_res[i] += local_grad_up[q][comp_i][k]
									   * ( par.rho_s
										 * local_Wt[qs](k)
										 - 
										 par.rho_f
										 * local_J[qs]
										 * local_up[q](k)
										 )
									   *local_fe_f_v.shape_value(i,q)
									   *fe_v_s.JxW(qs);

					 if (update_jacobian)
					 {
						for(unsigned int j=0; j < dofs_f.size(); ++j)
						{
						   comp_j = fe_f.system_to_component_index(j).first;
						   
						   if(comp_j < dim)
						   {
							  if (comp_j == comp_i)
							  {
								 //: (rho_s-rho_f*J)*del_u'.v of del_M_alpha1"(4)"
								 local_jacobian(i,j) += alpha
													  * (par.rho_s 
														 - par.rho_f
														 * local_J[qs]
														 )
													  * local_fe_f_v.shape_value(j, q)
													  * local_fe_f_v.shape_value(i, q)
													  * fe_v_s.JxW(qs);
								 
								 
								 //: [rho_s (grad_delu) w'- rho_f*J*(grad_delu) u].v of del_N_alpha1"(3&7)"
								 if(!par.stokes_flow_like)
									for(unsigned int k=0; k<dim; ++k)
									   local_jacobian(i,j) += local_fe_f_v.shape_grad(j, q)[k]
															* (
															   par.rho_s
															   * local_Wt[qs](k)
															   - par.rho_f
															   * local_J[qs]
															   * local_up[q](k)
															   )
															* local_fe_f_v.shape_value(i, q)
															* fe_v_s.JxW(qs);
							  
							  }
							  
							  
							  //: -rho_f*J*(grad_u del_u).v of del_N_alpha1"(8)" 
							  if(!par.stokes_flow_like)
								 local_jacobian(i,j) -= par.rho_f
													  * local_J[qs]
													  * local_grad_up[q][comp_i][comp_j]
													  * local_fe_f_v.shape_value(j, q)
													  * local_fe_f_v.shape_value(i, q)
													  * fe_v_s.JxW(qs);
							  
						   
						   }
						}
						
						for(unsigned int j=0; j < dofs_s.size(); ++j)
						{
						   unsigned int wj = j + fe_f.dofs_per_cell;
						   
						   comp_j = fe_s.system_to_component_index(j).first;
						   
						   //: -rho_f*J*(F^(-T):Grad_del_w)*(u'-b).v of del_M_alpha1"(1)"
						   local_jacobian(i,wj) -= fe_v_s.JxW(qs)
												   *( par.rho_f
													 * local_J[qs]
													 * (local_invFT[qs][comp_j]
														* fe_v_s.shape_grad(j, qs))
													 * (local_upt[q](comp_i)
														- local_force[qs](comp_i)
														* (par.csm_test ? 0.0: 1.0))
													  )*local_fe_f_v.shape_value(i, q);
						   
						   //: (rho_s*(grad_u del_w').v of del_N_alpha1"(1)"
						   if ( !par.stokes_flow_like)
						   {
							  local_jacobian(i,wj) += fe_v_s.JxW(qs)
													 * ( par.rho_s
														* local_grad_up[q][comp_i][comp_j]
														* alpha
														* fe_v_s.shape_value(j, qs)
													  )*local_fe_f_v.shape_value(i, q);
						   
						   
						   //: -rho_f*J*(F^(-T):Grad_delw)*grad_u_u .v of del_N_alpha1"(4)"
						   for(unsigned int k=0; k<dim; ++k)
							  local_jacobian(i,wj) -= fe_v_s.JxW(qs)
												   * par.rho_f
												   * local_J[qs]
												   * (local_invFT[qs][comp_j]
													  * fe_v_s.shape_grad(j, qs))
												   * local_grad_up[q][comp_i][k]
												   * local_up[q](k)
												   * local_fe_f_v.shape_value(i, q);
							  
						   }
						   
						   if ( !par.semi_implicit)//: Need to change for csm test
						   {
							  //: (rho_s - rho_f*J)*((grad u' del_w).v+ (u'-b).(grad v del_w)) 
							  //: of del_M_alpha1"(2&3)"
							  local_jacobian(i,wj) += fe_v_s.JxW(qs)
												   * ( par.rho_s 
													  - par.rho_f
													  * local_J[qs])
												   * ( local_grad_upt[q][comp_i][comp_j]
													  * local_fe_f_v.shape_value(i, q)
													  + ( local_upt[q](comp_i)
														 - local_force[qs](comp_i))
													  * local_fe_f_v.shape_grad(i, q)[comp_j]) 
													  * fe_v_s.shape_value(j, qs);
							  
							  for(unsigned int k=0; k<dim; ++k)
							  {
								 //: rho_s*((grad_grad_u del_w)w').v of del_M_alpha1"(2)"
								 local_jacobian(i,wj) += fe_v_s.JxW(qs)
														 * par.rho_s
														 * (local_hessian_up[q][comp_i][comp_j][k]
															*local_Wt[qs](k)
															*fe_v_s.shape_value(j, qs))
														 * local_fe_f_v.shape_value(i, q);
								 
								 //: (rho_s*(grad_u w') - rho_f*J*(grad_u u)).(grad v del_w) of del_N_alpha1"(9&10)"
								 //: -rho_f*J*((grad_gradu_del_w)u + grad_u(grad_u_del_w)).v of del_N_alpha1"(5&6)"					 
								 if(!par.stokes_flow_like)
									local_jacobian(i,wj) += fe_v_s.JxW(qs)
															*(
															  local_grad_up[q][comp_i][k]
															  *( par.rho_s
																* local_Wt[qs](k)
																- par.rho_f
																* local_J[qs]
																* local_up[q](k))
															  * local_fe_f_v.shape_grad(i, q)[comp_j]
															  *fe_v_s.shape_value(j, qs)
															- local_J[qs]
															* par.rho_f
															* (local_hessian_up[q][comp_i][comp_j][k]
															   * local_up[q](k)
															   +
															   local_grad_up[q][comp_i][k]
															   * local_grad_up[q][k][comp_j])
															* fe_v_s.shape_value(j, qs)
															* local_fe_f_v.shape_value(i, q)
															);
							  }
						   
						   }
							  
						}

					 }
					 
				  //xr}
				  
				 //xv if(!par.same_viscosity) //:Term 7
				 //xv {
					 // $ J (\eta_{s} - \eta_{f}) [\nabla_{x} u + (\nabla_{x} u)^{T}] \cdot \nabla_{x} v $
					 
					 for(unsigned int k=0; k<dim; ++k) 
						local_res[i] += local_J[qs]
									   *(par.eta_s
										 -par.eta_f)
									   *(local_grad_up[q][comp_i][k]
										 +local_grad_up[q][k][comp_i])
									   *local_fe_f_v.shape_grad(i,q)[k]
									   *fe_v_s.JxW(qs);	
					 
					 if (update_jacobian)
					 {
						for(unsigned int j=0; j < dofs_f.size(); ++j)
						{
						   comp_j = fe_f.system_to_component_index(j).first;
						   
						   if(comp_j < dim)
						   {
							  //: J*(eta_s - eta_f)*((grad_delu)^T + grad_delu): grad_v of del_D_alpha1"(8&7)"
							  local_jacobian(i,j) += local_J[qs]
												   * (par.eta_s
													  - par.eta_f)
												   * (
													  local_fe_f_v.shape_grad(j, q)[comp_i]
													  * local_fe_f_v.shape_grad(i, q)[comp_j]
													  +
													  ((comp_i == comp_j)? 1.0 :0.0)
													  *local_fe_f_v.shape_grad(j, q)
													  * local_fe_f_v.shape_grad(i, q)
													  )
												   * fe_v_s.JxW(qs);		
						   }
						}
						
						for(unsigned int j=0; j < dofs_s.size(); ++j)
						{
						   comp_j = fe_s.system_to_component_index(j).first;
						   unsigned int wj = j + fe_f.dofs_per_cell;
						   
						   for(unsigned int k=0; k<dim; ++k)
						   {
							  
							  //: J*(eta_s-eta_f)*((F^(-T):Grad del_w)*(grad_u+(grad_u)^T):grad_v of del_D_alpha1"(1&2)" 
							  local_jacobian(i,wj) += local_J[qs]
													  * (par.eta_s 
														 - par.eta_f)
													  * ( local_invFT[qs][comp_j]
														 * fe_v_s.shape_grad(j, qs))
													  * ( local_grad_up[q][comp_i][k]
														 + local_grad_up[q][k][comp_i])
													  * local_fe_f_v.shape_grad(i, q)[k]
													  * fe_v_s.JxW(qs);
							  //: J*(eta_s-eta_f)*[ (grad_(grad_u + (grad_u)^T)) del_w : grad_v + (grad_u + (grad_u)^T): grad_grad_v del_w ]  of del_D_alpha1"(3&4 and 5&6)
							  if ( !par.semi_implicit) 
								 local_jacobian(i,wj) += local_J[qs]
														 * (par.eta_s 
															- par.eta_f)
														 * (
															(local_hessian_up[q][comp_i][k][comp_j]
															 +
															 local_hessian_up[q][k][comp_i][comp_j]
															 )
															* fe_v_s.shape_value(j, qs)
															* local_fe_f_v.shape_grad(i, q)[k]
															+
															(local_grad_up[q][comp_i][k]
															 + local_grad_up[q][k][comp_i]
															 )
															*local_fe_f_v.shape_hessian(i, q)[k][comp_j]
															*fe_v_s.shape_value(j, qs)
														  )
														 *fe_v_s.JxW(qs);
						   }
						}
					 }   
				  
				  //xv}
				  // If the solid is compressible then the contribution of the
				  // Lagrange multiplier satisfying incompressibility should
				  // be removed over B
				  if (par.solid_is_compressible) //:Term 17
				  {
					 // $ J p \nabla_{x} \cdot v $

					 local_res[i] += local_J[qs]
									* local_up[q](dim)
									* local_fe_f_v.shape_grad(i,q)[comp_i]
									* fe_v_s.JxW(qs);
					 
					 
					 if (update_jacobian)
					 {
						for (unsigned int j=0; j<dofs_s.size(); ++j)
						{
						   unsigned int wj = j + fe_f.dofs_per_cell;
						   
						   comp_j = fe_s.system_to_component_index(j).first;

						   //: J(F^(-T):Grad_delw) p div_v of del_BT_beta1"(1)" 
						   local_jacobian(i, wj) += local_J[qs]
												* ( local_invFT[qs][comp_j]
												   * fe_v_s.shape_grad (j, qs))
												* local_up[q](dim)
												* local_fe_f_v.shape_grad(i, q)[comp_i]
												* fe_v_s.JxW(qs);
							  
						   //: J*{(grad_p . del_w) div_v + p grad(div_v). delw} of del_BT_beta1"(2&3)"
						   if( !par.semi_implicit)
							  local_jacobian(i, wj) += local_J[qs]
													  * fe_v_s.shape_value(j, qs)
													  *( local_grad_up[q][dim][comp_j]
														 * local_fe_f_v.shape_grad(i, q)[comp_i]
														 +
														 local_up[q](dim)
														 * local_fe_f_v.shape_hessian(i, q)[comp_j][comp_i]
														 )
													  * fe_v_s.JxW(qs); 
								 
						   }
						
						for (unsigned int j=0; j<dofs_f.size(); ++j)
						{
						   comp_j = fe_f.system_to_component_index(j).first;
						   
						   //: J*del_p*div_v del_BT_beta1"(4)"
						   if(comp_j == dim)
							  local_jacobian(i,j) += local_J[qs]
												   * local_fe_f_v.shape_value(j, q)
												   * local_fe_f_v.shape_grad(i, q)[comp_i]
												   * fe_v_s.JxW(qs);
						   
						}

					 }
						   
				  }
				 
			   }			   
			   else if (par.solid_is_compressible)
			   {
				   // Contributions due to the compressibility of the solid //:Terms 14, 15, 16
				  // First we "subtract" the contribution due to div u over B
				  // $+ q J \nabla_{x} \cdot u $ 
				  local_res[i] += local_J[qs]
								 * local_div_u[q]
								 * local_fe_f_v.shape_value(i,q)
								 * fe_v_s.JxW(qs);
				  
				  // Next we add penalty on the pressure over B
				  // $+ c_{1} J (p-c_{2} p_{s}) q $ 
				  local_res[i] +=  (sgn_c1*c1)  
								 * local_J[qs]
								 * (local_up[q](dim)
									- c2
									* ps)
								 * local_fe_f_v.shape_value(i,q)
								 * fe_v_s.JxW(qs);
				  
				  if(update_jacobian)
				  {
					 for(unsigned int j=0; j<dofs_s.size(); ++j)
					 {
						unsigned int wj = j + fe_f.dofs_per_cell;

						comp_j = fe_s.system_to_component_index(j).first;
						
						//: J (F^(-T):Grad_delw) q { div_u + c1 p} of del_B_beta1"(1)" and  del_B_beta2"(1)"  
						local_jacobian(i, wj) += local_J[qs]
												* (local_invFT[qs][comp_j]
												   * fe_v_s.shape_grad(j, qs))
												* local_fe_f_v.shape_value(i, q)
												* ( local_div_u[q]
												   +
												   sgn_c1*c1
												   * local_up[q](dim)
												   )
												* fe_v_s.JxW(qs);
						
						//: c1*(-c2)*(-1/tr_I)* [ J (F^(-T):Grad_delw) tr(T^v_s) + D tr(Pe F^{T}) [delw] ] q of del_E_beta"(1,4&5)" 
						   local_jacobian(i,wj) += sgn_c1*c1
												* (-c2)
												* (-1.0/dim)
												*( local_J[qs]
												  * (local_invFT[qs][comp_j]
													 * fe_v_s.shape_grad(j, qs))
												  * 2.0
												  * par.eta_s
												  * local_div_u[q]
												  +
												  trace (DPeFT_dxi[qs][j])
												  )* local_fe_f_v.shape_value(i, q)
												* fe_v_s.JxW(qs);
						//end if-condition to check if c2 is not zero
						
						if(!par.semi_implicit)
						{
						   //: J q grad_div_u del_w ( 1.0 + c1(-c2)(-1/tr_I) 2 eta_s ) del_B_beta1"(2)" and del_E_beta"(2)"
						   for(unsigned int k=0; k<dim; ++k)
							  local_jacobian(i, wj) += local_J[qs]
													  * local_hessian_up[q][k][comp_j][k] 
													  * fe_v_s.shape_value(j, qs)
													  * local_fe_f_v.shape_value(i, q)
													  * ( 1.0
														 +
														 sgn_c1*c1
														 * (-c2)
														 * (-1.0/dim)
														 * 2.0
														 * par.eta_s)
													  * fe_v_s.JxW(qs);
						   
						   //: J [ div_u grad_q + c1 (q grad_p + p grad_q)] del_w of del_B_beta1"(3)" and del_P_beta2"(2&3)"
						   local_jacobian(i, wj) += local_J[qs]
												   * fe_v_s.shape_value(j, qs)
												   * (local_fe_f_v.shape_grad(i, q)[comp_j]
													  * local_div_u[q]
													  +
													  sgn_c1*c1
													  * (local_grad_up[q][dim][comp_j]
														 * local_fe_f_v.shape_value(i, q) 
														 + 
														 local_up[q](dim)
														 * local_fe_f_v.shape_grad(i, q)[comp_j] 
														 ) 
													  )* fe_v_s.JxW(qs);
						   
						   //: c1(-c2)(-1/tr_I)[ J 2. eta_s div_u + (Pe:FT)] grad_q.del_w of del_E_beta"(6&7)"
						   local_jacobian(i, wj) += sgn_c1*c1
												 * (-c2)
												 * (-1.0/dim)
												 * (local_J[qs]
													*2.0
													* par.eta_s
													* local_div_u[q]
													+ trace(PeFT)
													)
												 * local_fe_f_v.shape_grad(i, q)[comp_j]
												 * fe_v_s.shape_value(j,qs)
												 * fe_v_s.JxW(qs);
						}
					 }
					 
					 for(unsigned int j=0; j<dofs_f.size(); ++j)
					 {
						comp_j = fe_f.system_to_component_index(j).first;
						
						//: J*q*div_del_u ( 1 + c1*(-c2)*(-1/tr(I))*eta_s*2.0) of del_B_beta1"(4)" and del_E_beta"(3)"
						if(comp_j <dim)
						   local_jacobian (i,j) += local_J[qs]
												* local_fe_f_v.shape_value(i, q)
												* local_fe_f_v.shape_grad(j, q)[comp_j]
												* (1.0
												   + sgn_c1*c1
												   * (-c2)
												   * (-1./dim)
												   * par.eta_s
												   * 2.0
												   )
												* fe_v_s.JxW(qs);
						else //: c1*J*del_p*q of del_P_beta2"15(4)"
						   local_jacobian (i,j) += local_J[qs]
												* sgn_c1*c1	
												* local_fe_f_v.shape_value(i, q)
												* local_fe_f_v.shape_value(j, q)
												* fe_v_s.JxW(qs); 
						}
						
					 }
						
			   }
			}		 
		 }
	   
	  
		 // Equation in $V'$ add to global residual
		 apply_constraints(local_res,
						   local_jacobian,
						   xi.block(0),
						   dofs_f,
						   0);
		 distribute_residual(residual.block(0),
							 local_res,
							 dofs_f,
							 0);
		 if( update_jacobian )
		 {
			distribute_jacobian(JF.block(0,1),
								local_jacobian,
								dofs_f,
								dofs_s,
								0,
								fe_f.dofs_per_cell);
			
			distribute_jacobian (JF.block(0,0),
								 local_jacobian,
								 dofs_f,
								 dofs_f,
								 0,
								 0);
		 }
		 // ****************************************************
		 // Equation in $V'$: COMPLETED
		 // Equation in $Y'$: NOT YET COMPLETED
		 // ****************************************************
		 
		 
		 // Equation in $Y'$: initialization of residual.
		 set_to_zero(local_res);
		 if(update_jacobian) set_to_zero(local_jacobian);
		 
		 
		 // Equation in $Y'$: begin cycle over dofs of immersed domain.
		 for(unsigned int i=0; i<fe_s.dofs_per_cell; ++i)
		 {
			unsigned int wi = i + fe_f.dofs_per_cell;
			comp_i = fe_s.system_to_component_index(i).first;
			for(unsigned int q=0; q<local_quad.size(); ++q)
			{
			   unsigned int &qs = fluid_maps[c][q];
			   
			   // $- u(x,t)\big|_{x = s + w(s,t)} \cdot y(s)$.
			   local_res[wi] -= par.Phi_B
			   * local_up[q](comp_i)
			   * fe_v_s.shape_value(i,qs)
			   * fe_v_s.JxW(qs);
			   if( update_jacobian )
			   {
				  for(unsigned int j = 0; j < fe_f.dofs_per_cell; ++j)
				  {
					 comp_j = fe_f.system_to_component_index(j).first;
					 if( comp_i == comp_j )
					 {
						local_jacobian(wi,j) -= par.Phi_B
						* fe_v_s.shape_value(i,qs)
						* local_fe_f_v.shape_value(j,q)
						* fe_v_s.JxW(qs);
					 }
				  }
				  if( !par.semi_implicit )
					 for(unsigned int k = 0; k < fe_s.dofs_per_cell; ++k)
					 {
						unsigned int wk = k + fe_f.dofs_per_cell;
						unsigned int comp_k = fe_s.system_to_component_index(k).first;
						local_jacobian(wi,wk) -= par.Phi_B
						* fe_v_s.shape_value(i,qs)
						* fe_v_s.shape_value(k,qs)
						* local_grad_up[q][comp_i][comp_k]
						* fe_v_s.JxW(qs);
					 }
			   }
				  
			}
		 }
		 
		 
		 // Equation in Y': add to global residual.
/*apply_constraints(local_res,
						   local_jacobian,
						   xi.block(0),
						   dofs_f,
						   0);
 */
		 apply_constraints(local_res,
						   local_jacobian,
						   xi.block(1),
						   dofs_s,
						   fe_f.dofs_per_cell);
		 distribute_residual(residual.block(1),
							 local_res,
							 dofs_s,
							 fe_f.dofs_per_cell);
		 if( update_jacobian )
		 {
			distribute_jacobian (JF.block(1,0),
								 local_jacobian,
								 dofs_s,
								 dofs_f,
								 fe_f.dofs_per_cell,
								 0);
			if( !par.semi_implicit ) distribute_jacobian (JF.block(1,1),
														  local_jacobian,
														  dofs_s,
														  dofs_s,
														  fe_f.dofs_per_cell,
														  fe_f.dofs_per_cell);
		 }
		 
	  } 
		 
		 // ***************************
		 // Equation in $V'$: COMPLETED
		 // Equation in $Y'$: COMPLETED
		 // ***************************

	   

// Here we assemble the term in the equation
// in $Y'$ involving $\partial w/\partial t$: this term does not
// involve any relations concerning the fluid cells.
      set_to_zero(local_res);
      if(update_jacobian) set_to_zero(local_jacobian);

      for(unsigned int i=0; i<fe_s.dofs_per_cell; ++i)
        {
	  comp_i = fe_s.system_to_component_index(i).first;
	  unsigned int wi = i + fe_f.dofs_per_cell;
	  for(unsigned int qs=0; qs<nqps; ++qs)
            {

// $(\partial w/\partial t) \cdot y$.
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

// We now assemble the contribution just computed into the global
// residual.
	   apply_constraints(local_res,
						 local_jacobian,
						 xi.block(1),
						 dofs_s,
						 fe_f.dofs_per_cell);
      distribute_residual (residual.block(1),
			   local_res,
			   dofs_s,
			   fe_f.dofs_per_cell);
      if( update_jacobian ) distribute_jacobian (JF.block(1,1),
						 local_jacobian,
						 dofs_s,
						 dofs_s,
						 fe_f.dofs_per_cell,
						 fe_f.dofs_per_cell);

    }

// Cycle over the cells of the solid domain: END.


// -----------------------------------------------
// OPERATORS DEFINED OVER THE IMMERSED DOMAIN: END
// -----------------------------------------------
}

// Central management of the time stepping scheme.

template <int dim>
void
ImmersedFEMGeneralized<dim>::run ()
{
   double res_norm = 0.0;
   
   const double TOLF = (par.fsi_bm? 1e-8 : 1e-10);

// The variable <code>update_Jacobian</code> is set to true so to have a
// meaningful first update of the solution.
  bool update_Jacobian = true;
   
// The overall cycle over time begins here.
  for(double t = current_time + par.dt; (t - par.T) <= 1e-8; t += par.dt)
    {
	   //------------------TEST----------------------
	   //string ftest_name;
//	   if (abs(t-0.5)<1e-8)
//	   {
//		  if (par.this_is_a_restart)
//			 ftest_name = "-rst.txt";
//		  else 
//			 ftest_name = ".txt";
//		  
//		  ofstream fstream (("test_xi"+ftest_name).c_str());
//		  current_xi.print(fstream);
//		  fstream.close();
//	   }
	   //----------------------TEST END--------------	   
	   current_time = t;
	   ++time_step;
	  	   
// Initialization of two counters for monitoring the progress of the
// nonlinear solver.
      unsigned int       nonlin_iter = 0;
      unsigned int outer_nonlin_iter = 0;

//Impose the Dirichlet boundary conditions pertaining to the current time
// on the state of the system
	  apply_current_bc(current_xi,t);
	   
// The nonlinear solver iteration cycle begins here.
      while(true)
        {

// We view our system of equations to be of the following form:
//
// $f(\xi', \xi, t) = 0, \quad \xi(0) = \xi_{0}$.
//
// Denoting the current time step by $n$, the vector $\xi'(t_{n})$ is
// assumed to be a linear combination of $\xi(t_{i})$, with $i = n - m
// \ldots n$, with $m \le n$. For simplicity, here we implement an implicit
// Euler method, according to which $\xi'(t_{n}) = [\xi(t_{n}) -
// \xi(t_{n-1})]/dt$, where $dt$ is the size of the time step.


// Time derivative of the system's state.
	  current_xit.sadd (0, 1./par.dt, current_xi, -1./par.dt, previous_xi);

	  if (update_Jacobian == true)
            {

// Determine the residual and the Jacobian of the residual.
	      residual_and_or_Jacobian (current_res,
					JF,
					current_xit,
					current_xi,
					1./par.dt,
					t);

		 if(par.only_NS)
				  JF_inv.initialize (JF.block(0,0)); //: SR Inverse of the Jacobian of the (0,0) block only
		 else 
				  JF_inv.initialize (JF);//: Inverse of the Jacobian of the entire system

// Reset the <code>update_Jacobian</code> variable to the value specified
// in the parameter file.
	      update_Jacobian = par.update_jacobian_continuously;
            }
	  else
            {

// Determine the residual but do not update the Jacobian.
	      residual_and_or_Jacobian (current_res,
					dummy_JF,
					current_xit,
					current_xi,
					0,
					t);

            }

		   if(par.only_NS)
			  res_norm = current_res.block(0).l2_norm(); //: Norm of block(0) of the residual vector
		   else
			  res_norm = current_res.l2_norm(); // Norm of the residual.


// Is the norm of the residual sufficiently small?
	  if( res_norm < TOLF )
	  {

// Make a note and advance to the next step.
	      printf (
		" Step %03d, Res:  %-16.3e (converged in %d iterations)\n\n",
		time_step,
		res_norm,
		nonlin_iter
	      );
	      break;
	  }
	  else
	  {

// If the norm of the residual is not sufficiently small, make a note
// of it and compute an update.
	      cout
		<< nonlin_iter
		<< ": "
		<< res_norm
		<< endl;


// To compute the update to the current $\xi$, we first change the sign
// of the current value of the residual ...
	      current_res *= -1;

			   if(par.only_NS)
			   {
				  tmp_vec_n_dofs_up = current_res.block(0);
				  JF_inv.solve(tmp_vec_n_dofs_up);
				  
				  newton_update.block(0) = tmp_vec_n_dofs_up;
			   }
				else
				{
				  
// ... then we compute the update, which is returned by the method
// <code>solve</code> of the object <code>JF_inv</code>. The latter is of class
// <code>SparseDirectUMFPACK</code> and therefore the value of the (negative) of
// the current residual must be supplied in a container of type
// <code>Vector<double></code>.  So, we first transfer the information in
// <code>current_res</code> into temporary storage, and then we carry out the
// computation of the update.
	      tmp_vec_n_total_dofs = current_res;
	      JF_inv.solve(tmp_vec_n_total_dofs);

// Now that we have the updated of the solution into an object of type
// <code>Vector<double></code>, we repackage it into an object of
// type <code>BlockVector</code>.
	      newton_update = tmp_vec_n_total_dofs;
				}

// Finally, we determine the value of the updated solution.
	      current_xi.add(1., newton_update);


// We are here because the solution needed to be updated. The update
// was computed using whatever Jacobian was available.  If, on
// entering this section of the loop, the value of the residual was
// very poor and if the solution's method indicated in the parameter
// file did not call for a continuous update of the Jacobian, now we
// make sure that the Jacobian is updated before computing the next
// solution update.
	      if(res_norm > 1e-2) update_Jacobian = true;
            }


// We are here because the solution needed an update. So, start
// counting how many iterations are needed to converge.  If
// convergence is not achieved in 15 iterations update the Jacobian
// and try again.  The maximum number of 15-iteration cycles is set
// (arbitrarily) to three. The counter for the cycle is
// <code>outer_nonlin_iter</code>.
	  ++nonlin_iter;
	  if(nonlin_iter == 10)
            {
	      update_Jacobian = true;
	      nonlin_iter = 0;
	      outer_nonlin_iter++;
	      printf(
		"   %-16.3e (not converged in 10 iterations. Step %d)\n\n",
		res_norm,
		outer_nonlin_iter
	      );
			 ///  output_step (t, current_xi, time_step*1000, par.dt);//::Mult pt source
            }


// If convergence is not in our destiny, accept defeat, with as much
// grace as it can be mustered, and go home.
	  AssertThrow (outer_nonlin_iter <= 3,
		       ExcMessage ("No convergence in nonlinear solver."));
        }


// We have computed a new solution.  So, we update the state of the
// system and move to the next time step.
      previous_xi = current_xi;
	   previous_time =t;
      output_step (t, current_xi, time_step, par.dt);
	   if (par.fsi_bm)
	   {
		 // if ((time_step==1)||(time_step % par.output_interval==0))
		  fsi_bm_postprocess2();
	   }
      update_Jacobian = par.update_jacobian_continuously;
      if(par.update_jacobian_at_step_beginning) update_Jacobian = true;

    }
// End of the cycle over time.

  if(par.material_model == IFEMParametersGeneralized<dim>::CircumferentialFiberModel)
    calculate_error();

}
// End of <code>run()</code>.


// Writes results to the output file.

template <int dim>
void
ImmersedFEMGeneralized<dim>::output_step
(
  const double t,
  const BlockVector<double> &solution,
  const unsigned int step,
  const double h,
  const bool _output
)
{
  cout
    << "Time "
    << t
    << ", Step "
    << step
    << ", dt = "
    << h
    << endl;

  global_info_file
    << t
    << " ";

  if ((step % par.output_interval==0) || (_output))
  {
	 {
	vector<string> joint_solution_names (dim, "v");
	joint_solution_names.push_back ("p");
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dh_f);
	vector< DataComponentInterpretation::DataComponentInterpretation >
	  component_interpretation (dim+1,
				    DataComponentInterpretation::component_is_part_of_vector);
	component_interpretation[dim]
	  = DataComponentInterpretation::component_is_scalar;

	data_out.add_data_vector (
	  solution.block(0),
	  joint_solution_names,
	  DataOut<dim>::type_dof_data,
	  component_interpretation
	);
	
	data_out.build_patches (par.degree);
	ofstream output ((par.output_name
			  + "-fluid-"
			  + Utilities::int_to_string (step, 5)
			  + ".vtu").c_str());

	data_out.write_vtu (output);
      }
      {

	vector<string> joint_solution_names (dim, "W");
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dh_s);
	vector< DataComponentInterpretation::DataComponentInterpretation >
	  component_interpretation (dim,
				    DataComponentInterpretation::component_is_part_of_vector);

	data_out.add_data_vector (solution.block(1),
				  joint_solution_names,
				  DataOut<dim>::type_dof_data,
				  component_interpretation);
		

	data_out.build_patches (*mapping);
	ofstream output ((par.output_name
			  + "-solid-"
			  + Utilities::int_to_string (step, 5)
			  + ".vtu").c_str());
	data_out.write_vtu (output);
      }
    }
  {


// Assemble in and out flux.
    typename DoFHandler<dim,dim>::active_cell_iterator
      cell = dh_f.begin_active(),
      endc = dh_f.end();
    QGauss<dim-1> face_quad(par.degree+2);
    FEFaceValues<dim,dim> fe_v (fe_f,
				face_quad,
				update_values |
				update_JxW_values |
				update_normal_vectors);

    vector<Vector<double> > local_vp(face_quad.size(),
				     Vector<double>(dim+1));

    double flux=0;
    for(; cell != endc; ++cell)
      for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
	if(cell->face(f)->at_boundary())
	  {
	    fe_v.reinit(cell, f);
	    fe_v.get_function_values(solution.block(0), local_vp);
	    const vector<Point<dim> > &normals = fe_v.get_normal_vectors();
	    for(unsigned int q=0; q<face_quad.size(); ++q)
	      {
		Point<dim> vq;
		for(unsigned int d=0; d<dim; ++d) vq[d] = local_vp[q](d);
		flux += (vq*normals[q])*fe_v.JxW(q);
	      }
	  }
    global_info_file
      << flux
      << " ";
  }
  {


// Compute area of the solid, and location of its center of mass.
    typename DoFHandler<dim,dim>::active_cell_iterator
      cell = dh_s.begin_active(),
      endc = dh_s.end();
    FEValues<dim,dim> fe_v(*mapping, fe_s,
			   quad_s,
			   update_JxW_values |
			   update_quadrature_points);

    vector<Vector<double> > local_X(quad_s.size(),
				    Vector<double>(dim+1));
    double area=0;
    Point<dim> center;
    for(; cell != endc; ++cell)
      {
	fe_v.reinit(cell);
	const vector<Point<dim> > &qpoints = fe_v.get_quadrature_points();
	for(unsigned int q=0; q<quad_s.size(); ++q)
	  {
	    area += fe_v.JxW(q);
	    center += fe_v.JxW(q)*qpoints[q];
	  }
      }
    center /= area;
    global_info_file
      << area
      << " ";
    global_info_file
      << center
      << endl;
  }
}

// Determination of a vector of local dofs representing
//    the field <code>A_gamma</code>.

template <int dim>
void
ImmersedFEMGeneralized<dim>::get_Agamma_values
(
  const FEValues<dim,dim> &fe_v_s,
  const vector< unsigned int > &dofs,
  const Vector<double> &xi,
  Vector<double> &local_A_gamma
)
{
  set_to_zero(local_A_gamma);

  unsigned int qsize = fe_v_s.get_quadrature().size();

  vector< vector< Tensor<1,dim> > >
    H(qsize, vector< Tensor<1,dim> >(dim));
  fe_v_s.get_function_gradients(xi, H);

  vector<Tensor<2,dim,double> > P (qsize, Tensor<2,dim,double>());
  vector<Tensor<2,dim,double> > tmp1;
  vector< vector<Tensor<2,dim,double> > > tmp2;

  get_Pe_F_and_DPeFT_dxi_values (
    fe_v_s,
    dofs,
    xi,
    false,
    P,
    tmp1,
    tmp2
  );

  for( unsigned int qs = 0; qs < qsize; ++qs )
    {
      for (unsigned int k = 0; k < dofs.size(); ++k)
        {
	  unsigned int comp_k = fe_s.system_to_component_index(k).first;


//Agamma = P:Grad_y
	  local_A_gamma (k) +=
	    P[qs][comp_k]*fe_v_s.shape_grad(k, qs)
	    *fe_v_s.JxW(qs);
        }
    }
}

// Value of the product of the 1st Piola-Kirchhoff stress tensor and
// of the transpose of the deformation gradient at a given list of
// quadrature points on a cell of the immersed domain.

template <int dim>
template <class FEVal>
void
ImmersedFEMGeneralized<dim>::get_Pe_F_and_DPeFT_dxi_values (
  const FEVal &fe_v_s,
  const vector< unsigned int > &dofs,
  const Vector<double> &xi,
  const bool update_jacobian,
  vector<Tensor<2,dim,double> > &Pe,
  vector<Tensor<2,dim,double> > &vec_F,
  vector< vector<Tensor<2,dim,double> > > & DPeFT_dxi
)
{
  vector< vector< Tensor<1,dim> > >
    H(Pe.size(), vector< Tensor<1,dim> >(dim));
  fe_v_s.get_function_gradients(xi, H);

  Tensor<2,dim,double> F;

  bool update_vecF = (vec_F.size()!= 0);


// The following variables are used when the
// <code>CircumferentialFiberModel</code> is used.
  Point<dim> p;
  Tensor<1, dim, double> etheta;
  Tensor<2, dim, double> etheta_op_etheta;
  Tensor<2, dim, double> tmp;
   
   double J, beta;

  for( unsigned int qs = 0; qs < Pe.size(); ++qs )
  {
	 for(unsigned int i=0; i <dim; ++i)
	 {
		F[i] = H[qs][i];
		F[i][i] += 1.0;
	 }
	 
	 if (update_vecF)
		vec_F[qs] = F;
	 
	 switch (par.material_model)
	 {
		case IFEMParametersGeneralized<dim>::INH_0:
		   Pe[qs] = par.mu * ( F - transpose( invert(F) ) );
		   if( update_jacobian )
		   {
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
		      {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if( i == comp_k )
						  DPeFT_dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)
						  * F[j];
					   if( j == comp_k )
						  DPeFT_dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)
						  * F[i];
					   DPeFT_dxi[qs][k][i][j] *= par.mu;
					}
		      }
		   }
		   break;
		case IFEMParametersGeneralized<dim>::INH_1 :
		   Pe[qs] = par.mu * F;
		   if( update_jacobian )
		   {
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
		      {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if( i == comp_k )
						  DPeFT_dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)
						  * F[j];
					   if( j == comp_k )
						  DPeFT_dxi[qs][k][i][j] += fe_v_s.shape_grad(k,qs)
						  * F[i];
					   DPeFT_dxi[qs][k][i][j] *= par.mu;
					}
		      }
		   }
		   break;
		case IFEMParametersGeneralized<dim>::CircumferentialFiberModel:
		   p = fe_v_s.quadrature_point(qs) - par.ring_center;
		   
		   // Find the unit vector along the tangential direction
		   etheta[0]=-p[1]/p.norm();
		   etheta[1]= p[0]/p.norm();
		   
		   
		   // Find the tensor product of etheta and etheta
		   outer_product(etheta_op_etheta, etheta, etheta);
		   contract (Pe[qs], F, etheta_op_etheta);
		   Pe[qs] *= par.mu;
		   if( update_jacobian )
		   {
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
		      {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if( i == comp_k )
						  DPeFT_dxi[qs][k][i][j] += (fe_v_s.shape_grad(k,qs)
													 *etheta_op_etheta)*F[j];
					   if( j == comp_k )
						  DPeFT_dxi[qs][k][i][j] += (fe_v_s.shape_grad(k,qs)
													 *etheta_op_etheta)* F[i];
					   DPeFT_dxi[qs][k][i][j] *= par.mu;
					}
		      }
		   }
		   break;
		case IFEMParametersGeneralized<dim>::CNH_W1 :
		   J = determinant(F);

		   beta = par.nu/(1 - 2 * par.nu);

		   tmp = transpose(invert(F));
		   Pe[qs] = par.mu * ( F - std::pow(J, -2.0 * beta) * tmp);
			if( update_jacobian )
			{
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
			  {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
				 {
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if( i == comp_k )
						  DPeFT_dxi[qs][k][i][j] += par.mu
												   * fe_v_s.shape_grad(k,qs)
												   * F[j];
					   if( j == comp_k )
						  DPeFT_dxi[qs][k][i][j] += par.mu
												   * fe_v_s.shape_grad(k,qs)
												   * F[i];
					}
					
					DPeFT_dxi[qs][k][i][i] += 2.0
											* beta
											* par.mu
											* pow(J, -2.0*beta)
											*( tmp[comp_k]
											 * fe_v_s.shape_grad(k,qs));
				 }
			  }
		   }
		   break;
		case IFEMParametersGeneralized<dim>::CNH_W2 :
		   J = determinant(F);
		   beta = par.nu/(1 - 2 * par.nu);
		   
		   tmp = transpose(invert(F));
		   Pe[qs] = par.mu * F - (par.mu + par.tau)*std::pow(J, -2.0 * beta)* tmp;
						
		   if( update_jacobian )
		   {
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
			  {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if( i == comp_k )
						  DPeFT_dxi[qs][k][i][j] += par.mu
												* fe_v_s.shape_grad(k,qs)
												* F[j];
					   if( j == comp_k )
						  DPeFT_dxi[qs][k][i][j] += par.mu
												* fe_v_s.shape_grad(k,qs)
												* F[i];
					   if(i == j)
						  DPeFT_dxi[qs][k][i][j] += 2.0
												   * beta
												   * (par.mu
													  + par.tau)
												   * pow(J, -2.0*beta)
												   * fe_v_s.shape_grad(k,qs)
												   * tmp[comp_k];
					   
					}
			  }
		   }
		   break;
		case IFEMParametersGeneralized<dim>::STVK :
		   // Saint-Venant Kirchhoff material given as: P=F(2*mu*E + lambda*tr(E)*I)
		   // which will be represented here as P=F(2*mu*tmp + beta*tr(tmp)I)
		   beta = 2.0 * par.mu * par.nu/(1.0 - 2.0 *par.nu);
		   
		   tmp = 0.5*(transpose(F)*F);
		   switch (dim)
		   {
			  case 3:
				 tmp[2][2] -= 0.5;
			  case 2:  
				 tmp[0][0] -= 0.5;
				 tmp[1][1] -= 0.5;
		   }
		   
		   Pe[qs] = beta * trace(tmp)*F + 2.0* par.mu *(F * tmp);
		   
		   if( update_jacobian )
		   {
			  for( unsigned int k = 0; k < fe_s.dofs_per_cell; ++k )
			  {
				 DPeFT_dxi[qs][k] = 0.0;
				 unsigned int comp_k = fe_s.system_to_component_index(k).first;
				 
				 for( unsigned int i = 0; i < dim; ++i )
					for( unsigned int j = 0; j < dim; ++j )
					{
					   if(i == comp_k)
						  DPeFT_dxi[qs][k][i][j] += (beta
													 *trace(tmp)
													 *fe_v_s.shape_grad(k,qs)
													 *F[j]
													 + 
													 2.*par.mu
													 * (fe_v_s.shape_grad(k,qs)
														* tmp)
													 *F[j]
													 );
					   if(j == comp_k)
						  DPeFT_dxi[qs][k][i][j] += (beta
													 *trace(tmp)
													 *fe_v_s.shape_grad(k,qs)
													 *F[i]
													 + 
													 2.*par.mu
													 * (fe_v_s.shape_grad(k,qs)
														* tmp)
													 *F[i]
													 );
					   
					   DPeFT_dxi[qs][k][i][j] += beta
												*( fe_v_s.shape_grad(k,qs)
												  * F[comp_k] )
												*( F[i] * F[j])
												+
												par.mu
												*( F[i]
												  * fe_v_s.shape_grad(k,qs) )
												*( F[comp_k] * F[j] )
												+
												par.mu
												*( F[i] * F[comp_k])
												*( fe_v_s.shape_grad(k,qs) 
												   * F[j]);
					}
			  }
		   }	
		   break;
		default:
		   break;
	 }
  }
}


// Determining the inverse transpose of the deformation gradient at all the 
// quadrature points of a solid cell.

template <int dim>
void
ImmersedFEMGeneralized<dim>::get_inverse_transpose
(const vector < Tensor <2, dim> > &F,
 vector < Tensor <2, dim> > &local_invFT)
{
   for(unsigned int q=0; q< F.size(); ++q)
	  local_invFT[q] = transpose(invert(F[q]));
}

   
// Assemblage of the local residual in the global residual.

template <int dim>
void
ImmersedFEMGeneralized<dim>::distribute_residual
(
  Vector<double> &residual,
  const vector<double> &local_res,
  const vector<unsigned int> &dofs_1,
  const unsigned int offset_1
)
{
  for(unsigned int i=0, wi=offset_1; i<dofs_1.size(); ++i,++wi)
    residual(dofs_1[i]) += local_res[wi];
}

// Assemblage of the local Jacobian in the global Jacobian.

template <int dim>
void
ImmersedFEMGeneralized<dim>::distribute_jacobian
(
  SparseMatrix<double> &Jacobian,
  const FullMatrix<double> &local_Jac,
  const vector<unsigned int> &dofs_1,
  const vector<unsigned int> &dofs_2,
  const unsigned int offset_1,
  const unsigned int offset_2
)
{

  for(unsigned int i=0, wi=offset_1; i<dofs_1.size(); ++i,++wi)
    for(unsigned int j=0, wj=offset_2; j<dofs_2.size(); ++j,++wj)
      Jacobian.add(dofs_1[i],dofs_2[j],local_Jac(wi,wj));
}

// Application of constraints to the local residual and to the local
// contribution to the Jacobian.

template <int dim>
void
ImmersedFEMGeneralized<dim>::apply_constraints
(
  vector<double> &local_res,
  FullMatrix<double> &local_jacobian,
  const Vector<double> &value_of_dofs,
  const vector<unsigned int> &dofs,
 unsigned int offset
)
{
   map<unsigned int,double>::iterator it;

if (offset == 0) //: i.e. constraints need to be applied for fluid dofs
  for(unsigned int i=0; i<dofs.size(); ++i)
  {
	 it = par.boundary_values.find(dofs[i]);
	 if(it != par.boundary_values.end() )
	 {

// Setting the value of the residual equal to the difference between
// the current value and the the prescribed value.
	  local_res[i] = scaling * ( value_of_dofs(dofs[i]) - it->second );
	  if( !local_jacobian.empty() )
            {

// Here we simply let the Jacobian know that the current dof is
// actually not a dof.
	      for(unsigned int j=0; j<local_jacobian.n(); ++j)
		local_jacobian(i,j) = 0;
	      local_jacobian(i,i) = scaling;
            }
        }

// Dealing with constraints concerning the pressure field.
	 if(par.all_DBC && !par.fix_pressure)
	 {
		if(dofs[i] == constraining_dof)
		{
			local_res[i] = 0;
			if( !local_jacobian.empty() ) local_jacobian.add_row(i, -1, i);
		 }
	  }
    }
   else //: i.e. constraints need to be applied for solid dofs
	  for(unsigned int i=0, wi = offset; i <dofs.size(); ++i, ++wi)
	  {
		 it = par.boundary_values_solid.find(dofs[i]);
		 if(it != par.boundary_values_solid.end() || par.cfd_test )
		 {
			// Setting the value of the residual equal to the difference between
			// the current value and the the prescribed value.
			local_res[wi] = scaling * ( value_of_dofs(dofs[i]) - 
									   (par.cfd_test ? 0.0 : it->second ));//: SR---For cfd test, the presribed value is zero for all dofs of the solid
			if( !local_jacobian.empty() )
			{
			   // Here we simply let the Jacobian know that the current dof is actually not a dof.
			   for(unsigned int j=0; j<local_jacobian.n(); ++j)
				  local_jacobian(wi,j) = 0;
			   local_jacobian(wi,wi) = scaling;
			}
		 }
	  }			
}

// Assemble the pressure constraint into the residual.
template <int dim>
void
ImmersedFEMGeneralized<dim>::distribute_constraint_on_pressure
(
Vector<double> &residual,
const double average_pressure
)
{
residual(constraining_dof) += average_pressure*scaling/area;
}

// Assemble the pressure constraint into the Jacobian.
template <int dim>
void
ImmersedFEMGeneralized<dim>::distribute_constraint_on_pressure
(
SparseMatrix<double> &jacobian,
const vector<double> &pressure_coefficient,
  const vector<unsigned int> &dofs,
  const unsigned int offset
)
{
  for(unsigned int i=0, wi=offset; i<dofs.size(); ++i,++wi)
    jacobian.add(
      constraining_dof,
      dofs[i],
      pressure_coefficient[wi]*scaling/area
    );

}

// Determination of the dofs for the function
//    <code>M_gamma3_inv_A_gamma</code>.

template <int dim>
void
ImmersedFEMGeneralized<dim>::localize
(
  Vector<double> &local_M_gamma3_inv_A_gamma,
  const Vector<double> &M_gamma3_inv_A_gamma,
  const vector<unsigned int> &dofs
)
{
  for (unsigned int i = 0; i < dofs.size(); ++i)
    local_M_gamma3_inv_A_gamma (i) = M_gamma3_inv_A_gamma(dofs[i]);
}


// Determination of the volume flux vector corresponding to the point source.
template <int dim>
void
ImmersedFEMGeneralized<dim>::get_volume_flux_vector (const double t)
{
   double strength;
   
   for (unsigned int i=0; i < par.n_pt_source; ++i)
   {
	  // Evaluate the source strength at the current time
	  par.pt_source_strength[i]->set_time(t);
   
	  strength = par.pt_source_strength[i]->value(par.pt_source_location[i]);
   
	  VectorTools::create_point_source_vector(dh_f,
											  par.pt_source_location[i],
											  tmp_vec_n_dofs_up);
											 	  
	// What we have obtained using <code>deal.II</code> is this:
	// tmp_vec_n_dofs_up = 
	//	  integral_over_control_volume(Dirac_delta( x - x_source)*shape_fn(i))
	// for all test functions defined over the control volume	  
	// Also, we need to slightly modify the contribution by
	// multiplying it with a factor equal to the source strength over the density of
	// the fluid
	  tmp_vec_n_dofs_up *= strength/par.rho_f;
	  volume_flux += tmp_vec_n_dofs_up;
   }
   // We need to consider only the contribution of the test functions pertaining to 
   // the Lagrange multiplier.
   fill(volume_flux.begin(), volume_flux.begin()+n_dofs_u, 0.0); 
}


// Calculate the error for the equilibrium solution of corresponding
// to a ring with circumferential fibers.

template <int dim>
void
ImmersedFEMGeneralized<dim>::calculate_error () const
{
  ExactSolutionRingWithFibers<dim> exact_sol(par);

  const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
  const ComponentSelectFunction<dim> velocity_mask(
    make_pair(0,dim),
    dim+1
  );

  const QIterated<dim> qiter_err(QTrapez<1>(), par.degree+1);

  Vector<float> difference_per_cell(tria_f.n_active_cells());


  VectorTools::integrate_difference (
    dh_f,
    current_xi.block(0),
    exact_sol,
    difference_per_cell,
    qiter_err,
    VectorTools::L2_norm,
    &velocity_mask
  );
  const double v_l2_norm = difference_per_cell.l2_norm();

  VectorTools::integrate_difference (
    dh_f,
    current_xi.block(0),
    exact_sol,
    difference_per_cell,
    qiter_err,
    VectorTools::H1_seminorm,
    &velocity_mask
  );
  const double v_h1_seminorm = difference_per_cell.l2_norm();

  VectorTools::integrate_difference (
    dh_f,
    current_xi.block(0),
    exact_sol,
    difference_per_cell,
    qiter_err,
    VectorTools::L2_norm,
    &pressure_mask
  );
  const double p_l2_norm = difference_per_cell.l2_norm();
   
  ofstream file_write;
   
   string quad_name;
   switch (par.quad_s_type) 
   {
	  case IFEMParametersGeneralized<dim>::QGauss :
		 quad_name ="QG-"+Utilities::int_to_string(par.quad_s_degree) ;
		 break;
	  case IFEMParametersGeneralized<dim>::Qiter_Qtrapez :
		 quad_name ="QI-QT-"+Utilities::int_to_string(par.quad_s_degree) ;
		 break;
	  case IFEMParametersGeneralized<dim>::Qiter_Qmidpoint :
		 quad_name = "QI-QM-"+Utilities::int_to_string(par.quad_s_degree);
		 break;
	  default:
		 break;
   }
   string filename;
   if(dgp_for_p)
	  filename = "hello_world_error_norm_pFEDGP_"+quad_name+".dat";
   else
	  filename = "hello_world_error_norm_pFEQ_"+quad_name+".dat";
	  
  file_write.open(filename.c_str(), ios::out |ios::app);
  if (file_write.is_open())
    {
      file_write.unsetf(ios::floatfield);
      file_write
	<< "- & "
	<< setw(4)
	<< tria_s.n_active_cells()
	<< " & "
	<< setw(6)
	<< n_dofs_W
	<< " & "
	<< setw(4)
	<< tria_f.n_active_cells()
	<< " & "
	<< setw(6)
	<< n_dofs_up
	<< scientific
	<< setprecision(5)
	<< " & "
	<< setw(8)
	<< v_l2_norm
	<< " &-& "
	<< setw(8)
	<< v_h1_seminorm
	<< " &-& "
	<< setw(8)
	<< p_l2_norm
	<< " &- \\\\"
	<< endl;
    }
  file_write.close();

}


//Calculation and output of tip displacement of the flag and lift-drag on the 
// cylinder+flag in the Turek-Hron FSI benchmark test
template <int dim>
void ImmersedFEMGeneralized<dim>::fsi_bm_postprocess()
{
   //: Some geometric features of the benchmark test(s)
   const Point<dim> center_cyl (0.2, 0.2); //: Center of the cylinder
   const double D_cyl = 0.1; //: Diameter of the cylinder
   const double H = 0.41; //: Height of the channel
   const double l_flag = 0.35; //: Length of the flag
   
   //Compute the the displacement of the pt. A (i.e. the midpt of the end of the tail)
   Point<dim> point_A (center_cyl[0]+0.5*D_cyl+l_flag, center_cyl[1]);
   Point<dim> point_B (center_cyl[0]-0.5*D_cyl, center_cyl[1]);
   Vector<double> disp_A (dim);
   Vector<double> sol_B (dim+1);
   double pressure_A = 0.0;
   double U_avg = 0.0;
   double c_D = 0.0;
   double c_L = 0.;

   if(par.cfd_test)
   {
	  //: Point A now corresponds to midpoint of the aft edge of the cylinder
	  point_A[0] -= l_flag; 
	  
	  //: Extract the value of pressure at point A
	  VectorTools::point_value(dh_f, 
							   current_xi.block(0), 
							   point_A, 
							   sol_B);
	  pressure_A = sol_B(dim);
	  
	  //: Evaluate the average value of the inflow velocity for t=4s for CFDBM1-3
	  par.u_g.set_time(4.0);
	  U_avg = par.u_g.value(Point<dim>(0., 0.5*H))/1.5;
	  par.u_g.set_time(current_time);
	  
   }
   else
	  VectorTools::point_value(dh_s, 
							current_xi.block(1), 
							point_A, 
							disp_A);
   
   VectorTools::point_value(dh_f, 
							current_xi.block(0), 
							point_B, 
							sol_B);
   
   //-------------------- Loop over FLUID CELLS-----------------//
   //Calculating the drag and lift values corresponding to the free surface of the cylinder
   
   typename DoFHandler<dim,dim>::active_cell_iterator
   cell = dh_f.begin_active(), endc = dh_f.end();
   
   QGauss <dim-1> quad_face (par.degree+2);
   unsigned int n_qpf = quad_face.size();
   
   FEFaceValues<dim> fe_f_face_v (fe_f, quad_face,
								  update_values | update_gradients |
								  update_normal_vectors |
								  update_quadrature_points |
								  update_JxW_values);
   
   unsigned int n_dofs = fe_f.dofs_per_cell;
   vector <unsigned int> dofs_f (n_dofs);
   
   vector < Vector <double> > sol_f (n_qpf, Vector <double> (dim+1));
   vector < vector< Tensor <1, dim> > > sol_grad_f (n_qpf, vector< Tensor <1,dim> > (dim+1));
   
   Tensor < 1, dim > drag_lift_cyl;
   Tensor < 1, dim > drag_lift_flag_f;
   Tensor < 1, dim > drag_lift_flag_s;
   Tensor < 1, dim > drag_lift_flag_avg;
   
   
   //: Variables needed for Turek-style calculations
   double gradn_u_t;
   Point<dim> normal_vector;
   Point<dim> tangent_vector;
   vector <double> drag_lift_turekstyle(dim);
   double c_L_turekstyle = 0.;
   double c_D_turekstyle = 0.;
   //:-----------------------
   unsigned int c_no = 0;
   for(; cell !=endc; ++cell, ++c_no)
   { 
	  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
	  {
		 if (cell->face(face)->at_boundary() 
			 && (cell->face(face)->boundary_indicator() == 80
			 || (par.cfd_test && cell->face(face)->boundary_indicator() == 81)
				 )
			)
		 {
			cell->get_dof_indices (dofs_f);
			
			fe_f_face_v.reinit (cell, face);
			
			fe_f_face_v.get_function_values (current_xi.block(0), sol_f);
			fe_f_face_v.get_function_gradients (current_xi.block(0), sol_grad_f);
			
			for(unsigned int q = 0; q < n_qpf; ++q)//loop over quadrature pts
			{
			   for(unsigned int i = 0; i < dim; ++i)//loop over dim
				  for(unsigned int j = 0; j < dim; ++j)//loop over dim
					 drag_lift_cyl[i] += //(T_f - p I)*(-n da) //-n since +n is outward wrt solid cell boundary
										  (par.eta_f 
											 *(sol_grad_f[q][i][j]
												+ sol_grad_f[q][j][i])
											  - (i == j ? sol_f[q](dim) : 0.0)
											 )
											 *(-fe_f_face_v.normal_vector(q)(j))
											 *fe_f_face_v.JxW(q);
			   //Turek-style calculations to follow:
			   normal_vector = -fe_f_face_v.normal_vector(q);
			   tangent_vector[0] = normal_vector[1];
			   tangent_vector[1] = -normal_vector[0];
			   gradn_u_t = 0.0;
			   for(unsigned int i = 0; i < dim; ++i)
				  gradn_u_t +=tangent_vector[i]
							  *(sol_grad_f[q][i]
								*normal_vector);
			   
			   drag_lift_turekstyle[0] += (-sol_f[q](dim)
										  * normal_vector[0]
										  + par.eta_f
										  * gradn_u_t
										  * normal_vector[1])
										  *fe_f_face_v.JxW(q);
			   drag_lift_turekstyle[1] += (-sol_f[q](dim)
										  * normal_vector[1]
										  - par.eta_f
										  * gradn_u_t
										  * normal_vector[0])
										  *fe_f_face_v.JxW(q);

			}//loop over q
		 }//if cond.
	  }//loop over faces
   }//loop over cell
   
   ///-------------------- Loop over FLUID CELLS (end)-----------------//
   if(par.cfd_test && (abs(U_avg)>1e-8))
   {
	  c_D = (2./(par.rho_f*D_cyl))*(drag_lift_cyl[0]/U_avg)/ U_avg;
	  c_L = (2./(par.rho_f*D_cyl))*(drag_lift_cyl[1]/U_avg)/ U_avg;
	  
	  c_D_turekstyle = (2./(par.rho_f*D_cyl))*(drag_lift_turekstyle[0]/U_avg)/ U_avg;
	  c_L_turekstyle = (2./(par.rho_f*D_cyl))*(drag_lift_turekstyle[1]/U_avg)/ U_avg;
	  
   }
   else //: if the regular FSI BM test is being performed
   {
   //We find the solid cell in which point A resides and then 
   //associate a quadrature point with point A. Note a quadrature point can only
   //lie in a unit cell. 
   const pair<typename DoFHandler<dim, dim>::active_cell_iterator, Point<dim> > cell_and_point
   = GridTools::find_active_cell_around_point (StaticMappingQ1<dim, dim>::mapping,
											   dh_s,
											   point_A);
   
   typename DoFHandler<dim, dim>::active_cell_iterator cell_having_point_A = cell_and_point.first;
   Point<dim> unit_cell_point_A = GeometryInfo<dim>::project_to_unit_cell(cell_and_point.second);
   
   
   Quadrature<dim> quad_point_A (unit_cell_point_A);
   
   FEFieldFunction<dim, DoFHandler<dim>, Vector<double> > up_field (dh_f,
																	current_xi.block(0));
																	
	QIterated<dim-1> quad_face_s (QMidpoint<1>(), 5);
	
   vector <typename DoFHandler<dim>::active_cell_iterator> fluid_cells;
   vector <vector<Point<dim> > > fluid_qpoints;
   vector< vector<unsigned int > > fluid_maps;
   
   FEFaceValues <dim, dim> fe_s_face_v (fe_s, 
										quad_face_s, 
										update_values |
										update_gradients |
										update_normal_vectors |
										update_quadrature_points |
										update_JxW_values);
   
   FEValues <dim, dim> fe_s_v_mapped_point_A (*mapping, 
												fe_s, 
												quad_point_A,
												update_quadrature_points);
   
   FEFaceValues <dim, dim> fe_s_face_v_mapped (*mapping,
											   fe_s,
											   quad_face_s,
											   update_quadrature_points);
	  
	  FEValues <dim, dim> fe_s_v (fe_s,
								  quad_s,
								  update_values |
								  update_JxW_values);
	  
	  FEValues <dim, dim> fe_s_v_mapped (*mapping,
										 fe_s,
										 quad_s,
										 update_quadrature_points);


   //Calculation of the drag-lift over the flag based on Cauchy stress in fluid...
   unsigned int n_dofs_s = fe_s.dofs_per_cell;
   vector <unsigned int> dofs_s (n_dofs_s);
   unsigned int n_qps = quad_face_s.size();
   
   vector < vector< Tensor <1, dim> > > sol_grad_s (n_qps, vector< Tensor <1,dim> > (dim));
   
   vector< Tensor <1, dim> > flag_sum_traction_s_e (n_qps);
   vector< Tensor <1, dim> > flag_sum_traction_s_inc(n_qps);
   Tensor <1, dim> flag_sum_traction_s_inc2;
   Tensor <1, dim> flag_sum_traction_s_v;
   Tensor <1, dim> flag_sum_traction_f;
   Tensor <1, dim> flag_sum_traction_s;

   
   vector < Vector <double> > projected_p (n_qps, Vector <double> (dim));

   vector<Tensor<2,dim,double> > Pe(n_qps, Tensor<2,dim,double>());
   vector< vector<Tensor<2,dim,double> > > DPeFT_dxi;
   
   vector<Tensor<2,dim,double> > F(n_qps, Tensor<2,dim,double>());   
   vector<Tensor<2,dim,double> > inv_FT(n_qps, Tensor<2,dim,double>());
   Vector<double> det_F(n_qps);
	  
	  set_to_zero(tmp_vec_n_dofs_W);
   typename DoFHandler<dim,dim>::active_cell_iterator
   cell_s = dh_s.begin_active(), endc_s = dh_s.end();
	 
	  //The following segment of code is used to find the L2-projection of the 
	  //field p on to the solid domain. 
	  for (; cell_s != endc_s; ++cell_s)
	  {
		 cell_s->get_dof_indices (dofs_s);
		 
		 fe_s_v_mapped.reinit(cell_s);
		 fe_s_v.reinit(cell_s);
		 
		 up_field.compute_point_locations (fe_s_v_mapped.get_quadrature_points(),
										   fluid_cells, 
										   fluid_qpoints,
										   fluid_maps);
		 
		 for(unsigned int c=0; c<fluid_cells.size(); ++c)
		 {
			Quadrature<dim> local_quad (fluid_qpoints[c]);
			FEValues<dim> local_fe_f_v (fe_f,
										local_quad,
										update_values |
										update_gradients);
			
			local_fe_f_v.reinit(fluid_cells[c]);
			
			set_to_zero(sol_f);
			sol_f.resize (local_quad.size(), Vector<double>(dim+1));
			local_fe_f_v.get_function_values (current_xi.block(0),
											  sol_f);
			
			for(unsigned int q=0; q<local_quad.size(); ++q)
			{
			   unsigned int &qs = fluid_maps[c][q];
			   
			   for(unsigned int i=0; i < n_dofs_s; ++i)
				  tmp_vec_n_dofs_W (dofs_s[i]) +=
				  fe_s_v.shape_value(i, qs)
				  * sol_f[q](dim)
				  * fe_s_v.JxW(qs);
				  
			}	
					
		 }
		 
	  }
	  
	  M_gamma3_inv.solve(tmp_vec_n_dofs_W);
	  
//The actual calculations of the drag and the lift on the flag are done in the 
//following section of the code:	  
   for (cell_s = dh_s.begin_active(); cell_s != endc_s; ++cell_s)
   {
	  for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
	  {
		 if (cell_s->face(face)->at_boundary() && (cell_s->face(face)->boundary_indicator() !=81))
		 {		   
			cell_s->get_dof_indices(dofs_s);
			
			fe_s_face_v.reinit (cell_s, face);
			
			fe_s_face_v.get_function_values (tmp_vec_n_dofs_W,
												projected_p);
			
			fe_s_face_v.get_function_gradients (current_xi.block(1),
												sol_grad_s);
												
			//Contribution due to the elastic stress of the solid--------
			get_Pe_F_and_DPeFT_dxi_values (fe_s_face_v,
										   dofs_s,
										   current_xi.block(1),
										   false,
										   Pe,
										   F,
										   DPeFT_dxi);
										   
			for(unsigned int qs = 0; qs < n_qps; ++qs)
			{
			   det_F(qs) = determinant(F[qs]);
			   inv_FT[qs] = transpose(invert(F[qs]));
			   
				flag_sum_traction_s_e[qs] = //Pe*N
									(Pe[qs]
									 *fe_s_face_v.normal_vector(qs))
									 *fe_s_face_v.JxW(qs);
			   
			   //Using the projected value of p
			   if(!par.solid_is_compressible)
				  flag_sum_traction_s_inc[qs] = //(- p_projected I)*(J F^(-T) N dA) 
				  - projected_p[qs](0) 
				  * det_F(qs)
				  * (inv_FT[qs]
					 * fe_s_face_v.normal_vector(qs))
				  * fe_s_face_v.JxW(qs); 
			}
						
			fe_s_face_v_mapped.reinit(cell_s, face);
			   
			up_field.compute_point_locations (fe_s_face_v_mapped.get_quadrature_points(),
												 fluid_cells, 
												 fluid_qpoints,
												 fluid_maps);
			
			for(unsigned int c=0; c<fluid_cells.size(); ++c)
			{
			   Quadrature<dim> local_quad (fluid_qpoints[c]);
			   FEValues<dim> local_fe_f_v (fe_f,
										   local_quad,
										   update_values |
										   update_gradients);
				  
			   local_fe_f_v.reinit(fluid_cells[c]);
				  
			   set_to_zero(sol_f);
			   sol_f.resize (local_quad.size(), Vector<double>(dim+1));
			   local_fe_f_v.get_function_values (current_xi.block(0),
												 sol_f);
				  
				  
			   set_to_zero(sol_grad_f);
			   sol_grad_f.resize (local_quad.size(), 
								  vector< Tensor<1,dim> >(dim+1)
									 );
			   local_fe_f_v.get_function_gradients (current_xi.block(0),
													   sol_grad_f);
				
			 
			   for(unsigned int q=0; q<local_quad.size(); ++q)
			   {
				  unsigned int &qs = fluid_maps[c][q];
					
				flag_sum_traction_f = 0.0;
				flag_sum_traction_s_v = 0.0;
				flag_sum_traction_s_inc2 = 0.0;
				
				  for(unsigned int i = 0; i < dim; ++i)
				  {
					 for(unsigned int j = 0; j < dim; ++j)
					 {
						flag_sum_traction_f[i] += //(T_f - p I)*(J F^(-T) N dA) 
						(par.eta_f
						   *(sol_grad_f[q][i][j]
						   + sol_grad_f[q][j][i])
						   - (i == j ? sol_f[q](dim): 0.0)
						   )
						*det_F(qs)
						* (inv_FT[qs]
						   *fe_s_face_v.normal_vector(qs))[j]
						*fe_s_face_v.JxW(qs);
						
						flag_sum_traction_s_v[i] += //T_s*(J F^(-T) N dA) 
						par.eta_s
						   *(sol_grad_f[q][i][j]
						   + sol_grad_f[q][j][i])
						*det_F(qs)
						* (inv_FT[qs]
						   *fe_s_face_v.normal_vector(qs))[j]
						*fe_s_face_v.JxW(qs);
						}
						
					/*	if(!par.solid_is_compressible)
						flag_sum_traction_s_inc2[i] = //(- p I)*(J F^(-T) N dA) 
						   - sol_f[q](dim)
						   * det_F(qs)
						   * (inv_FT[qs]
						   * fe_s_face_v.normal_vector(qs))[i]
						   * fe_s_face_v.JxW(qs); 
					 */					 
					 
					}
				
				flag_sum_traction_s = flag_sum_traction_s_v 
				+ flag_sum_traction_s_inc[qs]
				+ flag_sum_traction_s_e[qs];
				
				drag_lift_flag_f += flag_sum_traction_f;
				drag_lift_flag_s += flag_sum_traction_s;
				
				drag_lift_flag_avg += 0.5*( flag_sum_traction_f + flag_sum_traction_s);
				
			   }
				  
			}
			
			//--------------------------------------------------------------//
			//-------        CALCULATION OF PRESSURE AT POINT A ------------//
			//--------------------------------------------------------------//			
			//Calculate the pressure at point A & the viscous stress (if any)
			if (cell_s == cell_having_point_A)
			{
			   fe_s_v_mapped_point_A.reinit(cell_s);
			   
			   up_field.compute_point_locations (fe_s_v_mapped_point_A.get_quadrature_points(),
												 fluid_cells, 
												 fluid_qpoints,
												 fluid_maps);
			   
			   Assert(fluid_cells.size() == 1, ExcMessage("Mapped point A found in multiple fluid cells!"));
			   Quadrature<dim> quad_bg (fluid_qpoints[0]);
			   
			   FEValues<dim> fe_v_bg (fe_f, quad_bg,
									  update_values | update_gradients);
			   fe_v_bg.reinit(fluid_cells[0]);
			   
			   //Localize the solution for the obtained fluid cell...but before that, resize the vectors
			   sol_f.resize(1, Vector <double> (dim+1));
			   
			   fe_v_bg.get_function_values (current_xi.block(0), sol_f);
			   pressure_A = sol_f[0](dim);
			   
			}
			
		 }//if face of solid cell is at the boundary
	  }//loop over faces of solid cells
   }//End loop over solid cells
}//For FSI test only 
   
   fsi_bm_out_file.unsetf(ios_base::floatfield);
   fsi_bm_out_file 
   << current_time	
   << "\t"
   << scientific
   << (par.cfd_test ? c_D_turekstyle : disp_A(0)) 
   << "\t" 
   << (par.cfd_test ? c_L_turekstyle : disp_A(1))	
   << "\t"
   << sol_B(dim)	//: Pressure at stagnation pt
   << "\t"	
   << sol_B(dim) - pressure_A //: Pressure drop  
   << "\t"
   << (par.cfd_test ? c_D : (drag_lift_cyl[0] + drag_lift_flag_f[0])) //: Drag coefficient/Total drag based on fluid stress
   << "\t" 
   << (par.cfd_test ? c_L : (drag_lift_cyl[1] + drag_lift_flag_f[1])) //: Lift coefficient/Total lift based on fluid stress
   << "\t"
   << drag_lift_cyl[0] + drag_lift_flag_s[0] //: Drag based on integrating traction in the solid 
   << "\t" 
   << drag_lift_cyl[1] + drag_lift_flag_s[1] //: Lift based on integrating traction in the solid
   << "\t"
   << drag_lift_cyl[0] + drag_lift_flag_avg[0] //: Drag based on avg. of the traction at the interface
   << "\t" 
   << drag_lift_cyl[1] + drag_lift_flag_avg[1] //: Lift based on avg. of the traction at the interface
   <<endl;
}


template <int dim>
void ImmersedFEMGeneralized<dim>::fsi_bm_postprocess2()
{
   //: Some geometric features of the benchmark test(s)
   const Point<dim> center_cyl (0.2, 0.2); //: Center of the cylinder
   const double D_cyl = 0.1; //: Diameter of the cylinder
   const double H = 0.41; //: Height of the channel
   const double l_flag = 0.35; //: Length of the flag
   
   //Compute the the displacement of the pt. A (i.e. the midpt of the end of the tail)
   Point<dim> point_A (center_cyl[0]+0.5*D_cyl+l_flag, center_cyl[1]);
   Point<dim> point_B (center_cyl[0]-0.5*D_cyl, center_cyl[1]);
   Vector<double> disp_A (dim);
   Vector<double> sol_B (dim+1);
   double pressure_A = 0.0;
   double U_avg = 0.0;
   double c_D = 0.0;
   double c_L = 0.;
   
   Point<dim> drag_lift;
   
   if(par.cfd_test)
   {
	  //: Point A now corresponds to midpoint of the aft edge of the cylinder
	  point_A[0] -= l_flag; 
	  
	  //: Extract the value of pressure at point A
	  VectorTools::point_value(dh_f, 
							   current_xi.block(0), 
							   point_A, 
							   sol_B);
	  pressure_A = sol_B(dim);
	  
	  //: Evaluate the average value of the inflow velocity for t=4s for CFDBM1-3
	  par.u_g.set_time(4.0);
	  U_avg = par.u_g.value(Point<dim>(0., 0.5*H))/1.5;
	  par.u_g.set_time(current_time);
	  
   }
   else
	  VectorTools::point_value(dh_s, 
							   current_xi.block(1), 
							   point_A, 
							   disp_A);
   
   VectorTools::point_value(dh_f, 
							current_xi.block(0), 
							point_B, 
							sol_B);
   
   //-------------------- Loop over FLUID CELLS-----------------//
   //Calculating the drag and lift values corresponding to the free surface of the cylinder
   
   typename DoFHandler<dim,dim>::active_cell_iterator
   cell = dh_f.begin_active(), endc = dh_f.end();
   
   QGauss <dim-1> quad_face (par.degree+2);
   unsigned int n_qpf_face = quad_face.size();
   
   unsigned int n_qpf = quad_f.size();
   
   FEFaceValues<dim> fe_f_face_v (fe_f, 
								  quad_face,
								  update_values | 
								  update_gradients |
								  update_normal_vectors |
								  update_quadrature_points |
								  update_JxW_values);
   
   FEValues<dim> fe_f_v (fe_f,
						 quad_f,
						 update_values |
						 update_gradients |
						 update_JxW_values |
						 update_quadrature_points);
   
   unsigned int n_dofs = fe_f.dofs_per_cell;
   vector <unsigned int> dofs_f (n_dofs);
   
   vector < Vector <double> > sol_f_face (n_qpf_face, Vector <double> (dim+1));
   vector < vector< Tensor <1, dim> > > sol_grad_f_face (n_qpf_face,
														 vector< Tensor <1,dim> > (dim+1));
   
   vector < Vector <double> > sol_t_f (n_qpf, Vector <double> (dim+1));
   vector < Vector <double> > sol_f (n_qpf, Vector <double> (dim+1));
   vector < vector< Tensor <1, dim> > > sol_grad_f (n_qpf, vector< Tensor <1,dim> > (dim+1));
   vector < Vector<double> > local_force(n_qpf, Vector<double>(dim+1));

   
   unsigned int c_no = 0;
   for(; cell !=endc; ++cell, ++c_no)
   { 
	  cell->get_dof_indices (dofs_f);

	  fe_f_v.reinit (cell);
	  
	  fe_f_v.get_function_values (current_xit.block(0), sol_t_f);
	  fe_f_v.get_function_values (current_xi.block(0), sol_f);
	  fe_f_v.get_function_gradients (current_xi.block(0), sol_grad_f);
	  
	  par.force.vector_value_list(fe_f_v.get_quadrature_points(), local_force);

	  //: Volume integration over the entire control volume
	  for(unsigned int q = 0; q < n_qpf; ++q)//loop over quadrature pts
	  {
		 for(unsigned int i = 0; i < dim; ++i)//loop over dim
		 {
			drag_lift[i] += //- rho_f (u' + b)
			- par.rho_f
			*( sol_t_f[q](i)
			  + local_force[q](i)
			  )
			* fe_f_v.JxW(q);
			
			for(unsigned int j = 0; j < dim; ++j)//loop over dim
			  drag_lift[i] += //- rho_f grad u [u]
			   - par.rho_f
			   * sol_grad_f[q][i][j]
			   * sol_f[q](i)
			   * fe_f_v.JxW(q);
		 }
	  }
	  
	  //: Surface integration over all the boundaries of the control volume 
	  // EXCEPT those comprising the cylinder + the flag
	  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
	  {
		 if (cell->face(face)->at_boundary() 
			 && (cell->face(face)->boundary_indicator() < 80)
			 )
		 {			
			fe_f_face_v.reinit (cell, face);
			
			fe_f_face_v.get_function_values (current_xi.block(0), sol_f_face);
			fe_f_face_v.get_function_gradients (current_xi.block(0), sol_grad_f_face);
			
			for(unsigned int q = 0; q < n_qpf_face; ++q)//loop over quadrature pts
			{
			   for(unsigned int i = 0; i < dim; ++i)//loop over dim
				  for(unsigned int j = 0; j < dim; ++j)//loop over dim
					 drag_lift[i] += //(T_f - p I)*(n da)
					 (par.eta_f 
					  *(sol_grad_f_face[q][i][j]
						+ sol_grad_f_face[q][j][i])
					  - (i == j ? sol_f_face[q](dim) : 0.0)
					  )
					 *(fe_f_face_v.normal_vector(q)(j))
					 *fe_f_face_v.JxW(q);
			}//loop over q
		 }//if cond.
	  }//loop over faces
	  
   }//loop over cells
   
   ///-------------------- Loop over FLUID CELLS (end)-----------------//
   if(par.cfd_test && (abs(U_avg)>1e-8))
   {
	  c_D = (2./(par.rho_f*D_cyl))*(drag_lift[0]/U_avg)/ U_avg;
	  c_L = (2./(par.rho_f*D_cyl))*(drag_lift[1]/U_avg)/ U_avg;
   }
   else //: if the regular FSI BM test is being performed
   {
	  //We find the solid cell in which point A resides and then 
	  //associate a quadrature point with point A. Note a quadrature point can only
	  //lie in a unit cell. 
	  const pair<typename DoFHandler<dim, dim>::active_cell_iterator, Point<dim> > cell_and_point
	  = GridTools::find_active_cell_around_point (StaticMappingQ1<dim, dim>::mapping,
												  dh_s,
												  point_A);
	  
	  typename DoFHandler<dim, dim>::active_cell_iterator cell_having_point_A = cell_and_point.first;
	  Point<dim> unit_cell_point_A = GeometryInfo<dim>::project_to_unit_cell(cell_and_point.second);
	  
	  
	  Quadrature<dim> quad_point_A (unit_cell_point_A);
	  
	  const unsigned int n_qps = quad_s.size();
	  
	  FEFieldFunction<dim, DoFHandler<dim>, Vector<double> > up_field (dh_f,
																	   current_xi.block(0));
	  
	  
	  vector <typename DoFHandler<dim>::active_cell_iterator> fluid_cells;
	  vector <vector<Point<dim> > > fluid_qpoints;
	  vector< vector<unsigned int > > fluid_maps;
	  
	  FEValues <dim, dim> fe_s_v_mapped_point_A (*mapping, 
												 fe_s, 
												 quad_point_A,
												 update_quadrature_points);
	  
	  FEValues <dim, dim> fe_s_v (fe_s,
								  quad_s,
								  update_values |
								  update_gradients |
								  update_JxW_values);
	  
	  FEValues <dim, dim> fe_s_v_mapped (*mapping,
										 fe_s,
										 quad_s,
										 update_quadrature_points);
	  
	  vector< vector< Tensor<1,dim> > > sol_grad_s (n_qps, 
													vector< Tensor<1,dim> >(dim));
	  Tensor<2,dim,double> F;
	  vector<double> local_J(n_qps);
	  
      local_force.resize (n_qps, Vector<double>(dim+1));
      	  
	  unsigned int n_dofs_s = fe_s.dofs_per_cell;
	  vector <unsigned int> dofs_s (n_dofs_s);
	  	  	  
	  typename DoFHandler<dim,dim>::active_cell_iterator
	  cell_s = dh_s.begin_active(), endc_s = dh_s.end();
	  
	  for (; cell_s != endc_s; ++cell_s)
	  {
		 
		 fe_s_v_mapped.reinit (cell_s);
		 fe_s_v.reinit(cell_s);
		 
		 fe_s_v.get_function_gradients (current_xi.block(1),
											 sol_grad_s);
		 
		 
		 set_to_zero(local_force);
		 par.force.vector_value_list (fe_s_v_mapped.get_quadrature_points(),
									  local_force);
		 
		 for( unsigned int qs = 0; qs < n_qps; ++qs )
		 {
			for(unsigned int i=0; i <dim; ++i)
			{
			   F[i] = sol_grad_s[qs][i];
			   F[i][i] += 1.0;
			}
		 
			local_J[qs] = determinant(F);
		 }
		 
		 up_field.compute_point_locations (fe_s_v_mapped.get_quadrature_points(),
										   fluid_cells, 
										   fluid_qpoints,
										   fluid_maps);
		 
		 for(unsigned int c=0; c<fluid_cells.size(); ++c)
		 {
			Quadrature<dim> local_quad (fluid_qpoints[c]);
			FEValues<dim> local_fe_f_v (fe_f,
										local_quad,
										update_values |
										update_gradients);
			
			local_fe_f_v.reinit(fluid_cells[c]);
			
			set_to_zero(sol_t_f);
			sol_t_f.resize (local_quad.size(), Vector<double>(dim+1));
			local_fe_f_v.get_function_values (current_xit.block(0),
											  sol_t_f);
			set_to_zero(sol_f);
			sol_f.resize (local_quad.size(), Vector<double>(dim+1));
			local_fe_f_v.get_function_values (current_xi.block(0),
											  sol_f);
			
			
			set_to_zero(sol_grad_f);
			sol_grad_f.resize (local_quad.size(),
							  vector< Tensor<1,dim> > (dim+1)
							   );
			local_fe_f_v.get_function_gradients (current_xi.block(0), sol_grad_f);
			
						
			for(unsigned int q=0; q<local_quad.size(); ++q)
			{
			   unsigned int &qs = fluid_maps[c][q];
			   
			   for(unsigned int i = 0; i < dim; ++i)//loop over dim
			   {
				  drag_lift[i] +=
				  local_J[qs]
				  * par.rho_f
				  * (sol_t_f[q](i) 
					 - local_force[qs](i))
				  * fe_s_v.JxW(qs);
				  
				  for(unsigned int j = 0; j < dim; ++j)//loop over dim
					 drag_lift[i] +=
					 local_J[qs]
					 * par.rho_f
					 * sol_grad_f[q][i][j]
					 * sol_f[q](j)
					 * fe_s_v.JxW(qs);
					 
			   }
			   			   
			}	
			
		 }
		 
		 //--------------------------------------------------------------//
		 //-------        CALCULATION OF PRESSURE AT POINT A ------------//
		 //--------------------------------------------------------------//			
		 //Calculate the pressure at point A & the viscous stress (if any)
		 if (cell_s == cell_having_point_A)
		 {
			fe_s_v_mapped_point_A.reinit(cell_s);
			
			up_field.compute_point_locations (fe_s_v_mapped_point_A.get_quadrature_points(),
											  fluid_cells, 
											  fluid_qpoints,
											  fluid_maps);
			
			Assert(fluid_cells.size() == 1, ExcMessage("Mapped point A found in multiple fluid cells!"));
			Quadrature<dim> quad_bg (fluid_qpoints[0]);
			
			FEValues<dim> fe_v_bg (fe_f, quad_bg,
								   update_values | update_gradients);
			fe_v_bg.reinit(fluid_cells[0]);
			
			//Localize the solution for the obtained fluid cell...but before that, resize the vectors
			sol_f.resize(1, Vector <double> (dim+1));
			
			fe_v_bg.get_function_values (current_xi.block(0), sol_f);
			pressure_A = sol_f[0](dim);
			
		 }
		 
	  }//End loop over solid cells

   }//For FSI test only 
   fsi_bm_out_file.unsetf(ios_base::floatfield);
   fsi_bm_out_file 
   << current_time	
   << "\t"
   << scientific
   << (par.cfd_test ? 0.0 : disp_A(0)) 
   << "\t" 
   << (par.cfd_test ? 0.0 : disp_A(1))	
   << "\t"
   << sol_B(dim)	//: Pressure at stagnation pt
   << "\t"	
   << sol_B(dim) - pressure_A //: Pressure drop  
   << "\t"
   << (par.cfd_test ? c_D : drag_lift[0]) //: Drag coefficient/Total drag based on fluid stress
   << "\t" 
   << (par.cfd_test ? c_L : drag_lift[1]) //: Lift coefficient/Total lift based on fluid stress
   << endl;
}

template <int dim>
template <class Archive>
void ImmersedFEMGeneralized<dim>::serialize(Archive &ar, const unsigned int version)
{
   ar &current_time;
   ar &dt;
   ar &time_step;
}


template <int dim>
void ImmersedFEMGeneralized<dim>::restart_computations()
{
   // Load the details concerning the temporal integration.
   //Currently we are reading in: current_time, timestep and dt 
   ifstream ifs((par.output_name
			  + par.file_info_for_restart
			  + "resume.txt").c_str());
   boost::archive::text_iarchive ia_time (ifs);
   ia_time >> (*this);
   ifs.close();
   
   previous_time = current_time;

   ifstream fname_xi((par.output_name
					  + par.file_info_for_restart
					  + "xi.bin").c_str());
   tmp_vec_n_total_dofs.block_read (fname_xi);
   previous_xi = tmp_vec_n_total_dofs;
   fname_xi.close();
   
   
//ifstream fname_prev_xi((par.output_name
//						   + par.file_info_for_restart
//						   + "prev_xi.bin").c_str());
//   tmp_vec_n_total_dofs.block_read (fname_prev_xi);
//   previous_xi = tmp_vec_n_total_dofs;
//   fname_prev_xi.close();
   
   
   // Load the vector storing the current solution.
   //tmp_vec_n_total_dofs.load((par.file_info_for_restart + "_xi.txt").c_str());
   //current_xi = tmp_vec_n_total_dofs;
   
   // Load the vector storing the previous solution.
   //tmp_vec_n_total_dofs.load((par.file_info_for_restart + "_previous_xi.txt").c_str());
   //previous_xi = tmp_vec_n_total_dofs;

}

template <int dim>
void ImmersedFEMGeneralized<dim>::save_for_restart()
{
   // Assumption is that if the initial time is 0 then this the first time that this
   // computation 
   if (par.this_is_a_restart)
   {
	  // "if we have previously written a snapshot, then keep the last
	  // snapshot in case this one fails to save" --- ASPECT checkpoint_restart.cc
	  move_file (par.output_name
				 + par.file_info_for_restart
				 + "resume.txt",
				par.output_name
				 + par.file_info_for_restart
				 + "resume.txt.old");
	  move_file (par.output_name + par.file_info_for_restart + "xi.bin",
					par.output_name + par.file_info_for_restart + "xi.bin.old");
//	  move_file (par.output_name + par.file_info_for_restart + "prev_xi.bin",
//					par.output_name + par.file_info_for_restart + "prev_xi.bin.old");
   }
   
   // Load the details concerning the temporal integration.
   std::ofstream ofs ((par.output_name
					   + par.file_info_for_restart
					   + "resume.txt").c_str());
   boost::archive::text_oarchive oa_time (ofs);
   oa_time << (*this);
   ofs.close();
   
   
   // Load the vector storing the current solution.
 //  tmp_vec_n_total_dofs = current_xi;
 //  tmp_vec_n_total_dofs.save ((par.file_info_for_restart + "_xi.txt").c_str());
   
   // Load the vector storing the previous solution.
 //  tmp_vec_n_total_dofs = previous_xi;
 //  tmp_vec_n_total_dofs.save ((par.file_info_for_restart + "_previous_xi.txt").c_str());
   
   ofstream fname_xi((par.output_name + par.file_info_for_restart + "xi.bin").c_str());
   tmp_vec_n_total_dofs = current_xi;
   tmp_vec_n_total_dofs.block_write(fname_xi);
   fname_xi.close();

   
//   ofstream fname_prev_xi((par.output_name 
//						   + par.file_info_for_restart
//						   + "prev_xi.bin").c_str());
//   
//   tmp_vec_n_total_dofs = previous_xi;
//   tmp_vec_n_total_dofs.block_write(fname_prev_xi);
//   fname_prev_xi.close();
   
  // ifstream fname("test_vec_out.bin");
   //test_bvec.block_read(fname);

}   

// Simple initialization to zero function templated on a generic type.

template <int dim>
template <class Type>
void ImmersedFEMGeneralized<dim>::set_to_zero (Type &v) const
{
  v = 0;
}

// Simple initialization to zero function templated on a vector of
// generic type.
template <int dim>
template <class Type>
void ImmersedFEMGeneralized<dim>::set_to_zero (vector<Type> &v) const
{
  for(unsigned int i = 0; i < v.size(); ++i) set_to_zero(v[i]);
}

// Simple initialization to zero function templated on a table of
// generic type.
template <int dim>
template <class Type>
void ImmersedFEMGeneralized<dim>::set_to_zero (Table<2, Type> &v) const
{
  for(unsigned int i=0; i<v.size()[0]; ++i)
    for(unsigned int j=0; j<v.size()[1]; ++j) set_to_zero(v(i,j));
}

// Determination of the norm of a vector.
template <int dim>
double ImmersedFEMGeneralized<dim>::norm(const vector<double> &v)
{
  double norm = 0;
  for( unsigned int i = 0; i < v.size(); ++i) norm += v[i]*v[i];
  return norm = sqrt(norm);
}

template class ImmersedFEMGeneralized<2>;
template class ImmersedFEMGeneralized<3>;

