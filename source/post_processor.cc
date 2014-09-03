#include "post_processor.h"



// Constructor:
//    Initializes the FEM system of the control volume;
//    Initializes the FEM system of the immersed domain;
//    Initializes, corresponding dof handlers, and the quadrature rule;
//    It runs the <code>create_triangulation_and_dofs</code> function.

template <int dim>
PostProcessor<dim>::PostProcessor (IFEMParametersGeneralized<dim> &par)
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
  if (par.degree <= 1)
    cout
        << " WARNING: The chosen pair of finite element spaces is not  stable."
        << endl
        << " The obtained results will be nonsense."
        << endl;

  if ( Utilities::match_at_string_start(par.fe_p_name, string("FE_DGP")))
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
  
  std::ifstream in_test((par.output_name+"_post_global.gpl").c_str());
  ios::openmode mode;
  if(in_test) {
    mode = ios::app;
    std::string line;
    while(in_test >> std::ws && std::getline(in_test, line));
    // now we have last line
    sscanf(line.c_str(), "%lf", &current_time);
    previous_time = current_time;
    time_step = static_cast<unsigned int>(current_time/par.dt);
    dt = par.dt;
    in_test.close();
  } else {
    mode = ios::out;
    previous_time = 0.0;
    current_time = 0.0;
    time_step = 0;
    dt = par.dt;
  }
  
  global_info_file.open((par.output_name+"_post_global.gpl").c_str(), mode);
  
  fsi_bm_out_file.open((par.output_name+"_post_fsi_bm.out").c_str(), mode);

  create_triangulation_and_dofs ();
}

// Distructor: deletion of pointers created with <code>new</code> and
// closing of the record keeping file.

template <int dim>
PostProcessor<dim>::~PostProcessor ()
{
  delete mapping;
  global_info_file.close();

  fsi_bm_out_file.close();
}

// Determination of the current value of time dependent boundary
// values.




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
PostProcessor<dim>::create_triangulation_and_dofs ()
{
  GridIn<dim> grid_in_f;
  grid_in_f.attach_triangulation (tria_f);
  
  {
    ifstream file (par.fluid_mesh.c_str());
    Assert (file, ExcFileNotOpen (par.fluid_mesh.c_str()));


    // A grid in ucd format is expected.
    grid_in_f.read_msh (file);
  }
  {
    GridIn<dim, dim> grid_in_s;
    grid_in_s.attach_triangulation (tria_s);
  
    ifstream file (par.solid_mesh.c_str());
    Assert (file, ExcFileNotOpen (par.solid_mesh.c_str()));
  
    // A grid in ucd format is expected.
    grid_in_s.read_msh (file);
  }

  if (par.fsi_bm && dim == 2)
    {
      Point<dim> center_circ(0.2, 0.2);
      double radius_circ = 0.05;
      static const HyperBallBoundary<dim> boundary_cyl(center_circ,radius_circ);

      tria_f.set_boundary(80, boundary_cyl);
//  tria_f.set_boundary(81, boundary_cyl);

//  tria_s.set_boundary(81, boundary_cyl);
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

  // Read in first solution
  std::ifstream fluid_binary_file( (par.output_name + "-fluid-" +
				    Utilities::int_to_string (time_step, 5) +
				    ".bin").c_str() );
  AssertThrow(fluid_binary_file, 
	      ExcMessage("Fluid input file not found."));
  
  current_xi.block(0).block_read(fluid_binary_file);
    
  std::ifstream solid_binary_file( ( par.output_name + "-solid-" +
				     Utilities::int_to_string (time_step, 5) +
				     ".bin").c_str() );
  AssertThrow(solid_binary_file, 
	      ExcMessage("Solid input file not found."));

  current_xi.block(1).block_read(solid_binary_file);
      
  
  // Initialization of the current state of the system.
  previous_xi = current_xi;
  

  mapping = new MappingQEulerian<dim, Vector<double>, dim> (par.degree,
                                                            previous_xi.block(1),
                                                            dh_s);
}


// Central management of the time stepping scheme. Read all input
// files which are available.
template <int dim>
void
PostProcessor<dim>::run ()
{
// The overall cycle over time begins here.
  for (double t = current_time + par.dt; true; t += par.dt)
    {
      // Read the current solution, check that we are good, and
      // proceed with post processing
      current_time = t;
      ++time_step;

      std::ifstream fluid_binary_file( (par.output_name + "-fluid-" +
					Utilities::int_to_string (time_step, 5) +
					".bin").c_str() );
      // exit when no file could be read
      if(!fluid_binary_file)
	return;
      current_xi.block(0).block_read(fluid_binary_file);
    
      std::ifstream solid_binary_file( ( par.output_name + "-solid-" +
					 Utilities::int_to_string (time_step, 5) +
					 ".bin").c_str() );
      // exit when no file could be read
      if(!solid_binary_file)
	return;

      current_xi.block(1).block_read(solid_binary_file);
      

      current_xit.sadd (0, 1./par.dt, current_xi, -1./par.dt, previous_xi);
      post_process(t, time_step, dt);

// After we have post_processed the solution, we update the state of the
// system and move to the next time step.
      previous_xi = current_xi;
      previous_time =t;
    }
}
// End of <code>run()</code>.


template<int dim, int spacedim>
std::vector<unsigned int> get_point_dofs(const DoFHandler<dim,spacedim> &dh, 
					 const Point<spacedim> &p,
					 const Mapping<dim,spacedim> &mapping
					 =StaticMappingQ1<dim,spacedim>::mapping,
					 const double tol=1e-10) {
  std::vector< Point<spacedim> > support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
  std::vector<unsigned int> dofs;
  const double rel_tol=std::max(tol, tol*p.norm());
  for(unsigned int i=0; i<support_points.size(); ++i)
    if(support_points[i].distance(p) < rel_tol)
      dofs.push_back(i);
  if(dofs.size())
    {
      cout << "Found " << dofs.size() << " point dofs: ";
      for(unsigned int i=0; i<dofs.size(); ++i)
	cout << dofs[i] << ", ";
      cout << endl;
    }
  return dofs;
}

template<int dim>
bool face_not_on_cylinder(const typename DoFHandler<dim>::face_iterator &face) {
  if(dim == 2) 
    return face->boundary_indicator() < 80;
  else 
    return  ( (std::abs(std::abs(face->center()[0]-5.0) - .5) > 1e-6 ) &&
	      (std::abs(std::abs(face->center()[1]-2.0) - .5) > 1e-6 ) );
}

// Writes results to the output file.
template <int dim>
void
PostProcessor<dim>::post_process(const double t, const unsigned int step, const double dt)
{
  cout << "Time " << t << ", Step " << step << ", dt = " << dt << endl;
  
  global_info_file << t << " ";
  
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
    for (; cell != endc; ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            fe_v.reinit(cell, f);
            fe_v.get_function_values(current_xi.block(0), local_vp);
            const vector<Point<dim> > &normals = fe_v.get_normal_vectors();
            for (unsigned int q=0; q<face_quad.size(); ++q)
              {
                Point<dim> vq;
                for (unsigned int d=0; d<dim; ++d) vq[d] = local_vp[q](d);
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
    for (; cell != endc; ++cell)
      {
        fe_v.reinit(cell);
        const vector<Point<dim> > &qpoints = fe_v.get_quadrature_points();
        for (unsigned int q=0; q<quad_s.size(); ++q)
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
  // Compute Drag and lift
  {
    //: Some geometric features of the benchmark test(s)
    const double H = (dim == 2 ? 0.41 : 4.1); //: Height of the channel
    const double l_flag = (dim == 2 ? 0.35 : 3); //: Length of the flag
    const double D_cyl = (dim == 2 ? 0.1 : 1);
    const Point<dim> center_cyl = (dim == 2 ? Point<dim>(0.2, 0.2) : Point<dim>(5,2,H/2));
    
    //Compute the the displacement of the pt. A (i.e. the 
    Point<dim> point_A = (dim == 2 ? 
			  Point<dim>(center_cyl[0]+0.5*D_cyl+l_flag, center_cyl[1]) : 
			  Point<dim>(8.5, 2.5, 2.735));

    Point<dim> point_B = (dim == 2 ? Point<dim>(center_cyl[0]-0.5*D_cyl, center_cyl[1]) : 
			  Point<dim>(4.5, 2, 2.05));
    Vector<double> disp_A (dim);
    Vector<double> sol_B (dim+1);
    double pressure_A = 0.0;
    double U_avg = 0.0;
    double c_D = 0.0;
    double c_L = 0.;

    Point<dim> drag_lift;
    
    static vector<unsigned int> A_dofs = get_point_dofs(dh_s, point_A);
    if(A_dofs.size() == 0)
      {
	VectorTools::point_value(dh_s,
				 current_xi.block(1),
				 point_A,
				 disp_A);
      }
    else 
      { 
	for(int d=0; d<dim; ++d)
	  disp_A(d) = current_xi.block(1)(A_dofs[d]);
      }

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
    for (; cell !=endc; ++cell, ++c_no)
      {
	cell->get_dof_indices (dofs_f);

	fe_f_v.reinit (cell);

	fe_f_v.get_function_values (current_xit.block(0), sol_t_f);
	fe_f_v.get_function_values (current_xi.block(0), sol_f);
	fe_f_v.get_function_gradients (current_xi.block(0), sol_grad_f);

	par.force.vector_value_list(fe_f_v.get_quadrature_points(), local_force);

	//: Volume integration over the entire control volume
	for (unsigned int q = 0; q < n_qpf; ++q) //loop over quadrature pts
	  {
	    for (unsigned int i = 0; i < dim; ++i) //loop over dim
	      {
		drag_lift[i] += //- rho_f (u' + b)
		  - par.rho_f
		  *( sol_t_f[q](i)
		     + local_force[q](i)
		     )
		  * fe_f_v.JxW(q);

		for (unsigned int j = 0; j < dim; ++j) //loop over dim
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
	    if (cell->face(face)->at_boundary() && 
		face_not_on_cylinder<dim>(cell->face(face)))
	      {
		fe_f_face_v.reinit (cell, face);

		fe_f_face_v.get_function_values (current_xi.block(0), sol_f_face);
		fe_f_face_v.get_function_gradients (current_xi.block(0), sol_grad_f_face);

		for (unsigned int q = 0; q < n_qpf_face; ++q) //loop over quadrature pts
		  {
		    for (unsigned int i = 0; i < dim; ++i) //loop over dim
		      for (unsigned int j = 0; j < dim; ++j) //loop over dim
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

	    for ( unsigned int qs = 0; qs < n_qps; ++qs )
	      {
		for (unsigned int i=0; i <dim; ++i)
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

	    for (unsigned int c=0; c<fluid_cells.size(); ++c)
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


		for (unsigned int q=0; q<local_quad.size(); ++q)
		  {
		    unsigned int &qs = fluid_maps[c][q];

		    for (unsigned int i = 0; i < dim; ++i) //loop over dim
		      {
			drag_lift[i] +=
			  local_J[qs]
			  * par.rho_f
			  * (sol_t_f[q](i)
			     - local_force[qs](i))
			  * fe_s_v.JxW(qs);

			for (unsigned int j = 0; j < dim; ++j) //loop over dim
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
      << disp_A(0)
      << "\t"
      << disp_A(1)
      << "\t"
      << (dim == 3 ? disp_A(2) : 0)
      << (dim == 3 ? "\t" : "")
      << sol_B(dim)    //: Pressure at stagnation pt
      << "\t"
      << sol_B(dim) - pressure_A //: Pressure drop
      << "\t"
      << drag_lift[0] //: Drag coefficient/Total drag based on fluid stress
      << "\t"
      << drag_lift[1] //: Lift coefficient/Total lift based on fluid stress
      << endl;
 }

}


// Determining the inverse transpose of the deformation gradient at all the
// quadrature points of a solid cell.

template <int dim>
void
PostProcessor<dim>::get_inverse_transpose
(const vector < Tensor <2, dim> > &F,
 vector < Tensor <2, dim> > &local_invFT) const
{
  for (unsigned int q=0; q< F.size(); ++q)
    local_invFT[q] = transpose(invert(F[q]));
}



// Simple initialization to zero function templated on a generic type.

template <int dim>
template <class Type>
void PostProcessor<dim>::set_to_zero (Type &v) const
{
  v = 0;
}

// Simple initialization to zero function templated on a vector of
// generic type.
template <int dim>
template <class Type>
void PostProcessor<dim>::set_to_zero (vector<Type> &v) const
{
  for (unsigned int i = 0; i < v.size(); ++i) set_to_zero(v[i]);
}

// Simple initialization to zero function templated on a table of
// generic type.
template <int dim>
template <class Type>
void PostProcessor<dim>::set_to_zero (Table<2, Type> &v) const
{
  for (unsigned int i=0; i<v.size()[0]; ++i)
    for (unsigned int j=0; j<v.size()[1]; ++j) set_to_zero(v(i,j));
}

// Determination of the norm of a vector.
template <int dim>
double PostProcessor<dim>::norm(const vector<double> &v)
{
  double norm = 0;
  for ( unsigned int i = 0; i < v.size(); ++i) norm += v[i]*v[i];
  return norm = sqrt(norm);
}

template class PostProcessor<2>;
template class PostProcessor<3>;

