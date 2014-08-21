
#include "../tests.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>


int
main()
{
  initlog();

  Triangulation<3,3> tria;
  GridIn<3,3> gi;
  gi.attach_triangulation(tria);
  std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_solid.msh");
  gi.read_msh(infile);
  
  Triangulation<3,3>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
  
  for(; cell != endc; ++cell) {
    for(unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
      if(cell->face(f)->at_boundary()) {
	// if(std::abs(cell->face(f)->center()[0])<1e-10) {// x=0 
	//   cell->face(f)->set_boundary_indicator(1); // 1 = inflow
	//   deallog << "Inflow: " << cell->face(f) 
	// 	  << ", face center: " << cell->face(f)->center() 
	// 	  << std::endl;
	// }
	// else
	  if(std::abs(cell->face(f)->center()[0] - 5.5)>1e-9) {
	  cell->face(f)->set_boundary_indicator(1); // 0 = fixed bc, 1 = neumann
	  deallog << "Outflow: " << cell->face(f) 
		  << ", face center: " << cell->face(f)->center() 
		  << std::endl;
	}
      }
  }

  GridOut go;
  GridOutFlags::Msh flags(true, true);
  go.set_flags(flags);
  go.write_msh(tria, deallog.get_file_stream());
  std::ofstream ofile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_solid_id.msh");
  go.write_msh(tria, ofile);  
}
