
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

  Triangulation<2,2> tria_2d;
  
  GridIn<2,2> gi;
  gi.attach_triangulation(tria_2d);
  std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_2d_isotropic.msh");  
  gi.read_msh(infile);

  Triangulation<3,3> smaller_slice, larger_slice;
  GridGenerator::extrude_triangulation(tria_2d, 2, 1.365, smaller_slice);
  GridGenerator::extrude_triangulation(tria_2d, 2, 1.37, larger_slice);

  Point<3> shift_vector(0,0,1.365); // First shift, for larger slice and solid
  Point<3> shift_vector2(0,0,1.365+1.37); // Second shift

  GridTools::shift(shift_vector, larger_slice);
  
  Triangulation<3,3> smaller_and_larger_slice;
  GridGenerator::merge_triangulations(smaller_slice, larger_slice, 
				      smaller_and_larger_slice);
  
  // Reuse first grid
  GridTools::shift(shift_vector2, smaller_slice);

  Triangulation<3,3> tria;
  GridGenerator::merge_triangulations(smaller_and_larger_slice, 
				      smaller_slice, 
				      tria);
  GridOut go;
  GridOutFlags::Msh flags(true, true);
  go.set_flags(flags);
  go.write_msh(tria, deallog.get_file_stream());
  std::ofstream ofile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_isotropic.msh");
  go.write_msh(tria, ofile);  
}
