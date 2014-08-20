
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
  std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_2d_solid.msh");
  gi.read_msh(infile);
  
  Triangulation<3,3> tria;
  GridGenerator::extrude_triangulation(tria_2d, 2, 1.37, tria);
  
  Point<3> shift_vector(0,0,1.365); // First shift, for larger slice and solid
  GridTools::shift(shift_vector, tria);
  
  GridOut go;
  GridOutFlags::Msh flags(true, true);
  go.set_flags(flags);
  go.write_msh(tria, deallog.get_file_stream());
  std::ofstream ofile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_solid.msh");
  go.write_msh(tria, ofile);  
}
