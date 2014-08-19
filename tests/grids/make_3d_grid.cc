
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

  Triangulation<2,2> tria;
  GridIn<2,2> gi;
  gi.attach_triangulation(tria);
  std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_2d.msh");  

  gi.read_msh(infile);
  Triangulation<3,3> tria3, tria4;
  GridGenerator::extrude_triangulation(tria, 2, 1.365, tria3);
  GridGenerator::extrude_triangulation(tria, 2, 1.37, tria4);

  
  Point<3> shift_vector(0,0,1.365); // First shift
  Point<3> shift_vector2(0,0,1.365+1.37); // Second shift

  GridTools::shift(shift_vector, tria4);
  Triangulation<3,3> tria_3_4;
  GridGenerator::merge_triangulations(tria3, tria4, tria_3_4);
  
  GridTools::shift(shift_vector2, tria3);
  Triangulation<3,3> final_tria;
  GridGenerator::merge_triangulations(tria_3_4, tria3, final_tria);
  
  GridOut go;
  go.write_msh(final_tria, deallog.get_file_stream());
  // std::ofstream ofile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d.msh");
  // go.write_msh(final_tria, ofile);  
}
