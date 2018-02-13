
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

  Triangulation<2,2> tria_2d, tria_2d_isotropic;
  {
    GridIn<2,2> gi;
    gi.attach_triangulation(tria_2d);
    std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_2d.msh");  
    gi.read_msh(infile);
  }
  {
    GridIn<2,2> gi;
    gi.attach_triangulation(tria_2d_isotropic);
    std::ifstream infile(SOURCE_DIR "/../../meshes/SchaeferTurek_2d_isotropic.msh");
    gi.read_msh(infile);
  }
  
  GridOut go;
  GridOutFlags::Msh flags(true, true);
  go.set_flags(flags);
  go.write_msh(tria_2d, deallog.get_file_stream());
  go.write_msh(tria_2d_isotropic, deallog.get_file_stream());
  // std::ofstream ofile(SOURCE_DIR "/../../meshes/SchaeferTurek_3d.msh");
  // go.write_msh(tria, ofile);  
  // ofile.close();
  // ofile.open(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_isotropic.msh");
  // go.write_msh(tria_isotropic, ofile);  
  // ofile.close();
  // ofile.open(SOURCE_DIR "/../../meshes/SchaeferTurek_3d_solid.msh");
  // go.write_msh(solid, ofile);  
}
