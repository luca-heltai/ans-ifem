
#include "../tests.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

// Immersed solid grid, used for the "floating ball" experiment.

int
main()
{
  initlog();

  // In the article, the center is not (0,0). We make it (0,0), so that its
  // later transformation is easier.
  const Point<2> center;
  const double radius = 0.125;

  const SphericalManifold<2,2> manifold(center);
  Triangulation<2,2> tria;
  GridGenerator::hyper_ball(tria, center, radius);
  tria.set_all_manifold_ids_on_boundary(0);
  tria.set_manifold(0, manifold);

  tria.refine_global(1);
  for(unsigned int i=2; i<6; ++i)
  {
    GridOut go;
    GridOutFlags::Msh flags(true, true);
    go.set_flags(flags);
    go.write_msh(tria, deallog.get_file_stream());
    std::ofstream ofile(SOURCE_DIR "/../../meshes/floating_ball_ref"+Utilities::int_to_string(i)+".msh");
    go.write_msh(tria, ofile);
    tria.refine_global(1);
  }
  deallog << "OK" << std::endl;
}
