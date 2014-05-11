// Copyright (C) 2014 by Luca Heltai (1), Saswati Roy (2), and
// Francesco Costanzo (3)
//
// (1) Scuola Internazionale Superiore di Studi Avanzati
//     E-mail: luca.heltai@sissa.it
// (2) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: sur164@psu.edu
// (3) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: costanzo@engr.psu.edu
//
// This file is subject to LGPL and may not be distributed without
// copyright and license information. Please refer to the webpage
// http://www.dealii.org/ -> License for the text and further
// information on this license.
#ifndef exact_solution_ring_with_fibers_h
#define exact_solution_ring_with_fibers_h

#include "ifem_parameters.h"
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

//! This class provides the exact distribution of the Lagrange
// multiplier for enforcing incompressibility both in the fluid and in
// the immersed domains for a specific problem. The latter concerns
// the equilibrium of a circular cylinder immersed in an
// incompressible Newtonian fluid.  The immersed domain is assumed to
// be an incompressible elastic material with an elastic response
// proportional to the stretch of elastic fibers wound in the
// circumferential direction (hoop). The constitutive equations of the
// cylinder correspond to the choice of "Solid constitutive model" as
// "CircumferentialFiberModel".  Finally, we refer to this particular
// problem as the "Hello world" problem for immersed methods.  We
// learned this expression from our colleague Boyce E. Griffith,
// currently at the Leon H. Charney Division of Cardiology, Department
// of Medicine, NYU School of Medicine, New York University.
template<int dim>
class ExactSolutionRingWithFibers :
  public Function<dim>
{
  public:

// No default constructor is defined. Simulation objects must be
// initialized by assigning the simulation parameters, which are
// elements of objects of type <code>IFEMParameters</code>.

    ExactSolutionRingWithFibers (IFEMParameters<dim> &par);

    void vector_value (const Point <dim> &p,
                       Vector <double> &values) const;

    void vector_value_list(const vector< Point<dim> > &points,
                           vector <Vector <double> > &values ) const;

// Inner radius of the ring.

    double R;

// Width of the ring.

    double w;

// Edge length of the square control volume.

    double l;

// Center of the ring.

    Point<dim> center;

  private:

    IFEMParameters<dim> &par;
};

#endif
