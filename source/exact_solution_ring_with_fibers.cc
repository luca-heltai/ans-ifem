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

#include "exact_solution_ring_with_fibers.h"

template<int dim>
ExactSolutionRingWithFibers<dim>::ExactSolutionRingWithFibers (
  IFEMParametersGeneralized<dim> &prm
)
  :
  Function<dim>(dim+1),
  par(prm)
{
  par.enter_subsection (
    "Equilibrium Solution of Ring with Circumferential Fibers"
  );
  R = par.get_double ("Inner radius of the ring");
  w = par.get_double ("Width of the ring");
  l = par.get_double ("Any edge length of the (square) control volume");
  center[0] = par.get_double ("x-coordinate of the center of the ring");
  center[1] = par.get_double ("y-coordinate of the center of the ring");
  par.leave_subsection ();
}

// It provides the Lagrange multiplier (pressure)
// distribution in the control volume and in the ring at equilibrium.

template<int dim>
void
ExactSolutionRingWithFibers<dim>::vector_value
(
  const Point <dim> &p,
  Vector <double> &values
) const
{
  double r = p.distance (center);

  values = 0.0;

  double p0 = -numbers::PI*par.mu/(2*l*l)*w*(2*R+w);

  if (r >= (R+w))
    values(dim) = p0;
  else if (r >= R)
    values(dim) = p0 + par.mu*log((R + w)/r);
  else
    values(dim) = p0 + par.mu*log(1.0 + w/R);
}

// It provides the Lagrange multiplier (pressure) distribution in the
// control volume and in the ring at equilibrium.

template<int dim>
void
ExactSolutionRingWithFibers<dim>::vector_value_list
(
  const vector< Point<dim> > &points,
  vector <Vector <double> > &values
) const
{
  Assert (points.size() == values.size(),
          ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int i = 0; i < values.size(); ++i)
    vector_value (points[i], values[i]);
}

template class ExactSolutionRingWithFibers<2>;
template class ExactSolutionRingWithFibers<3>;
