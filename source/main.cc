// Copyright (C) 2014 by Luca Heltai (1),
// Saswati Roy (2), and Francesco Costanzo (3)
//
// (1) Scuola Internazionale Superiore di Studi Avanzati
//     E-mail: luca.heltai@sissa.it
// (2) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: sur164@psu.edu
// (3) Center for Neural Engineering, The Pennsylvania State University
//     E-Mail: costanzo@engr.psu.edu
//
// This code was developed starting from the example
// step-33 of the deal.II FEM library.
//
// This file is subject to LGPL and may not be  distributed without
// copyright and license information. Please refer
// to the webpage http://www.dealii.org/ -> License
// for the  text  and further information on this license.
//
// Keywords: fluid-structure interaction, immersed method,
//           finite elements, monolithic framework
//
// Deal.II version:  deal.II 8.2.pre

/** \mainpage

   <br>
   <a name="Intro"></a>
   <h1> Introduction </h1>

   <p> We present the implementation of a solution scheme for
   fluid-structure interaction problems via the finite element
   software library <tt>deal.II</tt>.  Specifically, we implement an
   immersed finite element method in which two independent
   discretizations are used for the fluid and immersed deformable
   body.  In this type of formulation the support of the equations of
   motion of the fluid is extended to cover the union of the solid and
   fluid domains.  The equations of motion over the extended solution
   domain govern the flow of a fluid under the action of a body force
   field.  This body force field informs the fluid of the presence of
   the immersed solid.  The velocity field of the immersed solid is
   the restriction over the immersed domain of the velocity field in
   the extended equations of motion.  The focus of this paper is to
   show how the determination of the motion of the immersed domain is
   carried out in practice.  We show that our implementation is
   automatically obtained from the choice of finite element spaces
   over the immersed solid and the extended fluid domains.  We present
   a few results concerning the accuracy of the proposed method.  For
   a careful discussion of the proposed approach see Heltai, L. and
   F. Costanzo (2012), "Variational implementation of immersed finite
   element methods," <i>Computer Methods in Applied Mechanics and
   Engineering</i>, 229–232, p. 110–127, which we will
   denoted by [HC2012].  </p>



<a name="Governing Equations"></a>
<h3> Governing Equations </h3>
<p>
<b>Fluid and Solid Domains.</b>
Referring to the figure below, \f$B_{t}\f$ represents the configuration of a regular body at time \f$t\f$.   \f$B_{t}\f$ is a (possibly multiply connected) proper subset of a fixed control volume \f$\Omega\f$:  \f$\Omega\setminus B_{t}\f$ is filled by a fluid and \f$B_{t}\f$ is an <i>immersed body</i>.  Following common practice in continuum mechanics, we refer to \f$B_{t}\f$ as the <i>current</i> configuration of the immersed body.  By contrast, we denote by \f$B\f$ the reference configuration of the immersed body.  We denote the position of points in \f$B\f$ by \f$s\f$, whereas we denote the position at time \f$t\f$ of a point \f$P\f$ in \f$\Omega\f$ by \f$x_{P}(t)\f$.  For the examples in the results section we take \f$B\f$ to coincide with the initial configuration \f$B_{0}\f$.
<table width="60%" align="center">
  <tr>
    <td ALIGN="center">
      @image html step-feibm.geometry.png
    </td>
  </tr>
</table>

<b>Motion of the Immersed Body.</b>
The motion of the solid is described by a function \f$\chi(s,t)\f$, which gives the position \f$x\f$ at time \f$t\f$ of a particle of the immersed domain with position \f$s\f$ in the reference configuration \f$B\f$, i.e., \f$x = \chi(s,t)\f$ for \f$s \in B\f$. The deformation gradient is the tensor \f$F = \nabla_{s} \chi(s,t)\f$, where \f$\nabla_{s}\f$ denotes the gradient operator relative to position in the reference configuration.  We denote the determinant of \f$F(s,t)\f$ by \f$J(s,t)\f$ and we assume that \f$J(s,t) > 0\f$ for all \f$s \in B\f$ and \f$t\f$.  From a practical viewpoint, as it is often done in solids mechanics, we describe the motion of the body through its displacement function.  We denote the displacement by \f$w(s,t)\f$.  The relation between \f$\chi(s,t)\f$ and \f$w\f$ is as follows:
@f[
x = s + w(s,t).
@f]
From the above relation we have that \f$F(s,t) = I + \nabla_{s} w(s,t)\f$, where \f$I\f$ is the identity tensor.
</p>

<p>
<b>Constitutive Equations.</b>
The fluid is assumed to be Newtonian so that its Cauchy stress is \f$\sigma_{f} = -p I + \eta [\nabla_{x} u + (\nabla_{x} u)^{T}]\f$, where \f$p\f$ is the Lagrange multiplier for the enforcement of incompressibility, \f$\eta\f$ is the dynamic viscosity, \f$\nabla_{x}\f$ is the gradient operator relative to position in the current configuration, and \f$u\f$ is the velocity field.  For a Newtonian fluid, \f$p\f$ is also the pressure (mean normal stress).  The immersed solid is assumed to be incompressible and viscoelastic, with the viscous response identical to that of the fluid.  The elastic response is assumed to be admit a strain energy \f$W_{s}^{e}(F)\f$, which we assumed to be a convex function of the deformation gradient.  Hence the Cauchy stress in the solid is given by \f$\sigma_{s} = - p I + \eta [\nabla_{x} u + (\nabla_{x} u)^{T}] + \sigma_{s}^{e}\f$, with \f$\sigma_{s}^{e} = J^{-1} P_{s}^{e} F^{-T}\f$, where \f$P_{s}^{e} = \partial W_{s}^{e}/\partial F\f$ is the first Piola-Kirchhoff stress tensor of the solid.
</p>
<p>
<b>Velocity field and Displacement of the Immersed Body.</b>
The velocity field \f$u(x,t)\f$, with \f$x \in \Omega\f$, represents that velocity of the particle occupying the point \f$x\f$ at time \f$t\f$.  As such, this field describes the velocity of the fluid for \f$x \in \Omega/B_{t}\f$ and the velocity of the solid for \f$x \in B_{t}\f$.  Therefore, using the displacement function \f$w(s,t)\f$, for \f$s \in B\f$ and for al \f$t\f$, we have
@f[
  u(x,t)\big|_{x = s + w(s,t)} = \frac{\partial w(s,t)}{\partial t}.
@f]
</p>
<p>
<b>Conditions at the Boundary of the Immersed Body.</b>
The boundary of the immersed body is viewed as a material surface.  Therefore, the balance of linear momentum requires that the traction field be continuous across the boundary of the immersed body.  In addition, we assume that there is no slip between the immersed body and the surrounding fluid.
</p>
<b>Governing Equations: Strong Form.</b>
The motion of the system is governed by the following three equations, which, respectively, represent the balance of linear momentum, balance of mass accounting for incompressibility, and velocity compatibility:
@f[
  \nabla \cdot \sigma(x,t) + \rho b = \rho \biggl[\frac{\partial u(x,t)}{\partial t} + (\nabla u(x,t)) u(x,t) \biggr] \quad {\rm in}~\Omega,\quad
  \nabla \cdot u(x,t) = 0 \quad {\rm in}~\Omega, \quad
  u(x,t)\big|_{x = s + w(s,t)} = \frac{\partial w(s,t)}{\partial t} \quad {\rm in}~B,
@f]
where "\f$\nabla_{x} \cdot\f$" denotes the divergence operator (relative to position in the current configuration), \f$\rho\f$ is the density (here assumed to be a constant), \f$b\f$ is a (prescribed) body force field, and where \f$\sigma(x,t)\f$ is the Cauchy stress field in the entire domain \f$\Omega\f$, i.e., \f$\sigma(x,t) = \sigma_{f}(x,t)\f$ for \f$x \in \Omega/B_{t}\f$ and \f$\sigma(x,t) = \sigma_{s}(x,t)\f$ for \f$x \in B_{t}\f$.
</p>

<p>
As far as boundary conditions are concerned, we assume that a velocity and a traction distribution are prescribed on complementary subsets of the boundary of \f$\Omega\f$.  Specifically,  letting \f$\partial \Omega_{D} \cup \partial \Omega_{N} = \partial \Omega\f$, with \f$\partial \Omega_{D} \cap \partial \Omega_{N} = \emptyset\f$,
@f[
u(x,t) = u_{g}(x,t)~{\rm for}~ x\in \partial \Omega_{D}
\quad{\rm and}\quad
\sigma(x,t) n(x,t) = \tau_{g}(x,t)~{\rm for}~x \in \partial \Omega_{N},
@f]
where \f$u_{g}\f$ and \f$\tau_{g}\f$ are prescribed velocity and tractions distributions, and where \f$n\f$ denotes the outward unit normal to the \f$\partial \Omega\f$.
</p>

<a name ="Weak Formulation"></a>
<h3> Weak Formulation</h3>
The primary unknowns of the problem are the velocity field \f$u(x,t)\f$ in \f$\Omega\f$, the displacement field \f$w(s,t)\f$ in \f$B\f$, and the Lagrange multiplier field \f$p\f$.  For these we select appropriate function spaces \f$V\f$, \f$Y\f$, and \f$Q\f$ (for details see [HC2012]), respectively. With this in mind, the weak formulation from which the discrete formulation is derived is as follows:
@f[
\int_{\Omega} \rho(\dot{u}(x,t) - b(x,t)] \cdot v(x,t)
- \int_{\Omega} p(x,t) \, \nabla_{x} \cdot v(x,t)
+
\int_{\Omega} \eta [\nabla u(x,t) + (\nabla u(x,t))^{T}] \cdot \nabla v(x,t)
- \int_{\partial \Omega_{N}} \tau_{g}(x,t) \cdot v(x,t)
+ \int_{B} P_{s}^{e}(s,t) F^{T}(s,t) \cdot [\nabla_{x} v(x,t)\big|_{x = s + w(s,t)}] = 0,
@f]
@f[
\int_{\Omega} (\nabla_{x} \cdot u ) \; q = 0,
@f]
@f[
\Phi_{B} \int_{B} \biggl(\frac{\partial w(s,t)}{\partial t} - u(x,t)\big|_{x = s + w(s,t)} \biggr) \cdot y = 0,
@f]
for all \f$v \in V_{0}\f$, \f$y \in Y\f$, and \f$q \in Q\f$.  We observe that, since \f$F = I + \nabla_{s} w(s,t)\f$, \f$P_{s}^{e}\f$ is a function of the displacement gradient \f$\nabla_{s} w\f$.  Finally, \f$\Phi_{B}\f$ is a constant needed to ensure that the dimensions of the equations are homogeneous with the dimensions of the rest of the equations.

<p>
The above weak formulation can be viewed as consisting of a typical formulation for the Navier-Stokes equations with some nonstandard terms.  The latter are the last term on the left-hand of the first equation, which can be interpreted as a body force distribution that informs the fluid of the presence of the solid, and the very last equation, which is crucial for the tracking of the motion of the immersed body.  What makes these terms unusual is the fact that, in the implementation of the discrete formulation, they require the evaluation of integrals over the triangulation of the domain \f$B\f$ of functions defined over the triangulation of the domain \f$\Omega\f$. In the next section we provide a discussion of how this is implemented in practice.
</p>

<p>
In [HC2012] we have discussed in detail the fact that the discrete formulation derived from the above weak formulation requires a careful treatment of the term
@f[
\int_{B} P_{s}^{e}(s,t) F^{T}(s,t) \cdot [\nabla_{x} v(x,t)\big|_{x = s + w(s,t)}]
@f]
to yield a stable formulation.  Specifically, it was shown that the above term should be viewed as the composition of two operators: (i) an "elastic" operator defined through the following term
@f[
\int_{B} P_{s}^{e}(s,t) \cdot \nabla_{s} y,
@f]
and (ii) a "spread" operator whose definition stems from the last equation of the of the weak formulation (for details see, in particular, Remark 9 and Theorem 4 in  [HC2012]). The code in the present example includes the determination of the elastic and spread operators as well as their composition.
</p>

<a name = "implementation"></a>
<h3> Implementation </h3>
The implementation of the various terms that are common to the Navier-Stokes equations is done in a standard fashion and will not be discussed here.  As alluded above, here we limit ourselves to the description of how to carry out the integration over \f$B\f$ of functions that are available through their finite element representation over the triangulation of \f$\Omega\f$.

<p>
Referring to the third equation in the weak formulation, let's consider the term
@f[
\Phi_{B} \int_{B} u_{h}(x,t)\big|_{x = s + w_{h}(s,t)}  \cdot y_{h},
@f]
where \f$u_{h}\f$ denotes the finite element representation of \f$u\f$ given by interpolation functions supported over the triangulation \f$\Omega_{h}\f$ of \f$\Omega\f$, and where \f$y_{h}\f$ denotes shape functions supported over the triangulation \f$B_{h}\f$ of the domain \f$B\f$: the construction of the above term draws information from two independent triangulations.  The above integral is computed by summing contributions from each cell \f$K\f$ of \f$B_{h}\f$.  Each of these contributions is a sum over the \f$N_{Q}\f$ quadrature points.  We illustrate this via the following figure:
<table width="60%" align="center">
  <tr>
    <td ALIGN="center">
      @image html step-feibm.integration.png
    </td>
  </tr>
</table>
In our code, for each cell of the immersed body, we start by determining the position of the quadrature points of the element corresponding to the cell at hand.  The position of the quadrature point is determined both relative to the reference unit element and relative to the global coordinate system adopted for the calculation, through the mappings:
@f[
s_{K}: \hat{K} := [0,1]^{d} \mapsto K \in B_{h},
\quad{\rm and}\quad
s + w_{h}: K \mapsto {\rm current~position~of~solid~cell},
@f]
where \f$\hat{K}\f$ is the reference unit element and \f$d\f$ is the spatial dimension of the immersed solid.  These maps allow us to determine the global coordinates of the quadrature points. These coordinates are then passed to a search algorithm that identifies the cells in \f$\Omega_{h}\f$ that contain the points in question.  In turn, this identification allows us to evaluate the functions \f$v_h\f$. The overall operation is illustrated in the figure above where we show a cell of \f$B_{h}\f$ straddling four cells of \f$\Omega_{h}\f$ denoted fluid cells  A–D.  The quadrature points over the solid cell are denoted by filled circles.  The contribution to the above integral due to the solid cell is then computed by summing the partial contributions corresponding to each of the fluid cells intersecting the solid cell in question.  The implementation of an efficient search algorithm responsible for identifying the fluid cells intersecting an individual solid cell is the only technically challenging part of the procedure.  We use the built in facilities of the <code>deal.II</code> library to perform this task. Once the fluid cells containing the quadrature points of a given solid cell are found, we determine the value of \f$v_{h}\f$ at the quadrature points using the interpolation infrastructure inherent in the finite element representation of fields defined over \f$\Omega_{h}\f$.  The <code>deal.II</code> class we use for this implementation is the <code>FEFieldFunction</code>.

<a name="results"></a>
<h1>Results</h1>

<a name="Equilibrium of an immersed ring of a neo-Hookean material"></a>
<h3> Equilibrium of an Immersed Annulus </h3>
We report the results obtained by running our code with the parameter file <tt>prms/RingEqm_XXX_fref_Y_param.prm</tt>, where <tt>XXX</tt> is either <tt>dgp</tt> or <tt>feq</tt> and <tt>Y</tt> is 4, 5, 6 or 7, according to the type of pressure finite element and to the fluid refinement level.  The objective of this test is to compute the equilibrium state of an initially undeformed thick annular cylinder submerged in a stationary incompressible fluid that is contained in a rigid prismatic box having a square cross-section. Our simulation is two-dimensional with inner radius \f$R\f$ and thickness \f$w\f$ and filled with a stationary fluid contained in a square box of edge length \f$l\f$ (see figure below).
<table width="60%" align="center">
  <tr>
    <td width="20" align="center">
@image html step-feibm.annulus.png
    </td>
  </tr>
</table>
The cylinder is located coaxially with respect to that of the box and it is subjected to the hydrostatic pressure of the fluid \f$p_{i}\f$ and \f$p_{o}\f$ at its inner and outer walls, respectively. Negligible body forces are acting on the system and there is no inflow or outflow of fluid across the walls of the box. Since both the solid as well as the fluid are incompressible it is expected that neither the cylinder nor the fluid will move at all. Therefore, the problem reduces to determining the equilibrium solution for the Lagrange multiplier field \f$p\f$. The elastic behavior of the ring is governed by a continuous distribution of concentric fibers lying in the circumferential direction. The constitutive relation for the first Piola-Kirchhoff stress for the ring is then
@f[
\hat{P} = -p_{\rm s} F^{-T} + \mu^{e} F \hat{u}_{\Theta} \otimes \hat{u}_{\Theta},
@f]
where \f$\mu^{e}\f$ is a constant modulus of elasticity, \f$p_{s}\f$ is the Lagrange multiplier that enforces incompressibility of the annulus, \f$F\f$ is the deformation gradient, \f$\hat{u}_{\Theta}\f$ is a unit vector in the circumferential direction in the reference configuration.  Enforcing the balance of linear momentum and boundary conditions, it can be shown that the exct solution to this problem is as follows:
@f[
p=
p_{o}=-\frac{\pi \mu^{e}}{2 l^{2}} [ (R+w)^{2}-R^{2}]
\quad {\rm for}~R + w \leq r,
@f]
@f[
p= p_{s}=\mu^{e} \ln (\frac{R+w}{r})-\frac{\pi \mu^{e}}{2 l^{2}} [ (R+w)^{2}-R^{2}]\quad{\rm for}~R < r < R+w,
@f]
@f[
p= p_{i}=\mu^{e} \ln (1+\frac{w}{R})-\frac{\pi \mu^{e}}{2 l^{2}} [ (R+w)^{2}-R^{2}]\quad{\rm for}~r \leq R,
@f]
In our numerical simulations we have used \f$R = 0.25\,{\rm m}\f$, \f$w=0.06250\,{\rm m}\f$, \f$l=1.0\,{\rm m}\f$ and \f$\mu^{e}=1\,{\rm Pa}\f$. For these values, the above exact solution gives \f$p_{i}=0.16792\,{\rm Pa}\f$ and \f$p_{o}=-0.05522\,{\rm Pa}\f$. In our numerical solution, we have also used \f$\rho=1.0\,{\rm kg}/{\rm m}^{3}\f$, dynamic viscosity \f$\mu=1.0\,{\rm Pa}\!\cdot\!{\rm s}\f$, and time step size \f$h=1 \times 10^{-3}\,{\rm s}\f$. In addition, we have used \f$Q2\f$ elements to represent the displacement \f$w\f$ of the solid, whereas we have used (i) \f$Q2/P1\f$ elements, and (ii) \f$Q2/Q1\f$ elements to represent the velocity \f$v\f$ and Lagrange multiplier \f$p\f$ over the control volume. In the figure below a sample profile of \f$p\f$ over the entire control volume and its variation along different values of \f$y\f$, after one time step for \f$Q2/P1\f$ elements.
<table width="60%" align="center">
<caption align="bottom">The values of \f$p\f$ after one time step when using \f$Q2/P1\f$ elements.</caption>
<tr>
<td>@image html step-feibm.pQ2P1m.png</td>
<td>@image html step-feibm.pQ2P1p.png</td>
</tr>
</table>
In the figure below a sample profile of \f$p\f$ over the entire control volume and its variation along different values of \f$y\f$, after one time step for \f$Q2/Q1\f$ elements.
<table align="center" class="image">
<caption align="bottom">The values of \f$p\f$ after one time step when using \f$Q2/Q1\f$ elements.</caption>
<tr>
<td>@image html step-feibm.pQ2Q1m.png</td>
<td>@image html step-feibm.pQ2Q1p.png</td>
</tr>
</table>
We assess the convergence property of our numerical scheme by obtaining the convergence rate of the error between the exact and the numerical solutions of this equilibrium problem. The order of the rate of convergence (see the tables below) is \f$2.5\f$ for the \f$L^{2}\f$ norm of the velocity, 1.5 for the \f$H^{1}\f$ norm of the velocity and \f$1.5\f$ for the \f$L^{2}\f$ norm of the pressure. In all these numerical tests we have used 1856 cells with 15776 DoFs for the solid.
<table vspace="20" align="center" class="image">
<caption align="top">Convergence rates as a function of refinement in the control volume (refinement in the immersed domain is kept constant) for the \f$Q2/P1\f$ case.</caption>
<tr>
<td>@image html step-feibm.pQ2P1c.png</td>
</tr>
</table>
<table vspace="20" align="center" class="image">
<caption align="top">Convergence rates as a function of refinement in the control volume (refinement in the immersed domain is kept constant) for the \f$Q2/Q1\f$ case.</caption>
<tr>
<td>@image html step-feibm.pQ2Q1c.png</td>
</tr>
</table>


<a name="Equilibrium of an immersed ring of a neo-Hookean material"></a>
<h3> Disk Entrained in a Lid-Driven Cavity Flow </h3>
We report the results obtained by running our code with the parameter file <tt>LDCFlow_Ball_DGP_INH1_param.prm</tt>.  Referring to the figure below, the disk has a radius \f$R=0.2\,{\rm m}\f$ and its center \f$C\f$ is initially positioned at \f$x=0.6\,{\rm m}\f$ and \f$y=0.5\,{\rm m}\f$ in the square cavity whose each edge has the length \f$l=1.0\,{\rm m}\f$. Body forces on the system are negligible.  The constitutive model for the immersed solid is as follows:
@f[
\hat{P} = -p_{\rm s} I + \mu^{e} F,
@f]
where we have used the following parameters: \f$\rho=1.0\,{\rm kg}/{\rm m}^3\f$, dynamic viscosity \f$\mu=0.01\,{\rm Pa}\!\cdot\!{\rm s}\f$, shear modulus \f$\mu^{e} = 0.1\,{\rm Pa}\f$.  The prescribed value of the horizontal velocity is  \f$U=1.0\,{\rm m}/{\rm s}\f$. For our numerical simulations we have used \f$Q2\f$ elements to represent the displacement \f$w\f$ of the disk whereas we have used \f$Q2/P1\f$ element for the fluid. The disk is represented using 320 cells with 2626 DoFs and the control volume has 4096 cells and 45570 DoFs. The time step size \f$h=1 \times 10^{-2}\,{\rm s}\f$. We consider the time interval \f$0 < t \leq 8\,{\rm s}\f$ during which the disk is lifted from its initial position along the left vertical wall, drawn along underneath the lid and finally dragged downwards along the right vertical wall of the cavity. This is shown in the figure below.
<table align="center" class="image">
<caption align="bottom">Disk entrained in a lid-driven cavity flow.</caption>
<tr>
<td>@image html step-feibm.lidcavity0.png</td>
<td>@image html step-feibm.lidcavity1.png</td>
</tr>
<tr>
<td>@image html step-feibm.lidcavity2.png</td>
<td>@image html step-feibm.lidcavity3.png</td>
</tr>
<tr>
<td>@image html step-feibm.lidcavity4.png</td>
<td>@image html step-feibm.lidcavity5.png</td>
</tr>
</table>


 */

#include "immersed_fem.h"
#include "immersed_fem_generalized.h"
#include "ifem_parameters.h"
#include "ifem_parameters_generalized.h"

using namespace std;

//! The main function: essentially the same as in the
//! <code>deal.II</code> examples.
int main(int argc, char **argv)
{
  try
    {
      IFEMParametersGeneralized<3> par(argc,argv);
      ImmersedFEMGeneralized<3> test (par);
      test.run ();
    }
  catch (exception &exc)
    {
      cerr
          << endl
          << endl
          << "----------------------------------------------------"
          << endl;
      cerr
          << "Exception on processing: "
          << endl
          << exc.what()
          << endl
          << "Aborting!"
          << endl
          << "----------------------------------------------------"
          << endl;
      return 1;
    }
  catch (...)
    {
      cerr
          << endl
          << endl
          << "----------------------------------------------------"
          << endl;
      cerr
          << "Unknown exception!"
          << endl
          << "Aborting!"
          << endl
          << "----------------------------------------------------"
          << endl;
      return 1;
    }
  cout
      << "----------------------------------------------------"
      << endl
      << "Apparently everything went fine!"
      << endl;
  return 0;
}
