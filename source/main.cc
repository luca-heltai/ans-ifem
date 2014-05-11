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

// @sect3{Include files}
// We include those elements of the deal.ii library
// whose functionality is needed for our purposes.

#include "immersed_fem.h"
#include "ifem_parameters.h"

using namespace std;

// The main function: essentially the same as in the
// <code>deal.II</code> examples.
int main(int argc, char **argv)
{
  try
    {
      IFEMParameters<2> par(argc,argv);
      ImmersedFEM<2> test (par);
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
