IFEM source code
================

Copyright (C) 2014 by 
Luca Heltai (1), Saswati Roy (2), and Francesco Costanzo (3)

(1) Scuola Internazionale Superiore di Studi Avanzati
    E-mail: luca.heltai@sissa.it
(2) Center for Neural Engineering, The Pennsylvania State University
    E-Mail: sur164@psu.edu
(3) Center for Neural Engineering, The Pennsylvania State University
    E-Mail: costanzo@engr.psu.edu

This code was developed starting from the example
step-33 of the deal.II FEM library.

This file is subject to LGPL version LGPL 2.1 or later and may not be
distributed without copyright and license information. Please refer to
section 5 of this file for further information on this license.

1. Deal.II Requirements:
========================

The FEIBM source code requires the deal.II 8.0 library or greater. It
has also been tested with the current svn release deal.II 8.2.pre, and
a CMakeLists.txt file which will allow you to compile it with later
deal.II versions has been included.

In what follows, we assume that the user has installed the deal.II
library in the directory

DEAL_II_DIR

and that the user has defined the enviroment variable DEAL_II_DIR to
point to the correct location. If this enviroment variable is not set,
the user should specify it by hand when running cmake in order for
cmake to properly locate the deal.II library. For the program to work
properly, deal.II should be configured with support for UMFPACK.
  
2. Installation procedure:
==========================

The provided tgz archive should be unzipped in a dedicated 
subdirectory, with the commands

cd PATH_WHERE_YOU_WANT_THIS_CODE
tar xvfz ans-ifem-v1.0.tgz

or 

git clone https://bitbucket.org/heltai/ans-immersed-finite-element-method ans-ifem

The program can then be compiled by running

cd ans-ifem
cmake -DDEAL_II_DIR=/path/to/deal.II .
make

3. Running instructions:
========================

Once the program has been compiled, it can be run by typing 

./ifem

or 

./ifem parameters.prm

in the directory ans-ifem.
The program uses parameter files to set its runtime variables. The
file 

immersed_fem.prm 

is an example of such files, and it is the default one used if none is
specified at run time. If the specified file does not exists, the
program will attempt to create one with default values for you. 
The directory 

prms/

contains all parameter files used to produce the results presented in
the paper.

4. Extensive documentation:
===========================

If the user has the program Doxygen installed, a complete and
browsable documentation of the source code itself can be generated in
one of two ways: 

1. 

cd step-feibm/doc
make

In this case, the documentation will be accessible at the
address

DEAL_II_DIR/examples/step-feibm/doc/html/index.html


2.
If the user wants the deal.II documentation to be inlined, then 
the code should be extracted under the $DEAL_II_DIR/examples 
directory, i.e., under $DEAL_II_DIR/examples/step-feibm.
The documentation has been constructed in a way which is compatible
with the examples structure of the deal.II library. If the code is
extracted in the examples directory, then by typing

cd DEAL_II_DIR/
make online-doc

then the entire deal.II documentation, 
together with the documentation of the step-feibm
source code, will be accessible at the address

DEAL_II_DIR/doc/doxygen/deal.II/step_feibm.html

In the second case, the process is a lot longer but the documentation
generated thus has the added benefits of being fully integrated with
the |deal.II| documentation as well as having hyperlinks to all
the |deal.II| classes that have been used in our program.

5. Licence Informations
=======================

The step-feibm library has been placed under an Open Source license,
in the sense advocated by the Open Source Initiative. 

You are thus free to copy and use it, and you have free access to all
source codes. However, step-feibm is not in the public domain, it is
property of and copyrighted by the authors, and there are restrictions
on its use. We will give some hints on license issues first. The
legally binding license is the Q Public License (QPL), included here:

    THE Q PUBLIC LICENSE
    version 1.0

    Copyright (C) 1999-2005 Trolltech AS, Norway.
    Everyone is permitted to copy and distribute this license document.

    The intent of this license is to establish freedom to share and change the
    software regulated by this license under the open source model.

    This license applies to any software containing a notice placed by the
    copyright holder saying that it may be distributed under the terms of
    the Q Public License version 1.0. Such software is herein referred to as
    the Software. This license covers modification and distribution of the
    Software, use of third-party application programs based on the Software,
    and development of free software which uses the Software.


                                     Granted Rights

    1. You are granted the non-exclusive rights set forth in this license
       provided you agree to and comply with any and all conditions in this
       license. Whole or partial distribution of the Software, or software
       items that link with the Software, in any form signifies acceptance of
       this license.

    2. You may copy and distribute the Software in unmodified form provided
       that the entire package, including - but not restricted to - copyright,
       trademark notices and disclaimers, as released by the initial developer
       of the Software, is distributed.

    3. You may make modifications to the Software and distribute your
       modifications, in a form that is separate from the Software, such as
       patches. The following restrictions apply to modifications:

         a. Modifications must not alter or remove any copyright notices in
            the Software.

         b. When modifications to the Software are released under this
            license, a non-exclusive royalty-free right is granted to the
            initial developer of the Software to distribute your modification
            in future versions of the Software provided such versions remain
            available under these terms in addition to any other license(s) of
            the initial developer.

    4. You may distribute machine-executable forms of the Software or
       machine-executable forms of modified versions of the Software, provided
       that you meet these restrictions:

         a. You must include this license document in the distribution.

         b. You must ensure that all recipients of the machine-executable forms
            are also able to receive the complete machine-readable source code
            to the distributed Software, including all modifications, without
            any charge beyond the costs of data transfer, and place prominent
            notices in the distribution explaining this.

         c. You must ensure that all modifications included in the
            machine-executable forms are available under the terms of this
            license.

    5. You may use the original or modified versions of the Software to
       compile, link and run application programs legally developed by you
       or by others.

    6. You may develop application programs, reusable components and other
       software items that link with the original or modified versions of the
       Software. These items, when distributed, are subject to the following
       requirements:

         a. You must ensure that all recipients of machine-executable forms of
            these items are also able to receive and use the complete
            machine-readable source code to the items without any charge
            beyond the costs of data transfer.

         b. You must explicitly license all recipients of your items to use
            and re-distribute original and modified versions of the items in
            both machine-executable and source code forms. The recipients must
            be able to do so without any charges whatsoever, and they must be
            able to re-distribute to anyone they choose.

         c. If the items are not available to the general public, and the
            initial developer of the Software requests a copy of the items,
            then you must supply one.


                                Limitations of Liability

    In no event shall the initial developers or copyright holders be liable
    for any damages whatsoever, including - but not restricted to - lost
    revenue or profits or other direct, indirect, special, incidental or
    consequential damages, even if they have been advised of the possibility
    of such damages, except to the extent invariable law, if any, provides
    otherwise.


                                      No Warranty

    The Software and this license document are provided AS IS with NO WARRANTY
    OF ANY KIND, INCLUDING THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE.

                                     Choice of Law

    This license is governed by the Laws of Italy. Disputes shall be settled
    by Trieste City Court.
