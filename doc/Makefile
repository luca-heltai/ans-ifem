all: ../../../doc/doxygen/tutorial/doxygen/step-feibm.h Doxyfile
	doxygen Doxyfile

../../../doc/doxygen/tutorial/doxygen/step-feibm.h: ../step-feibm.cc intro.dox results.dox
	cd ../../../doc/doxygen/tutorial
	make
