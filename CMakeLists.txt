# Set the name of the project and target:
SET(TARGET "ifem")
SET(POSTPROCESSOR "postprocessor")

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.0
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
    REQUIRED
  )

LIST(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/
  ${CMAKE_SOURCE_DIR}/cmake/modules/
  )


INCLUDE(setup_external_macros)
FILE(GLOB _macro_files "cmake/macros/*.cmake")
FOREACH(_file ${_macro_files})
  INCLUDE(${_file})
ENDFOREACH()

ADD_SUBDIRECTORY(tests)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

INCLUDE_DIRECTORIES("include/")
PROJECT(${TARGET})
SET(TARGET_SRC
        source/exact_solution_ring_with_fibers.cc
        source/ifem_parameters_generalized.cc
        source/immersed_fem_generalized.cc
        source/main.cc)
ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})

PROJECT(${POSTPROCESSOR})
SET(POSTPROCESSOR_SRC
	source/exact_solution_ring_with_fibers.cc
	source/ifem_parameters_generalized.cc 
	source/post_processor.cc 
	source/main_post_processor.cc)
ADD_EXECUTABLE(${POSTPROCESSOR} ${POSTPROCESSOR_SRC})
DEAL_II_SETUP_TARGET(${POSTPROCESSOR})

#-- Add an Option to toggle the generation of the API documentation
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
if(BUILD_DOCUMENTATION)
  FIND_PACKAGE(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR 
      "Doxygen is needed to build the documentation. Please install it correctly or set BUILD_DOCUMENTATION to OFF")
  endif()
  #-- Configure the Template Doxyfile for our specific project
  configure_file(Doxyfile.in 
                 ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target (docs 
	COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
	SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
endif()
