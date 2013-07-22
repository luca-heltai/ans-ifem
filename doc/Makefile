# include paths and global variables
include $(DEAL_II_DIR)/common/Make.global_options

guard-%:
	if [ "${${*}}" == "" ]; then \
                echo "Environment variable $* not set"; \
                exit 1; \
        fi

default: tmp/step-feibm.cc tmp/step-feibm.h guard-DEAL_II_DIR
	rm -f filter
	ln -s $(DEAL_II_DIR)/doc/doxygen/filter .
	doxygen Doxyfile


# Paths of perl utilities
dealdoxy := $D/doc/doxygen/tutorial
program2toc := $(dealdoxy)/program2toc
intro2toc := $(dealdoxy)/intro2toc
program2doxygen :=  $(dealdoxy)/program2doxygen
program2plain :=  $(dealdoxy)/program2plain
program2doxyplain := $(dealdoxy)/program2doxyplain
create_anchors := $(dealdoxy)/create_anchors

# plain names of those files that we shall treat. simply take all those
# directories in examples/ that have a /doc subdirectory
example-names := step-feibm


# prefix names for generated intermediate files
gen-example-names  := $(addprefix tmp/generated/, $(example-names))

# names of commented program files, plain files, and tocs for the commented
# and commented and plain versions of the program as well as a combined one
example-toc-prog   := $(addsuffix _toc.prog,$(gen-example-names))
example-toc-plain  := $(addsuffix _toc.plain,$(gen-example-names))
example-toc-intro  := $(addsuffix _toc.intro,$(gen-example-names))
example-toc-results:= $(addsuffix _toc.results,$(gen-example-names))
example-toc        := $(addsuffix _toc.combined,$(gen-example-names))

# same for intermediate files for doxygen output
example-dox-prog   := $(addsuffix _prog.dox,$(gen-example-names))
example-dox-plain  := $(addsuffix _plain.dox,$(gen-example-names))
example-plain      := $(addprefix tmp/,$(addsuffix .cc,$(example-names)))

# finally a target for combined Doxygen files
example-doxygen    := $(addprefix tmp/,$(addsuffix .h,$(example-names)))


# a makefile command that extracts the base name of a program
get_basename = $(shell echo $(1) | $(PERL) -pi -e 's/(tmp\/generated|tmp).//g; s/\..*//g; s/_(prog|plain|toc.*)//g;')



$(example-toc-prog):
	@echo ================== Making $@
	@cat ../step-feibm.cc \
	  | $(PERL) $(program2toc) \
	  > $@


$(example-toc-plain):
	@echo ================== Making $@
	@cat tmp/generated/$(call get_basename, $@)_toc.prog \
	  | $(PERL) -pi -e 's/href=\"\#/href=\"\#plain-/g;' \
	  > $@


$(example-toc-intro):
	@echo ================== Making $@
	@cat intro.dox \
	  | egrep -i "<h[3456]>.*</h[3456]>" \
	  | $(PERL) $(intro2toc) \
	  > $@


$(example-toc-results):
	@echo ================== Making $@
	@cat doc/results.dox \
	  | egrep -i "<h[3456]>.*</h[3456]>" \
	  | $(PERL) $(intro2toc) \
	  > $@


$(example-toc):
	@echo ================== Making $@
	@echo "<table class=\"tutorial\" width=\"50%\">" > $@
	@echo "<tr><th colspan=\"2\"><b><small>Table of contents</small></b></th></tr>" >> $@
	@echo "<tr><td width=\"50%\" valign=\"top\">" >> $@
	@echo "<ol>" >> $@
	@echo "  <li> <a href=\"#Intro\" class=bold>Introduction</a>" >> $@
	@cat tmp/generated/$(call get_basename, $@)_toc.intro >> $@
	@echo "  <li> <a href=\"#CommProg\" class=bold>The commented program</a>" >> $@
	@cat tmp/generated/$(call get_basename, $@)_toc.prog >> $@
	@echo "</ol></td><td width=\"50%\" valign=\"top\"><ol>" >> $@
	@echo "  <li value=\"3\"> <a href=\"#Results\" class=bold>Results</a>" >> $@
	@cat tmp/generated/$(call get_basename, $@)_toc.results >> $@
	@echo "  <li> <a href=\"#PlainProg\" class=bold>The plain program</a>" >> $@
	@echo "</ol> </td> </tr> </table>" >> $@

$(example-dox-prog):
	@echo ================== Making $@
	@cat ../step-feibm.cc \
	  | $(PERL) $(program2doxygen) \
	  > $@

$(example-plain):
	@echo ================== Making $@
	@cat ../step-feibm.cc | $(PERL) $(program2plain) > $@

$(example-dox-plain):
	@echo ================== Making $@
	@echo " * <a name=\"PlainProg\"></a>" > $@
	@echo " * <h1> The plain program</h1>" >> $@
	@echo " * " >> $@
	@echo " * (If you are looking at a locally installed deal.II version, then the" >> $@
	@echo " * program can be found at <i>" >> $@
	@echo " * " ../step-feibm.cc | $(PERL) -pi -e 's!/! /!g;' >> $@
	@echo " * </i>. Otherwise, this is only" >> $@
	@echo " * the path on some remote server.)" >> $@
	@echo " @code" >> $@
	@cat ../step-feibm.cc | $(PERL) $(program2doxyplain) >> $@
	@echo " @endcode" >> $@


$(example-doxygen):
	@echo ================== Assembling $@
	@echo "/**" > $@
	@echo " * @page $(shell echo $(call get_basename,$@) | $(PERL) -pi -e 's/-/_/g;') \
	                The $(call get_basename,$@) tutorial program" >> $@
	@echo "@htmlonly" >> $@
	@cat tmp/generated/$(call get_basename, $@)_toc.combined >> $@
	@echo "@endhtmlonly" >> $@
	@cat intro.dox \
		| $(PERL) $(create_anchors) >> $@
	@echo >> $@
	@cat tmp/generated/$(call get_basename, $@)_prog.dox >> $@
	@cat results.dox \
		| $(PERL) $(create_anchors) >> $@
	@echo >> $@
	@echo " <a name=\"PlainProg\"></a>" >> $@
	@echo " <h1> The plain program</h1>" >> $@
	@echo " @include \"$(call get_basename, $@).cc\"" >> $@
	@echo " */" >> $@

# Previous version of the lines after the last 'create_anchor'
#	@cat generated/$(call get_basename, $@)_plain.dox >> $@

validate-xrefs:
	@$(PERL) $D/common/scripts/validate-xrefs.pl \
                $(filter-out head.html foot.html, \
                             $(shell echo *.html))


clean:
	-rm -f  tmp/*.cc tmp/*.h $(example-toc) \
		$(example-toc-prog) $(example-toc-plain) \
		$(example-toc-intro) $(example-toc-results) \
		$(example-dox-prog) $(example-dox-plain) $(example-doxygen)
	-rm -f Makefile.dep
	-rm -rf html latex


.PHONY: validate-xrefs clean


Makefile.dep: $(shell echo ../step-feibm.cc *.dox) Makefile
	@echo ================== Generating $@
	@for i in $(example-names) ; do \
		echo tmp/generated/$${i}_prog.html tmp/generated/$${i}_plain.html \
		     tmp/generated/$${i}_toc.prog \
		     tmp/generated/$${i}_prog.dox tmp/generated/$${i}_plain.dox : \
			`echo ../step-feibm.cc` ; \
		echo tmp/generated/$${i}_toc.plain : tmp/generated/$${i}_toc.prog ; \
		echo tmp/generated/$${i}_toc.combined : \
			tmp/generated/$${i}_toc.intro \
			tmp/generated/$${i}_toc.results \
			tmp/generated/$${i}_toc.prog \
			tmp/generated/$${i}_toc.plain ; \
		echo tmp/$$i.cc : `echo ../step-feibm.cc` ; \
		echo tmp/$$i.h : tmp/generated/$${i}_prog.dox tmp/generated/$${i}_plain.dox \
				tmp/generated/$${i}_toc.combined \
				intro.dox \
				results.dox \
				tmp/$$i.cc ; \
		echo tmp/generated/$${i}_toc.intro : \
				intro.dox \
			 ; \
		echo tmp/generated/$${i}_toc.results : \
				results.dox \
			 ; \
	 done \
	> $@

ifneq ($(CLEAN),yes)
include Makefile.dep
endif
