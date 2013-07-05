.PHONY: jobs

DATE  = 2013-06-25
#DATE = $(shell date +%Y-%m-%d)
ALL_NORM_PRE_FILES = $(foreach idx, $(shell seq -w 001 100), jobs/all-norm-pre-$(idx).bash)
ALL_NORM_FILES = $(ALL_NORM_PRE_FILES) jobs/all-norm-pmc-monolithic.bash
EXCL_NORM_PRE_FILES = $(foreach idx, $(shell seq -w 001 100), jobs/excl-norm-pre-$(idx).bash)
EXCL_NORM_FILES = $(EXCL_NORM_PRE_FILES) jobs/excl-norm-pmc-monolithic.bash jobs/excl-norm-pmc-hc.bash
INCL_NORM_PRE_FILES = $(foreach idx, $(shell seq -w 001 100), jobs/incl-norm-pre-$(idx).bash)
INCL_NORM_FILES = $(INCL_NORM_PRE_FILES) jobs/incl-norm-pmc-monolithic.bash

all: jobs

jobs/all-norm-pre-%.bash : all-norm-pre.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    -e 's/@IDX@/$*/g' \
	    < $< \
	    > $@
jobs/all-norm-pmc-monolithic.bash : all-norm-pmc-monolithic.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    < $< \
	    > $@

jobs/excl-norm-pre-%.bash : excl-norm-pre.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    -e 's/@IDX@/$*/g' \
	    < $< \
	    > $@
jobs/excl-norm-pmc-monolithic.bash : excl-norm-pmc-monolithic.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    < $< \
	    > $@
jobs/excl-norm-pmc-hc.bash : excl-norm-pmc-hc.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    < $< \
	    > $@

jobs/incl-norm-pre-%.bash : incl-norm-pre.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    -e 's/@IDX@/$*/g' \
	    < $< \
	    > $@
jobs/incl-norm-pmc-monolithic.bash : incl-norm-pmc-monolithic.bash.in
	@sed \
	    -e 's/@DATE@/$(DATE)/g' \
	    < $< \
	    > $@

jobs: $(ALL_NORM_FILES) $(EXCL_NORM_FILES) $(INCL_NORM_FILES)
