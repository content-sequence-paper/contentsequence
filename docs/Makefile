SPHINXOPTS ?= -W --keep-going
SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)
