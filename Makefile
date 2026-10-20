# North Caspian pipeline. Run inside the environment, e.g.
#   micromamba run -n caspian make verify-0
# `make all` rebuilds every result, figure and table from raw inputs.

PYTHON  ?= python
MAMBA   ?= micromamba
ENVNAME ?= caspian

.PHONY: help env lock test dahiti-download dahiti-qc study-area verify-0 verify-A verify-B all

help:
	@echo "targets: env lock test dahiti-qc verify-0 verify-A verify-B all"

## environment -------------------------------------------------------------
env:
	$(MAMBA) create -y -n $(ENVNAME) -f requirements.lock
	$(MAMBA) run -n $(ENVNAME) python -m pip install --no-deps --no-build-isolation -e .

lock:
	$(PYTHON) tools/pin_env.py
	$(MAMBA) env export -n $(ENVNAME) --explicit --md5 > requirements.lock

## checks ------------------------------------------------------------------
test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

verify-0:
	$(PYTHON) tools/verify.py 0

verify-A:
	$(PYTHON) tools/verify.py A

verify-B:
	$(PYTHON) tools/verify.py B

## Phase 0 -----------------------------------------------------------------
dahiti-download:
	$(PYTHON) scripts/p0_dahiti_download.py

dahiti-qc:
	$(PYTHON) scripts/p0_dahiti_qc.py

## full rebuild (extended as phases add targets) -----------------------------
study-area:
	$(PYTHON) scripts/a0_study_area.py

all: dahiti-qc study-area
