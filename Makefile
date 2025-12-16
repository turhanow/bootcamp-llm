.PHONY: venv install run start

PYTHON_BIN ?= python3.11
PYTHON := .venv/bin/python

venv:
	test -d .venv || $(PYTHON_BIN) -m venv .venv

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m app.main

start: install run
