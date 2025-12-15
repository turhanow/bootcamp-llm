.PHONY: run venv install

PYTHON=.venv/bin/python

venv:
	/opt/homebrew/bin/python3.11 -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m app.main
