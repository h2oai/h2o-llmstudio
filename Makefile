PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= $(PYTHON) -m pip
PIPENV ?= $(PYTHON) -m pipenv
PIPENV_PYTHON = $(PIPENV) run python
PIPENV_PIP = $(PIPENV_PYTHON) -m pip
PWD = $(shell pwd)

PHONY: pipenv
pipenv:
	$(PIP) install pip --upgrade
	$(PIP) install pipenv==2022.10.4

.PHONY: setup
setup: pipenv
	$(PIPENV) install --python $(PYTHON_VERSION)
	$(PIPENV_PIP) install deps/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl --force-reinstall

.PHONY: setup-dev
setup-dev: pipenv
	$(PIPENV) install --verbose --dev --python $(PYTHON_VERSION)
	$(PIPENV_PIP) install deps/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl --force-reinstall

clean-env:
	$(PIPENV) --rm

clean-data:
	rm -rf data

reports:
	mkdir -p reports

.PHONY: style
style: reports pipenv
	@echo -n > reports/style.log
	@echo -n > reports/style_errors.log
	@echo

	@echo "# flake8" >> reports/style.log
	-$(PIPENV) run flake8 | tee -a reports/style.log || echo "flake8 failed" >> reports/style_errors.log
	@echo

	@echo "" >> reports/style.log
	@echo "# mypy" >> reports/style.log
	-$(PIPENV) run mypy . | tee -a reports/style.log || echo "mypy failed" >> reports/style_errors.log
	@echo

	@if [ -s reports/style_errors.log ]; then exit 1; fi

.PHONY: format
format: pipenv
	$(PIPENV) run isort .
	$(PIPENV) run black .

.PHONY: isort
isort: pipenv
	$(PIPENV) run isort .

.PHONY: black
black: pipenv
	$(PIPENV) run black .

.PHONY: test
test: reports
	export PYTHONPATH=$(shell pwd) && $(PIPENV) run pytest -v -s -x \
		--junitxml=./reports/junit.xml \
		tests/* | tee reports/pytest.log

.PHONY: wave
wave:
	(export H2O_WAVE_MAX_REQUEST_SIZE=25MB && \
	 export H2O_WAVE_NO_LOG=True && \
	 export H2O_WAVE_PRIVATE_DIR="/download/@$(PWD)/output/download" && \
	 echo ----- ENVIRONMENT ----- && \
	 env && \
	 echo ----- PWD ----- && \
	 pwd && \
	 echo ---------- && \
	 $(PIPENV) run wave run app)

.PHONY: wave-no-reload
wave-no-reload:
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=True \
	H2O_WAVE_PRIVATE_DIR="/download/@$(PWD)/output/download" \
	$(PIPENV) run wave run --no-reload app

.PHONY: shell
shell:
	$(PIPENV) shell
