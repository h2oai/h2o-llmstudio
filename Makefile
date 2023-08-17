PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= $(PYTHON) -m pip
PIPENV ?= $(PYTHON) -m pipenv
PIPENV_PYTHON = $(PIPENV) run python
PIPENV_PIP = $(PIPENV_PYTHON) -m pip
PWD = $(shell pwd)

ifeq ($(origin H2O_LLM_STUDIO_WORKDIR), environment)
    WORKDIR := $(H2O_LLM_STUDIO_WORKDIR)
else
    WORKDIR := $(shell pwd)
endif

PHONY: pipenv
pipenv:
	$(PIP) install pip --upgrade
	$(PIP) install pipenv==2022.10.4

.PHONY: setup
setup: pipenv
	$(PIPENV) install --verbose --python $(PYTHON_VERSION)

.PHONY: setup-dev
setup-dev: pipenv
	$(PIPENV) install --verbose --dev --python $(PYTHON_VERSION)

.PHONY: export-requirements
export-requirements: pipenv
	$(PIPENV) requirements > requirements.txt

clean-env:
	$(PIPENV) --rm

clean-data:
	rm -rf data

clean-output:
	rm -rf output

reports:
	mkdir -p reports

.PHONY: style
style: reports pipenv
	@echo -n > reports/flake8_errors.log
	@echo -n > reports/mypy_errors.log
	@echo -n > reports/mypy.log
	@echo

	-$(PIPENV) run flake8 | tee -a reports/flake8_errors.log
	@if [ -s reports/flake8_errors.log ]; then exit 1; fi

	-$(PIPENV) run mypy . --check-untyped-defs | tee -a reports/mypy.log || echo "mypy failed" >> reports/mypy_errors.log
	@if [ -s reports/mypy_errors.log ]; then exit 1; fi

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
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); $(PIPENV) run pytest -v -s --junitxml=reports/junit.xml \
    -o log_cli=true -o log_level=INFO -o log_file=reports/tests.log \
    tests/* 2>&1 | tee reports/tests.log'

.PHONY: wave
wave:
	HF_HUB_ENABLE_HF_TRANSFER=True \
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(PIPENV) run wave run app

.PHONY: llmstudio
llmstudio:
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(PIPENV) run wave run --no-reload app

.PHONY: docker-build-nightly
docker-build-nightly:
	docker build -t gcr.io/vorvan/h2oai/h2o-llmstudio:nightly .

.PHONY: docker-run-nightly
docker-run-nightly:
ifeq (,$(wildcard ./data))
	mkdir data
endif
ifeq (,$(wildcard ./output))
	mkdir output
endif
	docker run \
		--runtime=nvidia \
		--shm-size=64g \
		--init \
		--rm \
		-u `id -u`:`id -g` \
		-p 10101:10101 \
		-v `pwd`/data:/workspace/data \
		-v `pwd`/output:/workspace/output \
		gcr.io/vorvan/h2oai/h2o-llmstudio:nightly

.PHONY: shell
shell:
	$(PIPENV) shell

setup-doc:  # Install documentation dependencies
	cd documentation && npm install 

run-doc:  # Run the doc locally
	cd documentation && npm start

update-documentation-infrastructure:
	cd documentation && npm update @h2oai/makersaurus
	cd documentation && npm ls

build-doc-locally:  # Bundles your website into static files for production 
	cd documentation && npm run build

serve-doc-locally:  # Serves the built website locally 
	cd documentation && npm run serve
