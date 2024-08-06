SHELL := /bin/bash

PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= $(PYTHON) -m pip
PIPENV ?= $(PYTHON) -m pipenv
PIPENV_PYTHON = $(PIPENV) run python
PIPENV_PIP = $(PIPENV_PYTHON) -m pip
PWD = $(shell pwd)
DOCKER_IMAGE ?= gcr.io/vorvan/h2oai/h2o-llmstudio:nightly
APP_VERSION=$(shell sed -n 's/^version = //p' pyproject.toml | tr -d '"')

ifeq ($(origin H2O_LLM_STUDIO_WORKDIR), environment)
    WORKDIR := $(H2O_LLM_STUDIO_WORKDIR)
else
    WORKDIR := $(shell pwd)
endif

ifeq ($(LOG_LEVEL), $(filter $(LOG_LEVEL), debug trace))
    PW_DEBUG = DEBUG=pw:api
else
    PW_DEBUG =
endif

PHONY: pipenv
pipenv:
	$(PIP) install pip==24.1.2
	$(PIP) install pipenv==2024.0.1

.PHONY: setup
setup: pipenv
	$(PIPENV) install --verbose --python $(PYTHON_VERSION)
	-$(PIPENV_PIP) install flash-attn==2.6.1 --no-build-isolation --upgrade --no-cache-dir

.PHONY: setup-dev
setup-dev: pipenv
	$(PIPENV) install --verbose --dev --python $(PYTHON_VERSION)
	-$(PIPENV_PIP) install flash-attn==2.6.1 --no-build-isolation --upgrade --no-cache-dir
	$(PIPENV) run playwright install

.PHONY: setup-no-flash
setup-no-flash: pipenv
	$(PIPENV) install --verbose --python $(PYTHON_VERSION)

.PHONY: setup-conda-nightly
setup-conda:
	@bash -c '\
		set -e; \
		source $$(conda info --base)/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda create -n llmstudio python=3.10 -y; \
		conda activate llmstudio; \
		conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y; \
		conda install pytorch pytorch-cuda=12.4 -c pytorch-nightly -c nvidia -y; \
		grep -v "nvidia" requirements.txt | grep -v "torch" | python -m pip install -r /dev/stdin; \
		python -m pip install flash-attn==2.6.1 --no-build-isolation --upgrade --no-cache-dir; \
	'

setup-ui: pipenv
	$(PIPENV) install --verbose --categories=dev-packages --python $(PYTHON_VERSION)
	$(PIPENV) run playwright install

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

	-$(PIPENV) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@if ! grep -Eq "Success: no issues found in [0-9]+ source files" reports/mypy.log ; then exit 1; fi

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
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); \
	$(PIPENV) run pytest -v --junitxml=reports/junit.xml \
	--import-mode importlib \
	--html=./reports/pytest.html \
	--cov=llm_studio \
	--cov-report term \
	--cov-report html:./reports/coverage.html \
    -o log_cli=true -o log_level=INFO -o log_file=reports/tests.log \
    tests/* 2>&1 | tee reports/tests.log'

# Use to quickly run a single test (e.g. make test-debug test=test_encode)
.PHONY: test-debug
test-debug: reports
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); \
	$(PIPENV) run pytest -v --junitxml=reports/junit.xml \
	--import-mode importlib \
	--html=./reports/pytest.html \
	-k $(test) \
	-s \
    -o log_cli=false -o log_level=WARNING -o log_file=/dev/null \
    tests/*'

# Only run the unit-tests (src)
.PHONY: test-unit
test-unit: reports
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); \
	$(PIPENV) run pytest -v --junitxml=reports/junit.xml \
	--import-mode importlib \
	--html=./reports/pytest.html \
	-k src \
	--cov=llm_studio/src \
	--cov-report term \
	--cov-report html:./reports/coverage.html \
    -o log_cli=true -o log_level=INFO -o log_file=reports/tests.log \
    tests/* 2>&1 | tee reports/tests.log'

.PHONY: test-ui
test-ui: reports setup-ui
	@bash -c 'set -o pipefail; \
	$(PW_DEBUG) $(PIPENV) run pytest \
	-v \
	--junitxml=reports/junit_ui.xml \
	--html=./reports/pytest_ui.html \
	-o log_cli=true \
	-o log_level=$(LOG_LEVEL) \
	-o log_file=reports/tests_ui.log \
	tests/ui/test.py 2>&1 | tee reports/tests_ui.log'

.PHONY: test-ui-headed
test-ui-headed: setup-ui
	$(PW_DEBUG) $(PIPENV) run pytest \
	-vvs \
	--headed \
	--video=on \
	--screenshot=on \
	--slowmo=100 \
	tests/ui/test.py 2>&1 | tee reports/tests.log

.PHONY: test-ui-github-actions  # Run UI tests in GitHub Actions. Starts the Wave server and runs the tests locally.
test-ui-github-actions: reports setup-ui
	@echo "Starting the server..."
	make llmstudio &
	@echo "Server started in background."
	@echo "Waiting 10s for the server to start..."
	sleep 10
	@echo "Running the tests..."
	LOCAL_LOGIN=True \
	PYTEST_BASE_URL=localhost:10101 \
	make test-ui
	@echo "Stopping the server..."
	make stop-llmstudio
	@echo "Server stopped."

.PHONY: wave
wave:
	nvidia-smi && \
	H2O_WAVE_APP_ACCESS_KEY_ID=dev \
	H2O_WAVE_APP_ACCESS_KEY_SECRET=dev \
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(PIPENV) run wave run app

.PHONY: llmstudio
llmstudio:
	nvidia-smi && \
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(PIPENV) run wave run --no-reload app

.PHONY: llmstudio-conda
llmstudio-conda:
	CONDA_ACTIVATE="source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate llmstudio" && \
	bash -c "$$CONDA_ACTIVATE && \
		nvidia-smi && \
		H2O_WAVE_MAX_REQUEST_SIZE=25MB \
		H2O_WAVE_NO_LOG=true \
		H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
		wave run --no-reload app"

.PHONY: stop-llmstudio
stop-llmstudio:
	@kill $$(lsof -ti :10101)

.PHONY: docker-build-nightly
docker-build-nightly:
	docker build -t $(DOCKER_IMAGE) .

# Run the Docker container with the nightly image
# Uses the local `llmstudio_mnt` directory as the mount point for the container
.PHONY: docker-run-nightly
docker-run-nightly:
ifeq (,$(wildcard ./llmstudio_mnt))
	mkdir llmstudio_mnt
endif
	docker run \
		--runtime=nvidia \
		--shm-size=64g \
		--init \
		--rm \
		--it \
		-u `id -u`:`id -g` \
		-p 10101:10101 \
		-v `pwd`/llmstudio_mnt:/home/llmstudio/mount \
		$(DOCKER_IMAGE)

# Perform a local Trivy scan for CVEs
# Get Trivy from https://aquasecurity.github.io/trivy/v0.53/getting-started/installation/
.PHONY: trivy-local
trivy-local: docker-build-nightly
	trivy image --scanners vuln --severity  CRITICAL,HIGH --timeout 60m $(DOCKER_IMAGE)

.PHONY: docker-clean-all
docker-clean-all:
	@CONTAINERS=$$(docker ps -a -q --filter ancestor=$(DOCKER_IMAGE)); \
	if [ -n "$$CONTAINERS" ]; then \
		docker stop $$CONTAINERS; \
		docker rm $$CONTAINERS; \
	fi
	docker rmi $(DOCKER_IMAGE)

.PHONY: bundles
bundles:
	rm -f -r bundles
	mkdir -p bundles
	cp -r static about.md bundles/
	sed 's/{{VERSION}}/${APP_VERSION}/g' app.toml.template > bundles/app.toml
	cd bundles && zip -r ai.h2o.llmstudio.${APP_VERSION}.wave *

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
