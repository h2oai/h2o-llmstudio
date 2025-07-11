SHELL := /bin/bash

UV ?= $(HOME)/.local/bin/uv
UVX ?= $(HOME)/.local/bin/uvx
RUN ?= $(UV) run
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

.PHONY: uv
uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: setup
setup: uv  # Install dependencies
	$(UV) sync --frozen --no-dev
	-$(UV) sync --frozen --no-dev --extra flash

.PHONY: setup-dev
setup-dev: uv  # Install dependencies including dev dependencies
	$(UV) sync --frozen --group dev
	-$(UV) sync --frozen --group dev --extra flash 
	$(UV) run playwright install

.PHONY: requirements
requirements: uv  # uv pip compile requirements.txt
	$(UV) export --no-hashes --no-dev --no-header --frozen --format requirements-txt > requirements.txt
	grep -v "sys_platform == 'win32'" requirements.txt | awk -F ';' '{print $$1}' > tmp_requirements.txt && mv tmp_requirements.txt requirements.txt

clean-env:
	rm -rf .venv

clean-data:
	rm -rf data

clean-output:
	rm -rf output

reports:
	mkdir -p reports

.PHONY: style
style: reports uv
	@echo -n > reports/mypy_errors.log
	@echo -n > reports/mypy.log
	@echo

	-$(UV) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@if ! grep -Eq "Success: no issues found in [0-9]+ source files" reports/mypy.log ; then exit 1; fi

.PHONY: format
format:  # Format and check code with ruff
	uv tool run ruff format
	uv tool run ruff check --fix

.PHONY: format-check
format-check:  # Check format and check code with ruff (fails if changes are needed)
	$(UV) tool run ruff format --check
	$(UV) tool run ruff check

.PHONY: test
test: reports
	@bash -c 'set -o pipefail; export PYTHONPATH=$(PWD); \
	$(UV) run pytest -v --junitxml=reports/junit.xml \
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
	$(UV) run pytest -v --junitxml=reports/junit.xml \
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
	$(UV) run pytest -v --junitxml=reports/junit.xml \
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
	$(PW_DEBUG) $(UV) run pytest \
	-v \
	--junitxml=reports/junit_ui.xml \
	--html=./reports/pytest_ui.html \
	-o log_cli=true \
	-o log_level=$(LOG_LEVEL) \
	-o log_file=reports/tests_ui.log \
	tests/ui/test.py 2>&1 | tee reports/tests_ui.log'

.PHONY: test-ui-headed
test-ui-headed: setup-ui
	$(PW_DEBUG) $(UV) run pytest \
	-vvs \
	-s \
	--headed \
	--video=on \
	--screenshot=on \
	--slowmo=1000 \
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
	HF_HUB_DISABLE_TELEMETRY=1 \
	H2O_WAVE_APP_ACCESS_KEY_ID=dev \
	H2O_WAVE_APP_ACCESS_KEY_SECRET=dev \
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(UV) run wave run llm_studio.app

.PHONY: llmstudio
llmstudio:
	nvidia-smi && \
	HF_HUB_DISABLE_TELEMETRY=1 \
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=true \
	H2O_WAVE_PRIVATE_DIR="/download/@$(WORKDIR)/output/download" \
	$(UV) run wave run --no-reload llm_studio.app

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
		-it \
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
