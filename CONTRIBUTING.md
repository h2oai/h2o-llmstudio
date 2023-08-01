# Contributing to H2O LLM STUDIO

Everyone is welcome to contribute, and we value everybody's contribution. You can fix bugs or new features, fix typos or
add missing documentation, We have a few guidelines to ensure that the process is as smooth as possible for everyone
involved.

1. Check if there is already [an open issue](https://github.com/h2oai/h2o-llmstudio/issues). If not, first open a new
   issue.
2. To avoid duplicated work, please comment on the issue that you are working on it. Also feel free to ask questions on
   the issue, we are happy to help.
3. Fork the repository and create a new branch from `main`. To develop, please follow the setup instructions below.
4. Once you feel ready, open a pull request with your changes. The PR can also be opened as a draft before the work
   is finished, so that you can get early feedback. We have a PR checklist below.
5. We will review your PR and provide feedback. If everything looks good, we will merge your PR.

## Setting up your development environment

We assume that you develop on Linux, unless you are working on a dedicated port to another OS.
Please follow the instructions in [README](https://github.com/h2oai/h2o-llmstudio/blob/main/README.md) to set up your
development environment. Run `make setup-dev` instead of `make setup` to install the development dependencies.

## Running linters and tests

Before creating a PR, please make sure that your code passes the linters and tests. To format your code,
run `make format`. You can check for any style issues by running `make style`. To run the tests, run `make test`.

## PR checklist
☐ The PR title should summarize your contribution.<br>
☐ Link the issue (e.g. closes #123) in your PR description.<br>
☐ To indicate a work in progress, please change the PR to draft mode.<br>
☐ Make sure existing tests pass by running `make test`.<br>
☐ Make sure `make style` pass.<br>
