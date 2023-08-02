# Contributing to H2O LLM STUDIO

H2O LLM Studio is an open source project released under the Apache Software Licence v2. Open Source projects live by
their user and developer communities. We welcome and encourage your contributions of any kind!

## Bug Reports and Feature Requests

Found a bug or have an idea for a new feature? Your feedback is invaluable! To ensure a smooth and collaborative
process, please follow these steps:

1. Provide the full error message and stack trace, if applicable.
2. Attach the model configuration yaml file if the error is related to model training.
3. Specify the commit hash of the version you are using (running `git rev-parse HEAD`) in your report. If you are
   pasting the UI error message, the commit hash will also be included in the error message.
4. If the error is reproducible, kindly include the steps to reproduce it.
5. If possible, attempt to reproduce the error using the default dataset.
6. Please mention any other details that might be useful, e.g. if you are using LLM Studio in a Docker container, etc.

## Pull Requests

You can contribute to the project by fixing bugs, adding new features, refactoring code, or enhancing documentation.
To ensure a smooth and collaborative process for everyone, please follow these guidelines:

1. Check if the issue you plan to address is already [reported](https://github.com/h2oai/h2o-llmstudio/issues). If not,
   please open a new issue
   to discuss your proposed changes.
2. Avoid duplicating work by commenting on the issue you're working on and feel free to seek assistance or ask
   questions; our team is happy to help.
3. Fork the repository and create a new branch from `main`. To develop, please follow the setup instructions below.
4. Implement your changes and commit them to your branch.
5. When you feel ready, open a pull request with your changes. You can also open the PR as a draft to receive early
   feedback. To facilitate the review process, we have provided a PR checklist below.
6. Our team will review your pull request and provide feedback. Once everything looks good, we will proceed to merge
   your contribution.

## Setting up your development environment

Follow the instructions in [README](https://github.com/h2oai/h2o-llmstudio/blob/main/README.md) to set up your
environment. Run `make setup-dev` instead of `make setup` to install the development dependencies.

## Running linters and tests

Before submitting your pull request, ensure that your code passes the linters and tests.
To format your code, run `make format`. You can check for any style issues by running `make style`. To run the tests,
run `make test`.

## PR checklist

Please make sure your pull request fulfills the following checklist:

☐ The PR title should provide a clear summary of your contribution.<br>
☐ Link the related issue (e.g., closes #123) in your PR description.<br>
☐ If your contribution is still a work in progress, change the PR to draft mode.<br>
☐ Ensure that the existing tests pass by running `make test`.<br>
☐ Make sure `make style` passes to maintain consistent code style.<br>
