# UI Testing for LLM-Studio
## Overview
The UI testing framework for LLM-Studio leverages the principles of Behaviour Driven Development (BDD), combining the power of Playwright for automation and Behave for writing UI tests. This approach offers the best of both worlds, as it makes the tests easily understandable for humans while remaining straightforward for machines to execute. By adopting this framework, it eliminates the complexities associated with using Selenium.

## Design
The framework is designed to be flexible, capable of running on local and remote machines seamlessly. It is agnostic to the location of the application, making it ideal for release testing across various instances of H2OAI Integrated Cloud (HAIC).

### Local Machine Setup
To set up and run UI tests locally, follow these steps:

```bash
export LOCAL_LOGIN=True
export PYTEST_BASE_URL=localhost:10101
make setup-dev
make llmstudio
make setup-ui
make test-ui-headed
```

### Remote Testing
You can conduct UI testing for LLM-Studio on a remote machine using the following approaches:

1. Running the App on a Remote Server
  - Set up the app on a remote Ubuntu instance:
    ```bash
    make setup-dev
    make llmstudio
    ```
  - Obtain the app URL.
  - Run the tests on the local machine:
    ```bash
    export PYTEST_BASE_URL=localhost:10101
    make setup-ui
    make test-ui-headed
    ```
2. Running the App on HAMC (with Okta Login)
```bash
export OKTA_USER=
export OKTA_PASSWORD=
export PYTEST_BASE_URL=
make test-ui
```

3. Running the App on HAIC (with Keycloak Login)
```bash
export KEYCLOAK_USER=
export KEYCLOAK_PASSWORD=
export PYTEST_BASE_URL=
make test-ui
```

### Test Results
The results of the UI tests are stored in `reports/junit_ui.xml`. These reports provide valuable insights into the success and failure of the tests, aiding in the continuous improvement of the application.
