import logging
from playwright.sync_api import Page
from pytest_bdd import given, scenarios, then, when

from tests.ui.utils import LLMStudioPage, login, handle_terms_and_conditions_page

scenarios("llm_studio.feature")


@given("LLM Studio home page is opened")
def open_llm_studio(page: Page, app_address: str):
    page.goto(app_address)
    handle_terms_and_conditions_page(page)


@when("I login to LLM Studio", target_fixture="llm_studio")
def login_to_llm_studio(logger: logging.Logger, page: Page, app_address: str):
    login(page, "okta", "H2O-Tester", "ManagedCloud5")
    return LLMStudioPage(logger, page, app_address)


@then("I should see the datasets")
def view_datasets(llm_studio: LLMStudioPage):
    llm_studio.view_datasets()
    llm_studio.assert_dataset_import("train_full")


@then("I upload dataset using filesystem")
def upload_dataset_using_filesystem(llm_studio: LLMStudioPage):
    llm_studio.import_dataset_from_filesystem(
        "/home/llmstudio/mount/data/user/oasst/train_full.pq"
    )
