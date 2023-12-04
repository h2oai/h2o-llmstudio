from hac_playwright.main import keycloak_login, okta_login, okta_otp_local
from hac_playwright.pages.base import BasePage
from playwright.sync_api import Page, expect
import os


def login(
    page: Page,
    login_type: str,
    username: str,
    password: str,
    secret: str = "",
):
    if login_type == "keycloak":
        keycloak_login(page, username, password)
    elif login_type == "okta":
        okta_login(page, username, password)
    elif login_type == "okta-otp-local":
        okta_otp_local(page, username, password, secret)
    else:
        raise ValueError(f"Unknown login type '{login_type}'")


def handle_terms_and_conditions_page(page: Page):
    terms_and_conditions = page.get_by_role("heading", name="EULA")

    if terms_and_conditions.is_visible():
        # If the heading is present, click the "I agree" button
        page.get_by_role("button", name="I agree").click()
    else:
        return 1


class LLMStudioPage(BasePage):
    page_name = "LLM Studio Page"

    def view_datasets(self):
        self.page.get_by_role("button", name="View datasets").click()

    def assert_dataset_import(self, dataset_name: str):
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element is not None (found)
        assert (
            dataset is not None
        ), f"Element with dataset name '{dataset_name}' not found"

        # Assert that the element is visible
        assert (
            dataset.is_visible()
        ), f"Element with dataset name '{dataset_name}' is not visible"

    def open_home_page(self):
        self.page.get_by_role("button", name="Home").click()

    def open_app_settings(self):
        self.page.get_by_role("button", name="Settings").click()

    def import_dataset_from_filesystem(self, path: str, filename: str):
        self.page.get_by_role("button", name="Import dataset").click()
        self.page.locator('[data-test="dataset\\/import\\/source"]').get_by_text(
            "Upload"
        ).click()
        self.page.get_by_role("option", name="Local").click()
        self.page.locator('[data-test="dataset\\/import\\/local_path"]').fill(path)
        self.page.locator('[data-test="dataset\\/import\\/2"]').click()

        # Dataset configuration
        self.page.locator('[data-test="dataset\\/import\\/name"]').fill(filename)
        self.page.locator('[data-test="dataset\\/import\\/4"]').click()

        # Data Validity check
        self.page.locator('[data-test="dataset\\/import\\/6"]').click()
        # self.page.get_by_role("button", name="Continue").click()

    def import_dataset_from_aws(
        self, bucket: str, access_key: str, secret_key: str, dataset_name: str
    ):
        self.page.get_by_role("button", name="Import dataset").click()
        self.page.locator('[data-test="dataset\\/import\\/source"]').get_by_text(
            "AWS S3"
        ).click()
        self.page.locator('[data-test="dataset\\/import\\/s3_bucket"]').fill(bucket)
        self.page.locator('[data-test="dataset\\/import\\/s3_access_key"]').fill(
            access_key
        )
        self.page.locator('[data-test="dataset\\/import\\/s3_secret_key"]').fill(
            secret_key
        )
        self.page.locator('[data-test="dataset\\/import\\/s3_filename"]').fill(
            dataset_name
        )
        self.page.get_by_role("button", name="Continue").click()

    def import_dataset_from_azure(
        self, connection: str, container: str, dataset_name: str
    ):
        self.page.get_by_role("button", name="Import dataset").click()
        self.page.locator('[data-test="dataset\\/import\\/source"]').get_by_text(
            "Azure Datalake"
        ).click()
        self.page.locator('[data-test="dataset\\/import\\/azure_conn_string"]').fill(
            connection
        )
        self.page.locator('[data-test="dataset\\/import\\/azure_container"]').fill(
            container
        )
        self.page.locator('[data-test="dataset\\/import\\/azure_filename"]').fill(
            dataset_name
        )
        self.page.get_by_role("button", name="Continue").click()

    def import_dataset_from_kaggle(
        self, kaggle_command: str, username: str, secret: str
    ):
        self.page.get_by_role("button", name="Import dataset").click()
        self.page.locator('[data-test="dataset\\/import\\/source"]').get_by_text(
            "Kaggle"
        ).click()
        self.page.locator('[data-test="dataset\\/import\\/kaggle_command"]').fill(
            kaggle_command
        )
        self.page.locator('[data-test="dataset\\/import\\/kaggle_access_key"]').fill(
            username
        )
        self.page.locator('[data-test="dataset\\/import\\/kaggle_secret_key"]').fill(
            secret
        )
        self.page.get_by_role("button", name="Continue").click()

    def delete_dataset(self, dataset_name: str):
        self.page.get_by_role("button", name="View datasets").click()
        self.page.locator('[data-test="dataset\\/list\\/delete"]').click()
        self.page.get_by_role("gridcell", name=dataset_name).click()
        self.page.locator('[data-test="dataset\\/delete\\/dialog"]').click()
        self.page.locator('[data-test="dataset\\/delete"]').click()

    def assert_dataset_deletion(self, dataset_name: str):
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element not found
        expect(dataset).not_to_be_visible()
