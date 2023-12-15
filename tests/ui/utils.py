import os

from hac_playwright.main import keycloak_login, okta_login, okta_otp_local
from hac_playwright.pages.base import BasePage
from playwright.sync_api import Page, expect, Selectors
from playwright._impl._map import Map
from playwright._impl._playwright import Playwright as PlaywrightImpl
import playwright._impl as obj


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


class LLMStudioPage(BasePage, Map, Selectors):
    page_name = "LLM Studio Page"

    def view_datasets(self):
        self.page.get_by_role("button", name="View datasets").click()

    def assert_dataset_import(self, dataset_name: str):
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element is not None and visible
        assert dataset is not None
        assert dataset.is_visible()

    def get_by_test_id(self, test_id):
        selector = f'[data-test="{test_id}"]'
        return self.page.locator(selector)

    def open_home_page(self):
        self.page.get_by_role("button", name="Home").click()

    def open_app_settings(self):
        self.page.get_by_role("button", name="Settings").click()

    def import_dataset_from_filesystem(self, path: str, filename: str):
        self.import_dataset("Local")
        self.get_by_test_id("dataset/import/local_path").fill(path)
        self.continue_button().click()
        # Dataset configuration
        self.get_by_test_id("dataset/import/name").fill(filename)
        self.get_by_test_id("dataset/import/4").click()
        # Data validity check
        self.get_by_test_id("dataset/import/6").click()

    def continue_button(self):
        return self.page.get_by_role("button", name="Continue")

    def import_dataset(self, source: str):
        button = self.page.get_by_role("button", name="Import dataset")
        button.click()
        # FIX: Selectors.set_test_id_attribute(self, "data-test")
        dropdown = self.get_by_test_id("dataset/import/source")
        dropdown.click()
        self.page.get_by_role("option", name=source).click()

    def import_dataset_from_aws(
        self, bucket: str, access_key: str, secret_key: str, dataset_name: str
    ):
        self.import_dataset("AWS S3")
        self.get_by_test_id("dataset/import/s3_bucket").fill(bucket)
        self.get_by_test_id("dataset/import/s3_access_key").fill(access_key)
        self.get_by_test_id("dataset/import/s3_secret_key").fill(secret_key)
        self.get_by_test_id("dataset/import/s3_filename").fill(dataset_name)
        self.continue_button().click()

    def import_dataset_from_azure(
        self, connection: str, container: str, dataset_name: str
    ):
        self.import_dataset("Azure Blob Storage")
        self.get_by_test_id("dataset/import/azure_conn_string").fill(connection)
        self.get_by_test_id("dataset/import/azure_container").fill(container)
        self.get_by_test_id("dataset/import/azure_filename").fill(dataset_name)
        self.continue_button().click()

    def import_dataset_from_kaggle(
        self, kaggle_command: str, username: str, secret: str
    ):
        self.import_dataset("Kaggle")
        self.get_by_test_id("dataset/import/kaggle_command").fill(kaggle_command)
        self.get_by_test_id("dataset/import/kaggle_access_key").fill(username)
        self.get_by_test_id("dataset/import/kaggle_secret_key").fill(secret)
        self.continue_button().click()

    def delete_dataset(self, dataset_name: str):
        # Go to dataset page
        self.view_datasets()
        self.get_by_test_id("dataset/list/delete").click()
        # Locate dataset to delete
        self.page.get_by_role("gridcell", name=dataset_name).click()
        # Confirm dataset deletion
        self.get_by_test_id("dataset/delete/dialog").click()
        # Delete dataset
        self.get_by_test_id("dataset/delete").click()

    def view_datasets(self):
        locator = self.page.get_by_role("button", name="View datasets")
        locator.click()

    def assert_dataset_deletion(self, dataset_name: str):
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element not found
        expect(dataset).not_to_be_visible()

    def create_experiment(self, name: str):
        # Get locator element
        locator = self.page.get_by_role("button", name="Create experiment")
        locator.click()
        self.experiment_name(name)
        self.llm_backbone("MaxJeblick/llama2-0b-unit-test")
        self.data_sample(0.01)
        self.max_length_prompt(32)
        self.max_length_answer(32)
        self.max_length(32)
        self.max_length_inference(32)
        self.run_experiment()

    def slider(self, slider_selector, target_value: float):
        is_completed = False
        i = 0
        # Get the slider element
        slider = self.get_by_test_id(slider_selector)
        slider.click()
        # Get the bounding box of the slider
        bounding_box = slider.bounding_box()
        x1 = bounding_box["x"]
        y = bounding_box["y"] + bounding_box["height"] / 2

        while not is_completed:
            self.page.mouse.move(x1, y)
            self.page.mouse.down()
            x2 = bounding_box["x"] + bounding_box["width"] * float(i) / 100
            self.page.mouse.move(x2, y)
            self.page.mouse.up()
            value_now = float(slider.get_attribute("aria-valuenow"))

            if value_now == target_value:
                is_completed = True
            else:
                # Move the slider a little bit (adjust the step as needed)
                step = 0.1  # Adjust this value based on your requirements
                x1 = x2
                i += step

    def run_experiment(self):
        locator = self.get_by_test_id("experiment/start/run")
        locator.click()

    def experiment_name(self, name: str):
        locator = self.get_by_test_id("experiment/start/cfg/experiment_name")
        locator.fill(name)

    def llm_backbone(self, value: str):
        locator = self.page.get_by_role("combobox", name="LLM Backbone")
        locator.fill(value)

    def data_sample(self, value):
        selector = "experiment/start/cfg/data_sample"
        self.slider(selector, value)

    def max_length_prompt(self, value):
        selector = "experiment/start/cfg/max_length_prompt"
        self.slider(selector, value)

    def max_length_answer(self, value):
        selector = "experiment/start/cfg/max_length_answer"
        self.slider(selector, value)

    def max_length(self, value):
        selector = "experiment/start/cfg/max_length"
        self.slider(selector, value)

    def max_length_inference(self, value):
        selector = "experiment/start/cfg/max_length_inference"
        self.slider(selector, value)

    def view_experiment(self, experiment_name: str):
        locator = self.page.get_by_role("button", name="View experiments")
        locator.click()
        i = self.find_experiment_index(experiment_name)
        status = self.page.locator(
            f'[data-automation-key="status"] >> nth={i}'
        ).inner_text()

        while True:
            if status in ["queued", "running"]:
                self.page.wait_for_timeout(3000)
                self.page.reload()
                locator.click()
                status = self.page.locator(
                    f'[data-automation-key="status"] >> nth={i}'
                ).inner_text()
                print(f"status={status}")
            elif status == "finished":
                break

    def find_experiment_index(self, experiment_name):
        i = 0
        while True:
            # Get the innerText of the element with the specified selector
            inner_text = self.page.locator(
                f'[data-automation-key="name"] >> nth={i}'
            ).inner_text()
            # Check if the current name matches the target name
            if inner_text != experiment_name:
                i += 1
            else:
                break
        return i
