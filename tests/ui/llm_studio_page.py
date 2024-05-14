import os

from hac_playwright.pages.base import BasePage
from playwright.sync_api import expect

CLOUD_FILESYSTEM_PATH = "/home/llmstudio/mount/data/user/oasst"
LOCAL_FILESYSTEM_PATH = os.path.join(os.getcwd(), "data/user/oasst")


class LLMStudioPage(BasePage):
    # Constants for selectors
    DATASET_IMPORT_SOURCE_SELECTOR = "dataset/import/source"
    CONTINUE_BUTTON_SELECTOR = "button[name='Continue']"
    DATASET_LIST_DELETE_SELECTOR = "dataset/list/delete"
    DATASET_DELETE_DIALOG_SELECTOR = "dataset/delete/dialog"
    DATASET_DELETE_SELECTOR = "dataset/delete"
    EXPERIMENT_RUN_SELECTOR = "experiment/start/run"
    EXPERIMENT_NAME_SELECTOR = "experiment/start/cfg/experiment_name"
    EXPERIMENT_LIST_DELETE_SELECTOR = "experiment/list/delete"
    EXPERIMENT_DELETE_DIALOG_SELECTOR = "experiment/delete/dialog"
    EXPERIMENT_DELETE_SELECTOR = "experiment/delete"
    EXPERIMENT_STATUS_SELECTOR = "[data-automation-key='status']"
    EXPERIMENT_INDEX_SELECTOR = "[data-automation-key='name']"
    FILESYSTEM_SELECTOR = "dataset/import/local_path"
    FILENAME_SELECTOR = "dataset/import/name"
    S3_BUCKET_SELECTOR = "dataset/import/s3_bucket"
    S3_ACCESS_KEY_SELECTOR = "dataset/import/s3_access_key"
    S3_SECRET_KEY_SELECTOR = "dataset/import/s3_secret_key"
    S3_FILENAME_SELECTOR = "dataset/import/s3_filename"
    AZURE_CONN_STRING = "dataset/import/azure_conn_string"
    AZURE_CONTAINER = "dataset/import/azure_container"
    AZURE_FILENAME = "dataset/import/azure_filename"
    KAGGLE_COMMAND = "dataset/import/kaggle_command"
    KAGGLE_USERNAME = "dataset/import/kaggle_username"
    KAGGLE_SECRET_KEY = "dataset/import/kaggle_secret_key"
    DATA_SAMPLING = "experiment/start/cfg/data_sample"
    MAX_LENGTH_PROMPT = "experiment/start/cfg/max_length_prompt"
    MAX_LENGTH_ANSWER = "experiment/start/cfg/max_length_answer"
    MAX_LENGTH = "experiment/start/cfg/max_length"
    MAX_LENGTH_INFERENCE = "experiment/start/cfg/max_length_inference"
    MIXED_PRECISION = "experiment/start/cfg/mixed_precision"
    EXPERIMENT_REFRESH_SELECTOR = "experiment/list/refresh"
    GPU_WARNING_SELECTOR = "experiment/start/error/proceed"

    def assert_dataset_import(self, dataset_name: str):
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element is not None and clickable
        assert dataset is not None
        dataset.click()

    def get_by_test_id(self, test_id):
        selector = f'[data-test="{test_id}"]'
        return self.page.locator(selector)

    def open_home_page(self):
        self.page.get_by_role("button", name="Home").click()

    def open_app_settings(self):
        self.page.get_by_role("button", name="Settings").click()

    def dataset_name(self, filename):
        self.get_by_test_id(self.FILENAME_SELECTOR).fill(filename)
        self.continue_button().click()
        self.continue_button().click()

    def import_dataset_from_filesystem(self, filepath: str):
        self.import_dataset("Local")
        if "LOCAL_LOGIN" in os.environ:
            path = f"{LOCAL_FILESYSTEM_PATH}/{filepath}"
        else:
            path = f"{CLOUD_FILESYSTEM_PATH}/{filepath}"
        self.get_by_test_id(self.FILESYSTEM_SELECTOR).fill(path)
        self.continue_button().click()

    def continue_button(self):
        return self.page.get_by_role("button", name="Continue")

    def import_dataset(self, source: str):
        button = self.page.get_by_role("button", name="Import dataset")
        button.click()
        # FIX: Selectors.set_test_id_attribute(self, "data-test")
        dropdown = self.get_by_test_id(self.DATASET_IMPORT_SOURCE_SELECTOR)
        dropdown.click()
        self.page.get_by_role("option", name=source).click()

    def import_dataset_from_aws(
        self, bucket: str, access_key: str, secret_key: str, dataset_name: str
    ):
        self.import_dataset("AWS S3")
        self.get_by_test_id(self.S3_BUCKET_SELECTOR).fill(bucket)
        self.get_by_test_id(self.S3_ACCESS_KEY_SELECTOR).fill(access_key)
        self.get_by_test_id(self.S3_SECRET_KEY_SELECTOR).fill(secret_key)
        self.get_by_test_id(self.S3_FILENAME_SELECTOR).fill(dataset_name)
        self.continue_button().click()

    def import_dataset_from_azure(
        self, connection: str, container: str, dataset_name: str
    ):
        self.import_dataset("Azure Blob Storage")
        self.get_by_test_id(self.AZURE_CONN_STRING).fill(connection)
        self.get_by_test_id(self.AZURE_CONTAINER).fill(container)
        self.get_by_test_id(self.AZURE_FILENAME).fill(dataset_name)
        self.continue_button().click()

    def import_dataset_from_kaggle(
        self, kaggle_command: str, username: str, secret: str
    ):
        self.import_dataset("Kaggle")
        self.get_by_test_id(self.KAGGLE_COMMAND).fill(kaggle_command)
        self.get_by_test_id(self.KAGGLE_USERNAME).fill(username)
        self.get_by_test_id(self.KAGGLE_SECRET_KEY).fill(secret)
        self.continue_button().click()

    def delete_dataset(self, dataset_name: str):
        # Go to dataset page
        self.view_datasets()
        self.get_by_test_id(self.DATASET_LIST_DELETE_SELECTOR).click()
        # Locate dataset to delete
        self.page.get_by_role("gridcell", name=dataset_name).click()
        # Confirm dataset deletion
        self.get_by_test_id(self.DATASET_DELETE_DIALOG_SELECTOR).click()
        # Delete dataset
        self.get_by_test_id(self.DATASET_DELETE_SELECTOR).click()

    def view_datasets(self):
        self.page.get_by_role("button", name="View datasets").click()

    def assert_dataset_deletion(self, dataset_name: str):
        self.view_datasets()
        dataset = self.page.get_by_role("button", name=dataset_name)
        # Assert that the element not found
        expect(dataset).not_to_be_visible()

    def create_experiment(self, name: str):
        self.page.get_by_role("button", name="Create experiment").click()
        self.experiment_name(name)

    def slider(self, slider_selector, target_value: str):
        is_completed = False
        i = 0.0
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
            value_now = slider.get_attribute("aria-valuenow")

            if value_now == target_value:
                is_completed = True
            else:
                # Move the slider a little bit (adjust the step as needed)
                step = 0.1  # Adjust this value based on your requirements
                x1 = x2
                i += step

    def run_experiment(self):
        self.get_by_test_id(self.EXPERIMENT_RUN_SELECTOR).click()
        locator = self.get_by_test_id(self.GPU_WARNING_SELECTOR)
        if locator.is_visible():
            locator.click()

    def experiment_name(self, name: str):
        self.get_by_test_id(self.EXPERIMENT_NAME_SELECTOR).fill(name)

    def llm_backbone(self, value: str):
        self.page.get_by_role("combobox", name="LLM Backbone").fill(value)

    def mixed_precision(self, value: bool):
        old_toggle_value = self.get_by_test_id(self.MIXED_PRECISION).get_attribute(
            "aria-checked"
        )
        assert old_toggle_value in ["true", "false"]
        assert value in ["true", "false"]

        if old_toggle_value != value:
            self.get_by_test_id(self.MIXED_PRECISION).click()

    def data_sample(self, value):
        self.slider(self.DATA_SAMPLING, value)

    def max_length_prompt(self, value):
        self.slider(self.MAX_LENGTH_PROMPT, value)

    def max_length_answer(self, value):
        self.slider(self.MAX_LENGTH_ANSWER, value)

    def max_length(self, value):
        self.slider(self.MAX_LENGTH, value)

    def max_length_inference(self, value):
        self.slider(self.MAX_LENGTH_INFERENCE, value)

    def view_experiment_page(self):
        self.page.get_by_role("button", name="View experiments").click()

    def view_experiment(self, experiment_name: str):
        self.view_experiment_page()
        i = self.find_experiment_index(experiment_name)
        status = self.page.locator(
            f"{self.EXPERIMENT_STATUS_SELECTOR} >> nth={i}"
        ).inner_text()
        self.page.reload()
        while True:
            if status in ["queued", "running"]:
                self.page.reload()
                self.view_experiment_page()
                status = self.page.locator(
                    f"{self.EXPERIMENT_STATUS_SELECTOR} >> nth={i}"
                ).inner_text()
            elif status == "finished":
                break

    def find_experiment_index(self, experiment_name):
        index = 0
        while index < 100:  # number of experiments
            # Get the innerText of the element with the specified selector
            inner_text = self.page.locator(
                f"{self.EXPERIMENT_INDEX_SELECTOR} >> nth={index}"
            ).inner_text()
            # Check if the current name matches the target name
            if inner_text != experiment_name:
                index += 1
            else:
                break
        return index

    def delete_experiment(self, experiment_name: str):
        # Go to experiment page
        self.view_experiment_page()
        # Click on Delete experiments button
        self.get_by_test_id(self.EXPERIMENT_LIST_DELETE_SELECTOR).click()
        # Locate experiment to delete
        self.page.get_by_role("gridcell", name=experiment_name).locator(
            f'div:has-text("{experiment_name}")'
        ).first.click()
        # Delete experiment
        self.get_by_test_id(self.EXPERIMENT_DELETE_DIALOG_SELECTOR).click()
        # Confirm experiment deletion
        self.get_by_test_id(self.EXPERIMENT_DELETE_SELECTOR).click()

    def assert_experiment_deletion(self, experiment_name: str):
        # Go to experiment page
        self.view_experiment_page()
        experiment = self.page.get_by_role("button", name=experiment_name)
        # Assert that the element not found
        expect(experiment).not_to_be_visible()
