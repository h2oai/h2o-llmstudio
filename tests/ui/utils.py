from hac_playwright.main import keycloak_login, okta_login, okta_otp_local
from playwright.sync_api import Page


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
