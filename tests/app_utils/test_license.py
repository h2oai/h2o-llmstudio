import base64
import datetime
from unittest import mock

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from llm_studio.app_utils.license import _LM


def get_temp_license(private_key, desired_date):
    license_text = f"license_version:1\n\
serial_number:0\n\
licensee_organization:no_org\n\
licensee_email:noreply@example.org\n\
licensee_user_id:0\n\
is_h2o_internal_use:true\n\
created_by_email:nobody@example.org\n\
creation_date:{desired_date}\n\
product:h2o-autodoc\n\
license_type:developer\n\
expiration_date:{desired_date}".encode('utf-8')
    signature = private_key.sign(
        license_text,
        padding.PKCS1v15(),
        hashes.SHA1()
    )
    assert len(signature) == 256, "Signature length should be 256"
    signed_message = signature + license_text
    encoded_signed_message = base64.urlsafe_b64encode(signed_message)
    return encoded_signed_message


@pytest.fixture(scope="module")
def private_key():
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )


@pytest.mark.license
@mock.patch("cryptography.hazmat.primitives.serialization.load_pem_public_key")
def test_verify_signature_ok(load_pem_mock, private_key):
    # Arrange
    load_pem_mock.return_value = private_key.public_key()
    message = b"license_text"
    signature = private_key.sign(
        message,
        padding.PKCS1v15(),
        hashes.SHA1()
    )
    signed_message = signature + message
    encoded_signed_message = base64.urlsafe_b64encode(signed_message)
    license_manager = _LM(lt=encoded_signed_message)

    # Act & Assert - no exception raised
    license_manager._verify(message, signature)


@pytest.mark.license
@mock.patch("cryptography.hazmat.primitives.serialization.load_pem_public_key")
def test_verify_signature_unsigned_license(load_pem_mock, private_key):
    # Arrange
    load_pem_mock.return_value = private_key.public_key()
    message = b"license_text"
    signature = private_key.sign(
        message,
        padding.PKCS1v15(),
        hashes.SHA1()
    )
    signed_message = signature + message
    encoded_signed_message = base64.urlsafe_b64encode(signed_message)
    license_manager = _LM(lt=encoded_signed_message)

    # Act & Assert - Exception raised because of invalid signature
    with pytest.raises(ValueError):
        license_manager._verify(b"other_than_license_text", signature)


@pytest.mark.license
@mock.patch("cryptography.hazmat.primitives.serialization.load_pem_public_key")
def test_verify_license_expiry_date_is_old(load_pem_mock, private_key):
    # Arrange
    yesterdays_date = datetime.datetime.strftime(
        datetime.datetime.now() - datetime.timedelta(days=1),
        '%Y/%m/%d'
    )
    load_pem_mock.return_value = private_key.public_key()
    license_manager = _LM()

    # Act & Assert - Exception raised because of invalid signature
    ok, message, license_properties, license_key = \
        license_manager._validate_license_key(
            get_temp_license(private_key, yesterdays_date)
        )
    assert ok is False, 'License should show as expired'
    assert 'License expired' in message


@pytest.mark.license
@mock.patch("cryptography.hazmat.primitives.serialization.load_pem_public_key")
def test_verify_license_expiry_date_still_valid(load_pem_mock, private_key):
    # Arrange
    tomorrows_date = datetime.datetime.strftime(
        datetime.datetime.now() + datetime.timedelta(days=1),
        '%Y/%m/%d'
    )
    load_pem_mock.return_value = private_key.public_key()
    license_manager = _LM()

    # Act & Assert - Exception raised because of invalid signature
    ok, message, license_properties, license_key = \
        license_manager._validate_license_key(
            get_temp_license(private_key, tomorrows_date)
        )
    assert ok is True, 'License should show as valid'
    assert message == '', 'Should not report any errors'
