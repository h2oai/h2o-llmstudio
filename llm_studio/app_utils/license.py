import base64
import datetime
import logging
import os
import sys
from pathlib import Path

import cryptography
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Environment variables
L_K = "LLMSTUDIO_LICENSE_KEY"
L_F = "LLMSTUDIO_LICENSE_FILE"
R_P = "/license/license.sig"
H_P = os.path.join(str(Path.home()), ".h2o_llmstudio", "license.sig")
DEFAULT_LOGGER = logging.getLogger()


class _LM:
    def __init__(self, logger=None, lt=None, lf=None):
        self.pid = os.getpid()
        self.lt = lt
        self.lf = lf
        self.logger = DEFAULT_LOGGER if logger is None else logger
        self.v, self.m, self.p, self.k = self._have_valid_license()

    @staticmethod
    def _read_file(path):
        with open(path, "rb", buffering=0) as f:
            message = f.readall()
            string_message = message.decode("UTF-8")
            return string_message

    @staticmethod
    def _verify(m, s):
        public_key_string = b"-----BEGIN PUBLIC KEY-----\n\
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAupoPgWD+gH89Y3woZkgP\
Z+RznTl2LRGoeU+QsWzVzY02attAz3TP+xldrnvw8En7lG2RU621puw5330WRsGW\
KvpkUxQPKdETQry6k2XrgFEU/zd1Asoxz8EvQCHejxDtU6/4v2oKGsAzNS6I7cHh\
zgToFg7NbXZszwLa6YvGkJ1Z8f8dsgMbTlLXsAiS39hdV+QkyRtpsYKSc8qSYEC8\
xe2weUUdoYQzSFp9yAOCqTSTKEibpCezurR8hkxNNxDyhxKaEJEg0kvxhl8GoVGH\
D5wkXMQ+2mr3JGn1h+W6AXH+G17GWXycQlEPQWxD3/PJAdh3sYEe/038E0j/U1xI\
aQIDAQAB\n\
-----END PUBLIC KEY-----"
        public_key = serialization.load_pem_public_key(
            data=public_key_string, backend=default_backend()
        )
        try:
            # FIXME: Update padding and hashing algos as these are deprecated
            public_key.verify(
                signature=s,
                data=m,
                padding=padding.PKCS1v15(),
                algorithm=hashes.SHA1(),
            )
        except cryptography.exceptions.InvalidSignature:
            raise ValueError("Invalid license file (invalid signature)")

    def _decode_license_key(self, esm):
        success = False
        try:
            dsm = base64.urlsafe_b64decode(esm)
            if len(dsm) < 257:
                raise ValueError("Invalid license file (base64 decode failed)")
            s = dsm[0:256]
            m = dsm[256:]
            self._verify(m, s)
            string_message = m.decode("UTF-8")
            success = True
            return string_message
        finally:
            if not success:
                self.logger.warning(
                    "The following license key could not be decoded and verified:"
                )
                self.logger.warning(esm)

    def _validate_license_key(self, esm):
        try:
            string_message = self._decode_license_key(esm)

            lines = string_message.split("\n")
            license_properties = {}
            for line in lines:
                line = line.strip()
                line = line.partition("#")[0]
                if len(line) <= 0:
                    continue
                (key, value) = line.split(":")
                key = key.strip()
                value = value.strip()
                license_properties[key] = value

            expiration_date_string = license_properties["expiration_date"]
            expiration_date = datetime.datetime.strptime(
                expiration_date_string, "%Y/%m/%d"
            )

            if self._calc_days_left(expiration_date) <= 0:
                raise ValueError(
                    "License expired (on " + expiration_date_string + ")"
                )

            return True, "", license_properties, esm
        except Exception as e:
            return False, str(e), {}, ""

    def _validate_license_file(self, path):
        if not os.path.exists(path) or not os.path.isfile(path):
            return False, "License file does not exist (" + path + ")", {}, ""
        self.logger.info("License file exists (" + path + ")")
        encoded_signed_message = self._read_file(path)
        return self._validate_license_key(encoded_signed_message)

    def _have_valid_license(self):
        self.logger.info(
            "-----------------------------------------------------------------"
        )
        self.logger.info("Checking whether we have a valid license...")

        error_message = []

        # try 0.1: Config license_text
        if self.lt is not None:
            self.logger.info("Trying the provided license_text in the config")
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_key(self.lt)
            if ok:
                return (
                    True,
                    "License is valid from license_text in config",
                    license_properties,
                    license_key,
                )
            else:
                error_message.append("license_text: " + message)

        # try 0.2: Config license_file
        if (
            self.lf is not None
            and os.path.exists(self.lf)
            and os.path.isfile(self.lf)
        ):
            self.logger.info("Trying the provided license_file in the config")
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_file(self.lf)
            if ok:
                return (
                    True,
                    "License is valid from license_file in config",
                    license_properties,
                    license_key,
                )
            else:
                error_message.append("license_file: " + message)

        # try 1: AUTODOC_LICENSE_FILE
        if L_F in os.environ:
            self.logger.info("Trying license from environment " + L_F)
            key = os.environ[L_F]
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_file(key)
            if ok:
                return (
                    True,
                    "License is valid from " + L_F,
                    license_properties,
                    license_key,
                )
            else:
                error_message.append(L_F + ": " + message)

        # try 2: AUTODOC_LICENSE_KEY
        if L_K in os.environ:
            self.logger.info("Trying license from environment " + L_K)
            key = os.environ[L_K]
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_key(key)
            if ok:
                return (
                    True,
                    "License is valid from " + L_K,
                    license_properties,
                    license_key,
                )
            else:
                error_message.append(L_K + ": " + message)

        # try 3: /license/license.sig
        if os.path.exists(R_P) and os.path.isfile(R_P):
            self.logger.info("Trying license at " + R_P)
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_file(R_P)
            if ok:
                return (
                    True,
                    "License is valid from " + R_P,
                    license_properties,
                    license_key,
                )
            else:
                error_message.append(R_P + ": " + message)

        # try 4: ~/.h2o_autodoc/license.sig
        if os.path.exists(H_P) and os.path.isfile(H_P):
            self.logger.info("Trying license at " + H_P)
            (
                ok,
                message,
                license_properties,
                license_key,
            ) = self._validate_license_file(H_P)
            if ok:
                return (
                    True,
                    "License is valid from " + H_P,
                    license_properties,
                    license_key,
                )
            else:
                error_message.append(H_P + ": " + message)

        return False, "\n".join(error_message), {}, ""

    @staticmethod
    def _calc_days_left(expiration_date):
        today = datetime.date.today()
        return max(0, (expiration_date.date() - today).days)

    # noinspection PyBroadException
    @staticmethod
    def _runtime_check(lt=None, lf=None, logger=DEFAULT_LOGGER):
        l_message = ""
        try:
            lm = _LM(logger=logger, lt=lt, lf=lf)
            if not lm.v:
                raise
            l_message = lm.m
        except Exception:
            logger.error(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            logger.error(
                l_message + "\n"
                "No other valid license key found\n"
                "The license can be specified in the following ways (precedence from top to bottom):\n"
                "  * Config Class Parameter: \n"
                "    - 'license_file'               : The file system location for the license file\n"
                "    - 'license_text'               : The license text\n"
                "  * Environment variable: \n"
                "    - 'AUTODOC_LICENSE_FILE'       : location of file containing the license key\n"
                "    - 'AUTODOC_LICENSE_KEY'        : license key\n"
                "  * License file in standard location: \n"
                "    - '/license/license.sig'       : file containing the license key\n"
                "    - '~/.h2o_autodoc/license.sig' : file containing the license key on user home\n"
            )
            logger.error(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            sys.exit(l_message)
