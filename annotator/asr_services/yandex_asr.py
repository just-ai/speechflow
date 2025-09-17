import json
import time
import uuid
import typing as tp
import logging
import datetime

from dataclasses import dataclass
from pathlib import Path

import yaml
import requests

from requests.adapters import HTTPAdapter
from urllib3 import Retry

from annotator.asr_services.cloud_asr import (
    ASRException,
    ASRRequestLimitException,
    CloudASR,
)
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.data_pipeline.dataset_parsers import AudioDSParser

__all__ = ["YandexASR"]

LOGGER = logging.getLogger("root")


@dataclass
class YandexASRConfig:
    api_key: str
    bucket_name: str
    storage_access_key: tp.Optional[str] = None
    storage_secret_key: tp.Optional[str] = None
    storage_endpoint_url: str = "https://storage.yandexcloud.net"


class YandexASR(CloudASR):
    """Generate transcription for audio files."""

    def __init__(
        self,
        asr_credentials: Path,
        locale_code: str,
        max_http_retries: int = 3,
        raise_on_converter_exc: bool = False,
        raise_on_asr_limit_exc: bool = False,
    ):
        super().__init__(
            raise_on_converter_exc=raise_on_converter_exc,
            raise_on_asr_limit_exc=raise_on_asr_limit_exc,
        )
        if asr_credentials.suffix == ".yml":
            credentials = yaml.load(
                asr_credentials.read_text(encoding="utf-8"), Loader=yaml.SafeLoader
            )
        else:
            credentials = json.loads(asr_credentials.read_text(encoding="utf-8"))

        self._credentials = YandexASRConfig(
            api_key=credentials.get("api_key"),
            bucket_name=credentials.get("bucket_name"),
            storage_access_key=credentials.get("storage_access_key"),
            storage_secret_key=credentials.get("storage_secret_key"),
        )
        self._locale_code = locale_code

        # self._clear_bucket()

        self._POST = (
            "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"
        )
        self._BODY = {
            "config": {
                "specification": {"languageCode": self._locale_code, "rawResults": True}
            },
        }
        self._HEADER = {"Authorization": f"Api-Key {self._credentials.api_key}"}
        self._GET = "https://operation.api.cloud.yandex.net/operations/{id}"

        s = requests.Session()
        retry = Retry(total=max_http_retries, backoff_factor=0.4)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        self.requests_session = s

    def _create_session(self, service_name: str = "s3"):
        import boto3

        # https://cloud.yandex.ru/docs/storage/tools/aws-cli
        # https://cloud.yandex.ru/docs/storage/tools/boto
        # https://cloud.yandex.ru/docs/iam/operations/sa/create-access-key
        session = boto3.session.Session(
            aws_access_key_id=self._credentials.storage_access_key,
            aws_secret_access_key=self._credentials.storage_secret_key,
            region_name="ru-central1",
        )
        return session.client(
            service_name=service_name,
            endpoint_url=self._credentials.storage_endpoint_url,
        )

    def _clear_bucket(self):
        try:
            s3 = self._create_session()
            objects = s3.list_objects(Bucket=self._credentials.bucket_name)
            if "Contents" in objects:
                for key in objects["Contents"]:
                    s3.delete_object(Bucket=self._credentials.bucket_name, Key=key["Key"])
        except Exception as e:
            LOGGER.error(e)

    def _transcription(self, metadata: Metadata) -> Metadata:
        metadata = AudioDSParser.audio_converter(metadata)[0]

        s3 = self._create_session()
        obj_name = f"{uuid.uuid4()}.ogg"
        s3.upload_fileobj(metadata["audio_data"], self._credentials.bucket_name, obj_name)

        md = {"audio_path": metadata["audio_path"]}
        try:
            obj_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._credentials.bucket_name, "Key": obj_name},
                ExpiresIn=3600,
            )

            self._BODY["audio"] = {"uri": obj_url}
            req = self.requests_session.post(
                self._POST, headers=self._HEADER, json=self._BODY
            ).json()

            if "id" not in req:
                if "limit" in req["message"]:
                    if self._raise_on_asr_limit_exc:
                        raise ASRRequestLimitException(req["message"])
                    else:
                        self._sleep()
                        self._transcription(metadata)
                else:
                    raise ASRException(req["message"])

            while True:
                time.sleep(5)
                req = self.requests_session.get(
                    self._GET.format(id=req["id"]), headers=self._HEADER
                ).json()
                msg = req.get("message", "")
                if "limit" in msg:
                    if self._raise_on_asr_limit_exc:
                        raise ASRRequestLimitException(msg)
                    else:
                        LOGGER.warning(f"{msg} - sleep...")
                        self._sleep()
                elif "code" in req and req["code"] == 13:
                    raise ASRException(msg)
                elif "error" in req:
                    raise ASRException(req["error"])
                elif req["done"]:
                    break

            if "chunks" not in req["response"]:
                raise ASRException("Speech in the audio file is not recognized!")

            chunks = req["response"]["chunks"]
            md["transcription"] = [data["alternatives"] for data in chunks]
            md["locale_code"] = self._locale_code
            return md
        finally:
            s3.delete_object(Bucket=self._credentials.bucket_name, Key=obj_name)

    @staticmethod
    def _to_text(transcription) -> str:
        lines = []
        for chunk in transcription:
            ts_begin = chunk[0]["words"][0]["startTime"][:-1]
            ts_begin = str(datetime.timedelta(seconds=float(ts_begin))).split(".")[0]
            ts_end = chunk[0]["words"][-1]["endTime"][:-1]
            ts_end = str(datetime.timedelta(seconds=float(ts_end))).split(".")[0]
            text = chunk[0]["text"]
            lines.append(f"{ts_begin}:{ts_end}\t{text}\n")

        return "".join(lines)
