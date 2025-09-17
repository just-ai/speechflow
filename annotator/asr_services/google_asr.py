import os
import uuid
import logging

from pathlib import Path

from annotator.asr_services.cloud_asr import ASRException, CloudASR
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.io import AudioChunk
from speechflow.logging import trace

__all__ = ["GoogleASR"]

LOGGER = logging.getLogger("root")


class GoogleASR(CloudASR):
    """Generate transcription for audio files."""

    def __init__(
        self,
        asr_credentials: Path,
        locale_code: str,
        raise_on_converter_exc: bool = False,
        raise_on_asr_limit_exc: bool = False,
    ):
        from google.cloud import storage

        super().__init__(
            raise_on_converter_exc=raise_on_converter_exc,
            raise_on_asr_limit_exc=raise_on_asr_limit_exc,
        )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = asr_credentials.as_posix()
        self._locale_code = locale_code
        self._bucket_name = "asr-bucket-big-files-rnd"

        storage_client = storage.Client()

        buckets = storage_client.list_buckets()
        for bucket in buckets:
            if self._bucket_name == bucket.name:
                self._clear_bucket(storage_client)
                break
        else:
            storage_client.create_bucket(self._bucket_name)

    def _clear_bucket(self, storage_client=None):
        from google.cloud import storage

        if storage_client is None:
            storage_client = storage.Client()

        blobs = storage_client.list_blobs(self._bucket_name)
        for blob in blobs:
            LOGGER.info(trace(self, f"Deleting file {blob.name}"))
            blob.delete()

    def _transcription(self, metadata: Metadata) -> Metadata:
        from google.cloud import speech as google_speech
        from google.cloud import storage

        storage_client = storage.Client()
        speech_client = google_speech.SpeechClient()

        md = {"audio_path": metadata["audio_path"]}

        config = google_speech.RecognitionConfig(
            encoding=google_speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=metadata["sr"],
            language_code=self._locale_code,
            enable_word_time_offsets=True,
        )

        if AudioChunk(md["audio_path"]).duration < 60:
            try:
                audio = google_speech.RecognitionAudio(
                    content=metadata["waveform"].to_bytes()
                )
                operation = speech_client.long_running_recognize(
                    config=config, audio=audio
                )
                response = operation.result(timeout=60)
            except Exception as e:
                raise ASRException(e)
        else:
            bucket = storage_client.get_bucket(self._bucket_name)
            blob = bucket.blob(str(uuid.uuid4()))
            blob.upload_from_string(
                metadata["waveform"].to_bytes(), content_type="audio/wav"
            )

            try:
                audio = google_speech.RecognitionAudio(
                    uri=f"gs://{self._bucket_name}/{blob.name}"
                )
                operation = speech_client.long_running_recognize(
                    config=config, audio=audio
                )
                response = operation.result(timeout=3600)
            except Exception as e:
                raise ASRException(e)
            finally:
                blob.delete()

        if not response.results:
            raise ASRException("Speech in the audio file is not recognized!")

        text = []
        timestamps = []
        for response in response.results:
            alternative = response.alternatives[0]
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                text.append(word)
                timestamps.append((word, start_time, end_time))

        md["transcription"] = {
            "text": " ".join(text),
            "timestamps": timestamps,
            "locale_code": self._locale_code,
        }
        return md

    @staticmethod
    def _to_text(transcription) -> str:
        return transcription["text"]
