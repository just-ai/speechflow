import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # fix for Torch Hub

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from speechflow.data_pipeline.datasample_processors.audio_augmentation import *
from speechflow.data_pipeline.datasample_processors.audio_processors import *
from speechflow.data_pipeline.datasample_processors.auxiliary import *
from speechflow.data_pipeline.datasample_processors.biometric_processors import *
from speechflow.data_pipeline.datasample_processors.image_processors import *
from speechflow.data_pipeline.datasample_processors.spectrogram_augmentation import *
from speechflow.data_pipeline.datasample_processors.spectrogram_processors import *
from speechflow.data_pipeline.datasample_processors.speech_quality import *
from speechflow.data_pipeline.datasample_processors.ssml_processors import *
from speechflow.data_pipeline.datasample_processors.tts_processors import *
from speechflow.data_pipeline.datasample_processors.tts_singletons import *
from speechflow.data_pipeline.datasample_processors.tts_text_processors import *
