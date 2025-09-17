from pathlib import Path

import numpy as np
import torch
import numpy.typing as npt

from speechflow.io import AudioChunk
from speechflow.utils.profiler import Profiler
from tts.forced_alignment.eval_interface import GlowTTSEvaluationInterface
from tts.vocoders.eval_interface import VocoderEvaluationInterface


def plot_spectrogram(spec: npt.NDArray):
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots()
    mel = np.flip(spec, axis=0)
    im = ax.imshow(mel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.25, pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.show()


if __name__ == "__main__":
    device = "cuda:0"

    tts_model_path = Path(
        "P:\\TTS\\GlowTTS\\13_May_2021_15_07_witcher_ru_stage2_\\_checkpoints\\epoch=129-step=129999.ckpt"
    )
    voc_model_path = Path("P:/TTS/hifigan_finetuned.ckpt")

    tts = GlowTTSEvaluationInterface(ckpt_path=tts_model_path, device=device)
    voc = VocoderEvaluationInterface(ckpt_path=voc_model_path, device=device)

    utterance = """
    Ай, у меня зуб болит!
        """
    ref_wav = Path("P:\\TTS\\2\\Beauclair_Citizen_Woman_13\\45373.wav")

    speaker = "Geralt"

    with Profiler():
        tts_out = tts.synthesize(utterance, ref_wav, speaker)
        voc_out = voc.synthesize(tts_out)

    mel = tts_out.spectrogram.cpu().numpy()
    plot_spectrogram(mel[0].transpose())

    if not isinstance(voc_out, list):
        voc_out = [voc_out]

    wave_sr = 22050
    waveform = []
    for out in voc_out:
        waveform.append(out.waveform.cpu()[0])
        # AudioChuck._apply_fade(waveform[-1], wave_sr, 0.01, 0.01)
        # waveform.append(torch.zeros(3000))
    waveform = torch.cat(waveform)

    AudioChunk(data=waveform.numpy(), sr=wave_sr).save(
        rf"P:\TTS\{speaker}_{tts_model_path.name}_{ref_wav.name}.wav", overwrite=True
    )
