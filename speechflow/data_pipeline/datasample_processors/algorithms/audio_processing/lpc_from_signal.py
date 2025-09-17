from math import floor

import numpy as np

from numpy.random import randn
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann


class LPCCompute:
    @staticmethod
    def create_overlapping_blocks(x, w, R=0.5):
        """Split the original signal into overlapping blocks.

        x - a vector representing the time-series signal
        w - array corresponding to weights of the window function
        R - optional overlapping factor

        Returns:

        B - list of overlapping blocks

        """
        n = len(x)
        nw = len(w)
        step = floor(nw * (1 - R))
        nb = floor((n - nw) / step) + 1

        B = np.zeros((nb, nw))

        for i in range(nb):
            offset = i * step
            B[i, :] = w * x[offset : nw + offset]

        return B

    @staticmethod
    def make_matrix_X(x, p):
        n = len(x)
        # [x_n, ..., x_1, 0, ..., 0]
        xz = np.concatenate([x[::-1], np.zeros(p)])

        X = np.zeros((n - 1, p))
        for i in range(n - 1):
            offset = n - 1 - i
            X[i, :] = xz[offset : offset + p]
        return X

    @staticmethod
    def solve_lpc(x, p, ii):
        """An implementation of LPC.

        A detailed explanation can be found at
        https://ccrma.stanford.edu/~hskim08/lpc/

        x - a vector representing the time-series signal
        p - the polynomial order of the all-pole filter

        a - the coefficients to the all-pole filter
        g - the variance(power) of the source (scalar)
        e - the full error signal

        NOTE: This is not the most efficient implementation of LPC.
        Matlab's own implementation uses FFT to via the auto-correlation method
        which is noticeably faster. (O(n log(n)) vs O(n^2))

        """
        b = x[1:].T

        X = LPCCompute.make_matrix_X(x, p)

        a = np.linalg.lstsq(X, b, rcond=None)[0]

        e = b.T - np.dot(X, a)
        g = np.var(e)

        return [a, g, e]

    @staticmethod
    def lpc_encode(x, p, w):
        """Encodes the input signal into lpc coefficients using 50% OLA.

        x - single channel input signal
        p - lpc order
        nw - window length

        A - the coefficients
        G - the signal power
        E - the full source (error) signal

        """
        B = LPCCompute.create_overlapping_blocks(x, w)

        [nb, nw] = B.shape

        A = np.zeros((p, nb))
        G = np.zeros((1, nb))
        E = np.zeros((nw, nb))

        for i in range(nb):
            [a, g, e] = LPCCompute.solve_lpc(B[i, :], p, i)

            A[:, i] = a
            G[:, i] = g
            E[1:, i] = -e

        return [A, G, E]

    @staticmethod
    def add_overlapping_blocks(B, R=0.5):
        """Reconstruct the original signal from overlapping blocks.

        B - list of overlapping blocks (see create_overlapping_blocks)

        x - the rendered signal

        """
        [count, nw] = B.shape
        step = floor(nw * R)

        n = (count - 1) * step + nw

        x = np.zeros((n,))

        for i in range(count):
            offset = i * step
            x[offset : nw + offset] += B[i, :]

        return x

    @staticmethod
    def run_source_filter(a, g, e, block_size, rand_noise: bool = False):
        if rand_noise:
            src = np.sqrt(g) * randn(block_size, 1)  # noise
        else:
            src = np.expand_dims(e, 1)

        b = np.concatenate([np.array([-1]), a])

        x_hat = lfilter([1], b.T, src.T).T
        return np.squeeze(x_hat), np.squeeze(src)

    @staticmethod
    def lpc_decode(A, G, w, E, lowcut=0, rand_noise: bool = False):
        """Decodes the LPC coefficients into.

        * A - the LPC filter coefficients
        * G - the signal power(G) or the signal power with fundamental frequency(GF)
               or the full source signal(E) of each windowed segment.
        * w - the window function
        * lowcut - the cutoff frequency in normalized frequencies for a lowcut
                  filter.

        """
        [ne, n] = G.shape
        nw = len(w)
        [p, _] = A.shape

        B_hat = np.zeros((n, nw))
        E_hat = np.zeros((n, nw))

        for i in range(n):
            B_hat[i, :], E_hat[i, :] = LPCCompute.run_source_filter(
                A[:, i], G[:, i], E[:, i], nw, rand_noise
            )

        # recover signal from blocks
        x_hat = LPCCompute.add_overlapping_blocks(B_hat)
        e_hat = LPCCompute.add_overlapping_blocks(E_hat)

        return x_hat, e_hat


if __name__ == "__main__":
    import scipy.io.wavfile
    import matplotlib.pyplot as plt

    from speechflow.utils.fs import get_root_dir

    wav_path = get_root_dir() / "tests/data/test_audio.wav"
    sample_rate, waveform = scipy.io.wavfile.read(wav_path)
    waveform = np.array(waveform)

    # normalize
    # waveform = 0.9 * waveform / max(abs(waveform));

    # resampling
    target_sample_rate = 22050
    win_len = 800
    lpc_order = 16  # number of poles

    target_size = int(len(waveform) * target_sample_rate / sample_rate)
    waveform = resample(waveform, target_size)
    sample_rate = target_sample_rate

    # Hann window
    sym = False  # periodic
    w = hann(win_len, sym)

    # Encode
    [A, G, E] = LPCCompute.lpc_encode(waveform, lpc_order, w)

    # Print stats
    original_size = len(waveform)
    model_size = A.size + G.size
    print("Original signal size:", original_size)
    print("Encoded signal size:", model_size)
    print("Data reduction:", original_size / model_size)

    xhat, ehat = LPCCompute.lpc_decode(A, G, w, E, rand_noise=True)
    scipy.io.wavfile.write("lpc_reconstruction_1.wav", sample_rate, xhat.astype(np.int16))

    if 0:  # waveform reconstruction from mel
        from speechflow.data_pipeline.datasample_processors.audio_processors import (
            SignalProcessor,
        )
        from speechflow.data_pipeline.datasample_processors.data_types import (
            SpectrogramDataSample,
        )
        from speechflow.data_pipeline.datasample_processors.spectrogram_processors import (
            LPCProcessor,
            MelProcessor,
            SpectralProcessor,
        )
        from speechflow.io import Config

        cfg = Config(
            {
                "load": {"sample_rate": sample_rate},
                "magnitude": {
                    "n_fft": win_len,
                    "hop_len": win_len // 2,
                    "win_len": win_len,
                    "win_type": "half",
                },
                "linear_to_mel": {"n_mels": 80},
                "mel_to_lpc": {"order": lpc_order, "ac_adjustment": False},
            }
        )

        signal_proc = SignalProcessor(("load",), cfg)
        sp_proc = SpectralProcessor(("magnitude",), cfg)
        mel_proc = MelProcessor(("linear_to_mel", "amp_to_db"), cfg)
        lpc_proc = LPCProcessor(("lpc_from_mel",), cfg)

        ds = SpectrogramDataSample(file_path=wav_path)
        ds = signal_proc.process(ds)
        ds = sp_proc.process(ds)
        ds = mel_proc.process(ds)
        ds = lpc_proc.process(ds)

        A = -1 * ds.lpc_feat[: A.shape[1], :].T

        xhat, ehat = LPCCompute.lpc_decode(A, G, w, E, rand_noise=False)
        scipy.io.wavfile.write(
            "lpc_reconstruction_2.wav", sample_rate, xhat.astype(np.int16)
        )

    fig = plt.figure(figsize=(18, 5))

    n = len(waveform)
    ts = np.array(range(0, n))
    plt.plot(ts, waveform, "r", label="original")

    xhat_padded = np.concatenate([xhat, np.zeros(n - len(xhat))])
    plt.plot(ts, xhat_padded, "b", label="predict")

    ehat_padded = np.concatenate([ehat, np.zeros(n - len(ehat))])
    plt.plot(ts, ehat_padded, "y", label="error")

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small")

    plt.show()
