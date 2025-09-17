import math
import typing as tp

import numpy as np
import torch
import numpy.typing as npt

from numba import njit
from scipy.fftpack import ifft

__all__ = ["LPCCompute", "LPCDecompose"]


class LPCError(Exception):
    """Exception raised in LPCDecompose."""

    pass


class LPCCompute:
    def __init__(
        self,
        order: int,
        ac_adjustment: bool = True,
        method: str = "levinson_durbin",
    ):
        self.order = order
        self.ac_adjustment = ac_adjustment
        self.method = method

    @staticmethod
    def _levinson_durbin(
        r: npt.NDArray, order: tp.Optional[int] = None, allow_singularity: bool = False
    ):
        r"""Levinson-Durbin recursion.
        Find the coefficients of a length(r)-1 order autoregressive linear process
        :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
        :param order: requested order of the autoregressive coefficients. default is N.
        :param allow_singularity: false by default. Other implementations may be True (e.g., octave)
        :return:
            * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
            * the prediction errors
            * the `N` reflections coefficients values
        This algorithm solves the set of complex linear simultaneous equations
        using Levinson algorithm.
        .. math::
            \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
            \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)
        where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
        :math:`T_0, T_1, \dots ,T_M`.
        .. note:: Solving this equations by Gaussian elimination would
            require :math:`M^3` operations whereas the levinson algorithm
            requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.
        This is equivalent to solve the following symmetric Toeplitz system of
        linear equations
        .. math::
            \left( \begin{array}{cccc}
            r_1 & r_2^* & \dots & r_{n}^*\\
            r_2 & r_1^* & \dots & r_{n-1}^*\\
            \dots & \dots & \dots & \dots\\
            r_n & \dots & r_2 & r_1 \end{array} \right)
            \left( \begin{array}{cccc}
            a_2\\
            a_3 \\
            \dots \\
            a_{N+1}  \end{array} \right)
            =
            \left( \begin{array}{cccc}
            -r_2\\
            -r_3 \\
            \dots \\
            -r_{N+1}  \end{array} \right)
        where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
        :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
        a vector of autocorrelation coefficients where lag 0 is the first
        element :math:`r_1`.
        """

        T0 = np.real(r[:, 0])
        T = r[:, 1:]
        M = T.shape[1]

        if order is None:
            M = T.shape[1]
        else:
            assert order <= M, "order must be less than size of the input data"
            M = order

        B = r.shape[0]

        realdata = np.isrealobj(r)
        if realdata is True:
            A = np.zeros((B, M), dtype=float)
            ref = np.zeros((B, M), dtype=float)
        else:
            A = np.zeros((B, M), dtype=complex)
            ref = np.zeros((B, M), dtype=complex)

        P = T0

        for k in range(0, M):
            save = T[:, k]
            if k == 0:
                temp = -save / P
            else:
                # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
                for j in range(0, k):
                    save = save + A[:, j] * T[:, k - j - 1]
                temp = -save / P
            if realdata:
                P = P * (1.0 - temp**2.0)
            else:
                P = P * (1.0 - (temp.real**2 + temp.imag**2))
            if not allow_singularity and any(P <= 0):
                raise ValueError("singular matrix")
            A[:, k] = temp
            ref[:, k] = temp  # save reflection coeff at each step
            if k == 0:
                continue

            khalf = (k + 1) // 2
            if realdata is True:
                for j in range(0, khalf):
                    kj = k - j - 1
                    save = A[:, j].copy()
                    A[:, j] = save + temp * A[:, kj]
                    if j != kj:
                        A[:, kj] += temp * save
            else:
                for j in range(0, khalf):
                    kj = k - j - 1
                    save = A[:, j].copy()
                    A[:, j] = save + temp * A[:, kj].conjugate()
                    if j != kj:
                        A[:, kj] = A[:, kj] + temp * save.conjugate()

        return A, P, ref

    @staticmethod
    def _celt_lpc(ac, order):
        error = ac[0]
        lpc = np.zeros(order)
        rc = np.zeros(order).astype(float)
        if ac[0] != 0:
            for i in range(order):
                rr = 0
                for j in range(i):
                    rr += lpc[j] * ac[i - j]
                rr += ac[i + 1]
                r = -rr / error
                rc[i] = r
                lpc[i] = r
                for j in range(int((i + 1) / 2)):
                    tmp1 = lpc[j]
                    tmp2 = lpc[i - 1 - j]
                    lpc[j] = tmp1 + (r * tmp2)
                    lpc[i - 1 - j] = tmp2 + (r * tmp1)
                error -= r * r * error
                if error < ac[0] / (2**10):
                    break
                if error < 0.001 * ac[0]:
                    break

        return lpc, error, rc

    def _linear_to_autocorr(self, linear: npt.NDArray) -> npt.NDArray:
        power = linear**2

        nbands, nframes = power.shape
        ncorr = 2 * (nbands - 1)
        R = np.zeros((ncorr, nframes))

        R[0:nbands, :] = power
        for i in range(nbands - 1):
            R[i + nbands - 1, :] = power[nbands - (i + 1), :]

        ac = ifft(R.T).real.T
        ac = ac[0:nbands, :]
        return ac

    def _autocorr_to_lpc(
        self,
        ac: npt.NDArray,
    ) -> tp.Tuple[npt.NDArray, npt.NDArray]:
        if self.ac_adjustment:
            # https://github.com/mozilla/LPCNet/blob/master/src/freq.c#L224
            # -40 dB noise floor
            ac[0, :] += (2.0 + ac[0, :]) * 1e-4
            # lag windowing
            for i in range(1, self.order + 1):
                ac[i, :] *= 1 - 6e-5 * i * i

        if self.method == "levinson_durbin":
            lpcs, errs, _ = self._levinson_durbin(
                ac.T.astype(np.float64), self.order, allow_singularity=True
            )

        elif self.method == "celt_lpc":
            lpcs = []
            for i in range(ac.shape[1]):
                lpc, _, _ = self._celt_lpc(ac.T[i].astype(np.float64), self.order)
                lpcs.append(lpc)
            lpcs = np.stack(lpcs)

        else:
            raise NotImplementedError(f"'{self.method}' not implemented.")

        lpcs = lpcs.T  # / (np.tile(errs.T, (self.order, 1)) + 1e-8)
        return lpcs.astype(np.float32)

    def linear_to_lpc(self, linear):
        return self._autocorr_to_lpc(self._linear_to_autocorr(linear))

    def lpc_reconstruction(self, lpcs, audio):
        num_points = lpcs.shape[-1]

        if audio.shape[0] == num_points:
            audio = np.pad(audio, (self.order, 0), "constant")

        elif audio.shape[0] != num_points + self.order:
            raise RuntimeError("dimensions of lpcs and audio must match")

        indices = np.reshape(np.arange(self.order), [-1, 1]) + np.arange(lpcs.shape[-1])

        signal_slices = audio[indices]
        pred = np.sum(lpcs * signal_slices, axis=0).clip(-1.0, 1.0)
        origin_audio = audio[self.order :]

        error = origin_audio - pred
        return origin_audio, pred, error


@njit
def lin2ulaw(x, bits=8):
    mu = 2**bits
    half_mu = mu / 2
    max_pair = np.array([0, 0], dtype=np.int32)
    min_pair = np.array([0, mu - 1], dtype=np.int32)
    s = np.sign(x)
    x = np.abs(x)
    u = s * (half_mu * np.log(1 + (mu - 1) * x) / np.log(mu))
    u = half_mu + np.round(u)
    max_pair[0] = u
    min_pair[0] = max_pair.max()
    return min_pair.min()


@njit
def ulaw2lin(u, bits: int = 8):
    mu = 2**bits
    half_mu = mu / 2
    u = u - half_mu
    s = np.sign(u)
    u = np.abs(u)
    return s * (1 / (mu - 1)) * (np.exp(u / half_mu * np.log(mu)) - 1)


@njit
def uint_2_float(y, bits: int = 8):
    mu = 2**bits
    return (y / (mu - 1) * 2) - 1.0


class LPCDecompose:
    def __init__(
        self,
        frame_size: int,
        ulaw_bits: int = 10,
        add_noise: bool = False,
        noise_std: float = 2,
    ):
        self.frame_size = frame_size
        self.ulaw_bits = ulaw_bits
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, wave, lpc):
        order, num_frames = lpc.shape
        wave_len = self.frame_size * num_frames

        wave = np.pad(wave, (0, wave_len - wave.shape[0]), "constant")
        frames = wave.reshape(-1, self.frame_size)

        accumulator = np.zeros(order, dtype=np.float32)
        exc_mem = np.array([2 ** (self.ulaw_bits - 1)], dtype=np.float32)
        result = np.zeros((num_frames, self.frame_size, 4), dtype=np.float32)

        if self.add_noise:
            full_noise = LPCDecompose.noise_gen(
                self.frame_size * num_frames, self.noise_std
            )
            full_noise = full_noise.reshape(num_frames, self.frame_size)
        else:
            full_noise = np.zeros((num_frames, self.frame_size), dtype=np.float32)

        for idx, (wave_frame, lpc_frame, noise_frame) in enumerate(
            zip(frames, lpc.T, full_noise)
        ):
            result[idx] = LPCDecompose.decompose(
                wave_frame,
                noise_frame,
                accumulator,
                result[idx],
                lpc_frame,
                exc_mem,
                self.ulaw_bits,
            )

        if np.abs(result[:, :, 0]).max() > 1:
            raise LPCError("lpc decomposition is unstable!")

        return result

    @staticmethod
    def noise_gen(size, noise_std):
        epsilon = 1e-8
        tmp = np.random.rand()
        noise_std = noise_std * tmp * tmp
        return np.round(
            noise_std
            * 0.707
            * (
                np.log(np.random.rand(size) + epsilon)
                - np.log(np.random.rand(size) + epsilon)
            )
        ).astype(np.float32)

    @staticmethod
    @njit
    def decompose(wave_frame, noise_frame, accumulator, result, lpc_frame, exc_mem, bits):
        max_pair = np.array([0, 0], dtype=np.float32)
        min_pair = np.array([0, 2**bits], dtype=np.float32)
        decomposition = np.zeros(4)

        for idx, (value, noise) in enumerate(zip(wave_frame, noise_frame)):
            pred = -(lpc_frame * accumulator).sum()
            e = lin2ulaw(value - pred, bits=bits)

            decomposition[0] = accumulator[0]
            decomposition[1] = pred
            decomposition[2] = exc_mem[-1]
            decomposition[3] = uint_2_float(e, bits=bits)

            e += noise
            max_pair[0] = e
            min_pair[0] = max_pair.max()
            e = min_pair.min()

            accumulator[1:] = accumulator[:-1]
            accumulator[0] = pred + ulaw2lin(e, bits=bits)
            exc_mem[-1] = e
            result[idx] = decomposition

        return result

    @staticmethod
    def torch_lin2ulaw(x, bits: int = 8):
        mu = 2**bits
        half_mu = mu / 2
        s = torch.sign(x)
        x = torch.abs(x)
        u = s * (half_mu * torch.log(1 + (mu - 1) * x) / math.log(mu))
        u = torch.clamp(half_mu + torch.round(u), 0, mu - 1)
        return u.long()

    @staticmethod
    def torch_ulaw2lin(u, bits: int = 8):
        mu = 2**bits
        half_mu = mu / 2
        u = u.float() - half_mu
        s = torch.sign(u)
        u = torch.abs(u)
        return s * (1 / (mu - 1)) * (torch.exp(u / half_mu * math.log(mu)) - 1)

    @staticmethod
    def torch_float_2_uint(y, bits: int = 8):
        mu = 2**bits
        return torch.round(((y + 1.0) / 2.0) * (mu - 1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
    from speechflow.io import AudioChunk, Config
    from speechflow.utils.fs import get_root_dir

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _lpc_order = 9
    _ulaw_bits = 10

    _cfg = Config(
        {
            "load": {"sample_rate": 22050},
            "preemphasis": {"beta": 0.85},
            "magnitude": {"n_fft": 1024, "hop_len": 256, "win_len": 1024},
            "linear_to_lpc": {"order": _lpc_order},
            "linear_to_mel": {"n_mels": 80},
            "lpc_from_mel": {"order": _lpc_order},
        }
    )

    signal_proc = SignalProcessor(("load", "preemphasis"), _cfg)
    sp_proc = SpectralProcessor(("magnitude",), _cfg)
    mel_proc = MelProcessor(("linear_to_mel", "amp_to_db"), _cfg)
    lpc_proc = LPCProcessor(("lpc_from_mel",), _cfg)

    ds = SpectrogramDataSample(file_path=_wav_path)
    ds = signal_proc.process(ds)
    ds = sp_proc.process(ds)
    ds = mel_proc.process(ds)
    ds = lpc_proc.process(ds)

    ds = lpc_proc.lpc_decompose(ds, ulaw_bits=_ulaw_bits, add_noise=True)

    waveform = ds.lpc_waveform[:, 0]
    pred = ds.lpc_waveform[:, 1]
    err = torch.tensor(ds.lpc_waveform[:, 2])
    err = LPCDecompose.torch_ulaw2lin(err, bits=_ulaw_bits).numpy()

    AudioChunk(data=waveform, sr=ds.audio_chunk.sr).save("orig.wav", True)
    AudioChunk(data=pred, sr=ds.audio_chunk.sr).save("pred.wav", True)
    AudioChunk(data=err, sr=ds.audio_chunk.sr).save("error.wav", True)

    fig = plt.figure(figsize=(18, 5))

    n = len(waveform)
    ts = np.array(range(0, n))
    plt.plot(ts, waveform, "r", label="original")

    pred_padded = np.concatenate([pred, np.zeros(n - len(pred))])
    plt.plot(ts, pred_padded, "b", label="predict")

    err_padded = np.concatenate([err, np.zeros(n - len(err))])
    plt.plot(ts, err_padded, "y", label="error")

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small")

    plt.show()
