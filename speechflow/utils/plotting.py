import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

import typing as tp

import numpy as np
import matplotlib
import numpy.typing as npt

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")

__all__ = [
    "figure_to_ndarray",
    "plot_1d",
    "plot_tensor",
    "plot_alignment",
    "plot_statistics",
    "plot_spectrogram",
    "plot_durations_and_signals",
    "phonemes_to_frame_ticks",
    "phonemes_to_frame_ticks_with_durations",
]


@tp.no_type_check
def figure_to_ndarray(fig: plt.Figure, tensorboard=True) -> npt.NDArray:
    """Convert a matplotlib Figure to an image represented as ndarray.

    :param fig: plt.figure
    :param tensorboard: reshape for tensorboard SummaryWriter
    :return: image encoded in in a 3d-darray

    """
    width, height = fig.canvas.get_width_height()

    buffer = fig.canvas.tostring_rgb()
    channels = 3  # RGB
    data = np.frombuffer(buffer, dtype=np.uint8()).reshape((height, width, channels))

    if tensorboard:
        # (h, w, c) -> (c, w, h)
        return data.transpose(2, 0, 1)
    else:
        # (h, w, c) -> (w, h, c)
        return data.transpose(1, 0, 2)


def plot_1d(signal: npt.NDArray, dpi: int = 80, **kwargs) -> tp.Union[plt.Figure, None]:
    """Plot a 1d signal.

    :param signal: (n_bands, n_frames)
    :param dpi: DPI
    :return: a matplotlib figure

    """
    w = signal.shape[0] if signal.ndim == 1 else signal.shape[1]

    fig = plt.figure()
    fig.set_size_inches(w / 16, 8)  # type: ignore
    fig.set_dpi(dpi)  # type: ignore
    ax = fig.gca()  # type: ignore

    if signal.ndim == 1:
        ax.plot(signal)
    else:
        for i in range(signal.shape[0]):
            ax.plot(signal[i])

    if not kwargs.get("dont_close", False):
        fig.canvas.draw()  # type: ignore
        return fig

    return None


def plot_tensor(t, title=None):
    import numpy as np
    import torch
    import matplotlib
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    matplotlib.use("TkAgg")

    if t.ndim > 3:
        raise NotImplementedError

    if t.ndim == 3:
        t = torch.cat([i for i in t][::-1], dim=1)

    if isinstance(t, torch.Tensor):
        t = t.t().detach().cpu().numpy()

    if t.ndim > 1:
        fig, ax = plt.subplots()
        plt.title(title)
        mel = np.flip(t, axis=0)
        im = ax.imshow(mel)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.25, pad=0.05)
        fig.colorbar(im, cax=cax)  # type: ignore
    else:
        plot_1d(t)

    plt.show()


def plot_alignment(
    alignment: npt.NDArray, phonemes: tp.Optional[tp.List[str]] = None, dpi: int = 80
) -> plt.Figure:
    """Plot alignment, optionally with phoneme_symb on x-axis.

    :param alignment: (n_frames, n_phonemes)
    :param phonemes: a list of phoneme_symb
    :param dpi: DPI
    :return: a matplotlib Figure

    """
    n_frames, n_phonemes = alignment.shape

    width = n_phonemes / 4 + 1.5
    fig, ax = plt.subplots(figsize=(width, width / 1.5), dpi=dpi, facecolor="w")  # type: ignore

    im = ax.imshow(alignment, origin="lower", cmap="viridis", aspect="auto")
    im.set_clim(0.0, 1.0)

    # set matplotlib colorbar size to match graph
    # https://stackoverflow.com/a/18195921
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.25, pad=0.05)
    fig.colorbar(im, cax=cax)

    # disable default black borders
    for spine in ax.spines.values():
        spine.set_color((0.0, 0.0, 0.0, 0.0))  # transparent RGBA

    ax.set_ylabel("spectrogram frames", fontsize="large")
    ax.tick_params(axis="y", which="major", length=10.0)

    if phonemes:
        # bars
        phoneme_bars_ticks = [-0.5]
        for i, _ in enumerate(phonemes):
            phoneme_bars_ticks.append(float(i) + 0.5)

        ax.set_xticks(ticks=phoneme_bars_ticks, minor=False)
        ax.set_xticklabels(labels=("" for _ in phoneme_bars_ticks), minor=False)
        ax.tick_params(axis="x", which="major", length=20.0)

        # labels
        phoneme_ticks = [i for i in range(len(phonemes))]
        ax.set_xticks(ticks=phoneme_ticks, minor=True)
        ax.set_xticklabels(labels=phonemes, minor=True)
        ax.tick_params(
            axis="x", which="minor", color=(0.0, 0.0, 0.0, 0.0), labelsize="large"
        )

    fig.tight_layout()

    fig.canvas.draw()
    plt.close(fig)

    return fig


def plot_statistics(e_pred=None, e_gt=None, p_pred=None, p_gt=None):
    fig, ax = plt.subplots(figsize=(12, 3))  # type: ignore

    y_min = min(p_gt.min(), p_pred.min())
    y_max = max(p_gt.max(), p_pred.max())

    ax.plot(p_gt, "b", linewidth=3, alpha=0.5)
    ax.plot(p_pred, "b--", linewidth=3, alpha=0.5)

    ax.set_ylim(y_min, y_max)
    ax.tick_params(
        labelsize="x-small",
        colors="tomato",
        bottom=False,
        labelbottom=False,
        left=True,
        labelleft=True,
    )
    ax.set_ylabel("F0", color="tomato")
    ax.yaxis.set_label_position("left")

    y_min = min(e_gt.min(), e_pred.min())
    y_max = min(e_gt.max(), e_pred.max())

    ax2 = ax.twinx()
    ax2.plot(e_gt, "r", linewidth=3, alpha=0.5)
    ax2.plot(e_pred, "r--", linewidth=3, alpha=0.5)

    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")

    fig.canvas.draw()
    data = figure_to_ndarray(fig)
    plt.close()

    return data


def plot_spectrogram(
    spectrogram: npt.NDArray,
    phonemes: tp.Optional[tp.List[str]] = None,
    phonemes_ticks: tp.Optional[tp.List[float]] = None,
    signal: tp.Optional[tp.Union[npt.NDArray, tp.Dict[str, npt.NDArray]]] = None,
    limits: tp.Optional[tp.Tuple[float, float]] = None,
    dpi: int = 80,
    **kwargs,
) -> tp.Union[plt.Figure, None]:
    """Plot a spectrogram.

    :param spectrogram: (n_bands, n_frames)
    :param phonemes: list of phoneme_symb
    :param phonemes_ticks: ticks e.g. from :func:`phonemes_to_frame_ticks`
    :param signal: 1d signal
    :param limits: spectrogram value limits
    :param dpi: DPI
    :return: a matplotlib figure

    """
    n_bands, n_frames = spectrogram.shape

    w, h = n_frames / 16, n_bands / 24
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi, facecolor="w")  # type: ignore

    im = ax.imshow(spectrogram, origin="lower", aspect="auto", cmap="viridis")
    if limits:
        im.set_clim(*limits)

    # set matplotlib colorbar size to match graph
    # https://stackoverflow.com/a/18195921
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.25, pad=0.5)
    fig.colorbar(im, cax=cax, ax=ax)

    # disable default black borders
    for spine in ax.spines.values():
        spine.set_color((0.0, 0.0, 0.0, 0.0))  # transparent RGBA

    ax.set_title("")
    ax.set_ylabel("mel bands", fontsize="large")
    ax.set_xlabel("frames", fontsize="large")

    ax.tick_params(axis="x", which="major", pad=20.0)
    ax.tick_params(
        axis="x", which="minor", length=2.0, color=(1.0,) * 4, labelsize="large"
    )

    if signal:
        if not isinstance(signal, dict):
            signal = {"signal": signal}

        for name, s in signal.items():
            ax.plot(
                np.arange(spectrogram.shape[1]),
                s,
                c=np.random.rand(3).tolist(),
                lw=1,
                label=name,
            )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small")

    if phonemes and phonemes_ticks:
        ax.set_xticks(phonemes_ticks)
        labels = ax.set_xticklabels(phonemes)
        for i, label in enumerate(labels):
            label.set_y(label.get_position()[1] - (i % 2) * 0.075)

    if phonemes_ticks:
        if isinstance(phonemes_ticks, tp.Dict):
            colors = ["r", "b", "k"]
            t = 3 * len(phonemes_ticks)
            all_xticks = []
            all_labels = []
            for i, (name, d) in enumerate(phonemes_ticks.items()):
                ax.vlines(
                    d,
                    -t,
                    n_bands + t,
                    color=colors[i],
                    linewidth=1,
                    linestyle="--",
                    label=name,
                )
                t -= 3
                all_xticks += d
                all_labels += [f"{colors[i]}{t}" for t in range(len(d))]

            ax.set_xticks(all_xticks)
            labels = ax.set_xticklabels(all_labels)
            for i, label in enumerate(labels):
                y = colors.index(label._text[0])
                label.set_y(label.get_position()[1] - y * 0.075)

            ax.legend(
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small"
            )
        else:
            ax.vlines(
                phonemes_ticks, -2, n_bands + 2, color="r", linewidth=1, linestyle="--"
            )

    fig.tight_layout()

    if not kwargs.get("dont_close", False):
        fig.canvas.draw()
        plt.close(fig)
        return fig
    else:
        plt.show()

    return None


def plot_durations_and_signals(spec, dura=None, symbols=None, signal=None):
    import torch
    import matplotlib

    matplotlib.use("TkAgg")

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy().transpose()

    if not isinstance(signal, dict):
        signal = {"signal": signal}

    for name, val in signal.items():
        val = val.cpu().numpy()[: spec.shape[1]]
        val = val / val.max() * (spec.shape[0] // 2)
        signal[name] = val

    if dura is not None:
        dura = dura.cpu().cumsum(0).long().numpy().tolist()
        plot_spectrogram(spec, symbols, dura, signal=signal, dont_close=True)
    else:
        plot_spectrogram(spec, signal=signal, dont_close=True)


def phonemes_to_frame_ticks(
    alignment: npt.NDArray, phonemes: tp.List[str]
) -> tp.List[float]:
    """Compute rough alignment between phoneme_symb and spectrogram frames.

    :param alignment: (n_frames, n_phonemes)
    :param phonemes: list of phoneme_symb
    :return: list with a frame number (possibly fractional) for each phoneme_symb

    """
    frame_ticks = []
    for i, phoneme in enumerate(phonemes):
        attention = alignment[:, i]
        frame_number = attention.argmax()
        frame_ticks.append(float(frame_number))

    return frame_ticks


def phonemes_to_frame_ticks_with_durations(
    durations: npt.NDArray, phonemes: tp.List[str]
) -> tp.List[float]:
    """Compute phoneme_symb-spectrogram alignment based on predicted phoneme_symb
    duration.

    :param durations: (durations in timesteps of decoder)
    :param phonemes: list of phoneme_symb
    :return: list with a frame number (possibly fractional) for each phoneme_symb

    """
    frame_ticks = []
    sum_ticks = 0
    for i in range(min(len(durations), len(phonemes))):
        sum_ticks += np.ceil(durations[i])
        frame_ticks.append(float(sum_ticks))
    for i in range(np.abs(len(durations) - len(phonemes))):
        frame_ticks.append(float(sum_ticks))

    return frame_ticks
