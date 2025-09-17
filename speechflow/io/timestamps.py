import typing as tp

from copy import deepcopy
from copy import deepcopy as copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Timestamps"]


@dataclass
class Timestamps:
    intervals: npt.NDArray

    def __post_init__(self):
        self.intervals = np.asarray(self.intervals, dtype=np.float64)

        if self.intervals.ndim != 2:
            raise ValueError(
                f"Incorrect shape ({self.intervals.shape}) of tstamps: should be (x, 2)"
            )

        min_diff = (self.intervals[:, 1] - self.intervals[:, 0]).min()
        if min_diff < 0:
            raise ValueError(f"timestamp interval with {min_diff} duration is found.")

        if len(self) > 1:
            diff = self.intervals[1:, 0] - self.intervals[:-1, 1]
            if np.round(diff.min(), 4) < 0:
                raise ValueError("Back to the future issue is found!")

    def __len__(self):
        return self.intervals.shape[0]

    def __getitem__(self, item: tp.Union[tp.SupportsIndex, slice]):
        return self.intervals[item]

    def __add__(self, offset: float):
        temp = copy(self)
        temp.intervals += offset
        return temp

    def __sub__(self, offset: float):
        return self + (-offset)

    @staticmethod
    def from_durations(durations: npt.NDArray) -> "Timestamps":
        cumsum = np.insert(np.cumsum(durations), 0, 0)
        return Timestamps(intervals=np.stack([cumsum[:-1], cumsum[1:]]).T)

    @staticmethod
    def from_list(
        array_list: tp.Union[
            tp.List[tp.Tuple[float, float]], tp.List[npt.NDArray], tp.List["Timestamps"]
        ]
    ) -> "Timestamps":
        if isinstance(array_list[0], tuple):
            return Timestamps(np.asarray(array_list, dtype=np.float64))
        else:
            return Timestamps(np.concatenate(array_list))

    @property
    def begin(self) -> float:
        return self[0][0]

    @property
    def end(self) -> float:
        return self[-1][1]

    @property
    def duration(self) -> float:
        return self.end - self.begin

    def copy(self) -> "Timestamps":
        return deepcopy(self)

    def append(self, ts: tp.Union[tp.List[tp.Tuple[float, float]], npt.NDArray]) -> None:
        ts = np.asarray(ts).astype(self.intervals.dtype)
        assert self.intervals[-1][0] < ts[0][0]
        if self.intervals[-1][1] > ts[0][0]:
            self.intervals[-1][1] = ts[0][0]
            assert self.intervals[-1][1] > self.intervals[-1][0]
        self.intervals = np.vstack((self.intervals, ts))

    def append_left(
        self, ts: tp.Union[tp.List[tp.Tuple[float, float]], npt.NDArray]
    ) -> None:
        ts = np.asarray(ts).astype(self.intervals.dtype)
        assert self.intervals[0][1] > ts[0][1]
        if self.intervals[0][0] <= ts[0][1]:
            self.intervals[0][0] = ts[0][1]
            assert self.intervals[0][0] < self.intervals[0][1]
        self.intervals = np.vstack((ts, self.intervals))

    def delete(self, index: int) -> None:
        self.intervals = np.delete(self.intervals, index, axis=0)

    def to_secs(self, sample_rate: int) -> "Timestamps":
        return Timestamps(self.intervals.astype(float) / sample_rate)

    def to_samples(self, sample_rate: int) -> "Timestamps":
        return Timestamps((self.intervals * sample_rate).astype(int))

    def to_durations(self) -> npt.NDArray:
        return np.diff(self.intervals, axis=1)[:, 0]

    def to_frames(
        self, hop_len: float, num_frames: int, as_int: bool = True
    ) -> "Timestamps":
        """Assumes timestamps array contains (len(text) + 1) numbers and the last
        timestamp stands for last sample (=wave length)."""
        if not as_int:
            return Timestamps(self.intervals / hop_len)

        frame_stamps = [hop_len * (i + 1) for i in range(num_frames)]
        ts_frame = [int(self.begin / hop_len)]
        previous = -1
        expand_count, succeeding_count, max_expand_count = 0, 0, 8
        for _, b in self.intervals:  # O(n^2) hack
            closest = None
            closest_value = self.end
            for i, x in enumerate(frame_stamps[max(previous, 0) :], max(previous, 0)):
                if i < previous:
                    continue
                if abs(x - b) <= closest_value:
                    closest = i
                    closest_value = abs(x - b)
                else:
                    break

            if closest == previous:
                closest = min(closest + 1, len(frame_stamps) - 1)
                expand_count += 1
                succeeding_count += 1
                assert (
                    succeeding_count <= max_expand_count
                    and expand_count <= max_expand_count * 2
                ), f"More than {max_expand_count} short phonemes are not allowed, got {succeeding_count} in a row and total {expand_count}! "
            else:
                succeeding_count = 0

            if closest is None:
                raise RuntimeError("error fix timestamp!")

            previous = closest
            ts_frame.append(closest + 1)

        assert np.abs(ts_frame[-1] - num_frames) < 2

        ts_frame[-1] = min(ts_frame[-1], num_frames)

        # if there is no frames left for the last phoneme, cut single frame from phonemes on the left.
        if ts_frame[-1] == ts_frame[-2] and len(ts_frame) > 2:
            max_idx = len(ts_frame) - 1
            # more than ten 1-frame phonemes indicate broken alignment so don't seek further.
            for j in range(1, min(10, max_idx - 1)):
                if ts_frame[max_idx - j] - ts_frame[max_idx - j - 1] > 1:
                    # if long phoneme was found cut single frame and make shift for all successive phonemes.
                    for k in range(1, j + 1):
                        ts_frame[max_idx - k] -= 1
                    break

        ts_frame = list(zip(ts_frame[:-1], ts_frame[1:]))
        assert len(ts_frame) == len(self)

        return Timestamps(ts_frame)

    def shift(self, index: int, duration: float) -> None:
        if duration == 0.0:
            return

        ts_left = Timestamps(self[:index])
        ts_right = Timestamps(self[index:])

        if duration > 0:
            duration = min(duration, ts_right.duration * 0.99)
        if duration < 0:
            duration = max(duration, -ts_left.duration * 0.99)

        dura_left = ts_left.to_durations()
        dura_right = ts_right.to_durations()

        dura_left *= (ts_left.duration + duration) / ts_left.duration
        dura_right *= (ts_right.duration - duration) / ts_right.duration

        new_ts = Timestamps.from_durations(np.concatenate([dura_left, dura_right]))
        new_ts += self.begin

        assert new_ts.duration == self.duration
        self.intervals = new_ts.intervals
