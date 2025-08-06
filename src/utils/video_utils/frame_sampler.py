import itertools
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from overrides import overrides

from .utils import pairwise, resample


class FrameSampler(ABC):
    """Returns the frame indices to seek for the given clip start and end frame indices."""

    @abstractmethod
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        raise NotImplementedError


class RandomFromUniformIntervalsFrameSampler(FrameSampler):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.max_frames = max_frames

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
        return [torch.randint(a, b + 1, size=()) for a, b in pairwise(ticks)]


class UniformFrameSampler(FrameSampler):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.max_frames = max_frames

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
        return [torch.round((a + b) / 2).to(torch.int) for a, b in pairwise(ticks)]


class FixedFrameFromUniformIntervalsFrameSampler(FrameSampler):
    def __init__(self, max_frames: int, frame_index_from_interval_start: int) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.frame_index_from_interval_start = frame_index_from_interval_start

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame + 1, steps=num_frames + 1, dtype=torch.int)
        return ticks[:-1] + self.frame_index_from_interval_start


class ConsecutiveFrameSampler(FrameSampler):
    def __init__(self, max_frames: int, fps: Optional[int] = None) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.fps = fps

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        if self.fps:
            indices = resample(num_frames=self.max_frames, original_fps=fps, new_fps=self.fps)
        else:
            indices = range(self.max_frames)

        smallest_possible_end = min(end_frame, start_frame + indices[-1])

        if isinstance(smallest_possible_end, torch.Tensor):
            smallest_possible_end = smallest_possible_end.item()  # To avoid a warning in the floor division.
        start = start_frame + (end_frame - smallest_possible_end) // 2

        return list(itertools.takewhile(lambda i: i <= end_frame, (start + i for i in indices)))