import logging
from abc import ABC, abstractmethod
from typing import Sequence, Union
from os import PathLike

import decord
import numpy as np
import torch
import torchvision.datasets
from overrides import overrides

LOGGER = logging.getLogger(__name__)
# See https://stackoverflow.com/a/53418245/1165181
TYPE_PATH = Union[PathLike, str, bytes]


class VideoReader(ABC):
    def __init__(self, path: TYPE_PATH) -> None:  # noqa
        pass

    def __call__(self, indices: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def time_to_indices(self, time: Union[float, Sequence[float]]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_avg_fps(self) -> float:
        raise NotImplementedError

    @staticmethod
    def from_path(path: TYPE_PATH) -> "VideoReader":
        return (AccImageVideoReader if torchvision.datasets.folder.is_image_file(path) else DecordVideoReader)(path)


decord.bridge.set_bridge("torch")


class DecordVideoReader(VideoReader):
    @overrides
    def __init__(self, path: TYPE_PATH) -> None:
        super().__init__(path)
        # Using `width` and `height` from VideoReader is actually faster because it resizes while decoding, however
        # it doesn't preserve the aspect ratio (even if setting only one of the two).
        # Using the GPU for decoding may actually be faster, but it isn't trivial how to optimize the whole data loading
        # process so to accomplish it.
        try:
            self.video_reader = decord.VideoReader(path, num_threads=1)
        except decord.DECORDError:
            LOGGER.error(f"An error occurred when trying to load the video with path {path}.")
            self.video_reader = None

    @overrides
    def __call__(self, indices: Sequence[int]) -> torch.Tensor:
        if self.video_reader:
            try:
                return self.video_reader.get_batch(indices)  # noqa
            except decord.DECORDError:
                # FIXME: change the handle for the path? Or how to get the path
                LOGGER.error(f"An error occurred when trying to read the video with path {self.video_reader._handle}"
                             f" and indices {indices}.")

        return torch.zeros(len(indices), 256, 256, 3)

    @overrides
    def __len__(self) -> int:
        return len(self.video_reader) if self.video_reader else 1

    @overrides
    def time_to_indices(self, time: Union[float, Sequence[float]]) -> np.ndarray:
        times = self.video_reader.get_frame_timestamp(range(len(self))).mean(-1) if self.video_reader else np.zeros(1)
        indices = np.searchsorted(times, time)
        # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
        return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                        indices - 1)

    @overrides
    def get_avg_fps(self) -> float:
        return self.video_reader.get_avg_fps() if self.video_reader else 1