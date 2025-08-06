from typing import Sequence, Iterable, Tuple, TypeVar
from torchvision.datasets.video_utils import VideoClips
import itertools

def resample(num_frames: int, original_fps: float, new_fps: float) -> Sequence[int]:
    """Returns essentially the same as `VideoClips._resample_video_idx`. Unlike it, it always checks for the max frames
    (the mentioned function doesn't do it when it returns a `slice`)."""
    indices = VideoClips._resample_video_idx(num_frames, original_fps, new_fps)

    if isinstance(indices, slice) and indices.stop is None:
        indices = range(*indices.indices((indices.start or 0) + num_frames * indices.step))

    return indices

# See https://docs.python.org/3/library/itertools.html#itertools-recipes
T = TypeVar("T")
def pairwise(it: Iterable[T]) -> Iterable[Tuple[T, T]]:
    a, b = itertools.tee(it)
    next(b, None)
    return zip(a, b)