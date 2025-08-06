import pandas as pd
from torch.utils.data import Dataset
from typing import List, Callable
import torch
from .video_reader import VideoReader
from .frame_sampler import FrameSampler, UniformFrameSampler
from abc import ABC, abstractmethod

class BaseCsvVideoDataset(Dataset, ABC):
    def __init__(self, csv_file: str, transforms: Callable, max_frames: int = 4, video_key='video_path'):
        """
        Base class for video dataset handling common functionality for loading, sampling,
        and transforming video frames.

        Args:
            csv_file (str): Path to the CSV file with annotations.
            transforms (callable): Transform to be applied on a sample of video frames.
            max_frames (int): Maximum number of frames to sample from each video. Defaults to 4.
            video_key (str): Column name for video file paths. Defaults to 'video_path'.
        """
        self.df = pd.read_csv(csv_file, sep=',')
        self.transforms = transforms
        self.frame_sampler = UniformFrameSampler(max_frames=max_frames)
        self.video_key = video_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row[self.video_key]

        # Load and sample frames from the video
        frames = self._load_and_sample_video(video_path)

        # Permute the frames to match PyTorch's (N, C, H, W) format
        frames = frames.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)

        # Apply transformations to the frames
        transformed_frames = self.transforms(frames)

        # Call the subclass-specific method to handle labels and captions
        return self.process_labels_and_captions(row, transformed_frames)

    def _load_and_sample_video(self, video_path: str) -> torch.Tensor:
        """
        Loads and samples frames from the video at the given path.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: A tensor containing the sampled frames from the video.
        """
        # Initialize a VideoReader for the given video path
        video_reader = VideoReader.from_path(video_path)

        # Get the start and end frames for sampling
        start_frame = 0
        end_frame = len(video_reader) - 1

        # Sample the frames using the provided FrameSampler
        frame_indices = self.frame_sampler(start_frame, end_frame, video_reader.get_avg_fps())

        # Retrieve the frames from the video
        frames = video_reader(frame_indices)
        
        return frames

    @abstractmethod
    def process_labels_and_captions(self, row: pd.Series, transformed_frames: torch.Tensor):
        """
        Abstract method to be implemented by subclasses to handle labels and captions.

        Args:
            row (pd.Series): A row from the DataFrame containing video metadata.
            transformed_frames (torch.Tensor): The transformed video frames.

        Returns:
            The output specific to the subclass, e.g., captions, correct answer, etc.
        """
        pass

class CsvVideoCaptionDataset(BaseCsvVideoDataset):
    def __init__(self, csv_file: str, transforms: Callable, max_frames: int = 4, video_key='video_path', caption_key='captions'):
        """
        Dataset for video captioning or retrieval tasks.

        Args:
            csv_file (str): Path to the CSV file with video paths and captions.
            transforms (callable): Transform to be applied on a sample of video frames.
            max_frames (int): Maximum number of frames to sample from each video. Defaults to 4.
            video_key (str): Column name for video file paths. Defaults to 'video_path'.
            caption_key (str): Column name for captions. Defaults to 'captions'.
        """
        super().__init__(csv_file, transforms, max_frames, video_key)
        self.caption_key = caption_key

    def process_labels_and_captions(self, row: pd.Series, transformed_frames: torch.Tensor):
        """
        Processes the captions for a video captioning task.

        Args:
            row (pd.Series): A row from the DataFrame containing video metadata.
            transformed_frames (torch.Tensor): The transformed video frames.

        Returns:
            Tuple[torch.Tensor, List[str]]: Transformed video frames and associated captions.
        """
        captions = eval(row[self.caption_key])
        return transformed_frames, captions

class CsvVideoMCQDataset(BaseCsvVideoDataset):
    def __init__(self, csv_file: str, transforms: Callable, max_frames: int = 4, num_answers: int = 4, video_key='video_path'):
        """
        Dataset for video multiple-choice question (MCQ) tasks.
        """
        super().__init__(csv_file, transforms, max_frames, video_key)
        self.num_answers = num_answers

    def process_labels_and_captions(self, row: pd.Series, transformed_frames: torch.Tensor):
        """
        Processes the captions and correct answer for a video MCQ task.

        Args:
            row (pd.Series): A row from the DataFrame containing video metadata.
            transformed_frames (torch.Tensor): The transformed video frames.

        Returns:
            Tuple[torch.Tensor, List[str], int, str]: Transformed video frames, captions (answer choices),
                                                      correct answer index, and correct answer template.
        """
        captions = [row[f"caption_{i}"] for i in range(self.num_answers)]
        correct_answer = row["correct_answer"]
        correct_answer_template = row["correct_answer_template"]
        return transformed_frames, captions, correct_answer, correct_answer_template