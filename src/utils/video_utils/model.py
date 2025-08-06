import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoCLIP(nn.Module):
    """
    A wrapper class for CLIP that extends its functionality to handle video data.
    
    This class provides methods to encode text and video data using the CLIP model, 
    and to perform a forward pass that returns a dictionary of features and scaling factors.

    Attributes:
        clip_model (torch.nn.Module): The pre-trained CLIP model used for encoding text and video frames.
    """
    
    def __init__(self, clip_model):
        """
        Initializes the VideoCLIP class with a pre-trained CLIP model.

        Args:
            clip_model (torch.nn.Module): A pre-trained CLIP model (e.g., from open_clip).
        """
        super().__init__()
        self.clip_model = clip_model
        self.transformer = clip_model.transformer
        self.visual = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Encodes the input text using the CLIP model's text encoder.

        Args:
            text (torch.Tensor): A tensor containing tokenized text input.

        Returns:
            torch.Tensor: A tensor representing the encoded text features.
        """
        return self.clip_model.encode_text(text, normalize)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of videos by processing each frame using the CLIP model's image encoder,
        and then averaging the frame-level embeddings to produce a single video-level embedding.

        Args:
            video (torch.Tensor): A tensor of shape (batch_size, num_frames, channels, height, width)
                                  representing a batch of videos.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) where each entry is the 
                          averaged embedding of the frames in a video.
        """
        batch_size = video.shape[0]

        # Reshape the video tensor to process each frame individually.
        # Shape: (batch_size * num_frames, channels, height, width)
        images = video.view(-1, *video.shape[2:])

        # Encode each frame using the CLIP model's image encoder.
        # Shape: (batch_size * num_frames, embedding_dim)
        image_features = self.clip_model.encode_image(images)

        # Normalize each frame embedding to have unit length.
        # Shape: (batch_size * num_frames, embedding_dim)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Reshape the encoded image tensor back to (batch_size, num_frames, embedding_dim).
        # Shape: (batch_size, num_frames, embedding_dim)
        image_features = image_features.view(batch_size, -1, *image_features.shape[1:])

        # Average the frame embeddings to get a single video embedding for each video.
        # Shape: (batch_size, embedding_dim)
        return image_features.mean(dim=1)

    def encode_image(self, video: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Encodes a batch of videos, allowing this method to be used interchangeably with `encode_video`.
        
        This method internally calls `encode_video` to ensure compatibility with the expected API.

        Args:
            video (torch.Tensor): A tensor of shape (batch_size, num_frames, channels, height, width)
                                  representing a batch of videos.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) where each entry is the 
                          averaged embedding of the frames in a video.
        """
        features = self.encode_video(video)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, video: torch.Tensor, text: torch.Tensor):
        """
        Computes the encoded features for video and text inputs.

        Args:
            video (torch.Tensor): A tensor of shape (batch_size, num_frames, channels, height, width)
                                  representing a batch of videos.
            text (torch.Tensor): A tensor containing tokenized text input.

        Returns:
            dict: A dictionary containing:
                  - "image_features": The encoded video features (from frame averaging).
                  - "text_features": The encoded text features.
                  - "logit_scale": The logit scale factor from the CLIP model.
                  - "logit_bias" (optional): The logit bias if present in the CLIP model.
        """
        # Encode video (as image features) and text features
        image_features = self.encode_video(video)
        text_features = self.encode_text(text)

        # Prepare the output dictionary
        out_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.clip_model.logit_scale.exp()
        }

        # Include logit_bias if it is present in the CLIP model
        if hasattr(self.clip_model, 'logit_bias') and self.clip_model.logit_bias is not None:
            out_dict['logit_bias'] = self.clip_model.logit_bias

        return out_dict


    def __call__(self, video: torch.Tensor, text: torch.Tensor):
        """
        Allows the model instance to be called directly with inputs, 
        simulating the behavior of invoking the forward method.

        Args:
            video (torch.Tensor): A tensor of shape (batch_size, num_frames, channels, height, width)
                                  representing a batch of videos.
            text (torch.Tensor): A tensor containing tokenized text input.

        Returns:
            dict: The output from the forward method.
        """
        return self.forward(video, text)

