import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class MCQLoss(nn.Module):
    def __init__(self):
        super(MCQLoss, self).__init__()

    def forward(self, image_features, text_features, correct_answers, logit_scale):
        """
        Computes the loss for the Multiple-Choice Questions (MCQ) task.

        Args:
            image_features (Tensor): Image feature embeddings of shape (batch_size, feature_dim)
            text_features (Tensor): Text feature embeddings of shape (num_options, batch_size, feature_dim)
            correct_answers (Tensor): Correct answer indices of shape (batch_size,)
            logit_scale (Tensor): Scalar tensor to scale the logits.
        Returns:
            loss (Tensor): Scalar tensor
        """
        # TODO: which is the correct way to compute the logits?
        # # Compute logits without permutation
        # logits = torch.einsum('bf,nbf->bn', image_features, text_features) * logit_scale

        # Compute logits: similarity between image features and each caption option
        # Shape of logits: (batch_size, num_options)
        logits = torch.einsum('bf,bnf->bn', image_features, text_features) * logit_scale

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, correct_answers)

        return loss

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels) 
        ) / 2
        #TODO off_diagonal_term = logits_per_image dot text_features[quadrant2[diagonal]]

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class MaskedClipLoss(ClipLoss):
    def __init__(self, batch_size, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__(local_loss=local_loss, gather_with_grad=gather_with_grad, cache_labels=cache_labels, rank=rank, world_size=world_size, use_horovod=use_horovod)
        self.batch_size = batch_size
        self.mask = self.create_mask()  # Initialize the mask specific to this subclass

    def create_mask(self):
        # Ensure batch_size is even to split into quadrants easily
        assert self.batch_size % 2 == 0, "Batch size must be an even number."
        
        # Calculate the size of one side of a quadrant
        half_size = self.batch_size // 2
        
        # Create the masks for the four quadrants
        # Quadrant 1: all ones
        Q1 = torch.ones((half_size, half_size))
        
        # Quadrant 2 and 3: diagonal ones
        Q2 = torch.eye(half_size)
        Q3 = torch.eye(half_size)
        
        # Quadrant 4: all ones
        Q4 = torch.ones((half_size, half_size))
        
        # Construct the top half of the matrix (Q1 and Q2)
        top_half = torch.cat((Q1, Q2), dim=1)
        
        # Construct the bottom half of the matrix (Q3 and Q4)
        bottom_half = torch.cat((Q3, Q4), dim=1)
        
        # Combine top and bottom halves to form the full mask
        return torch.cat((top_half, bottom_half), dim=0)

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        self.mask = self.mask.to(device)
        logits_per_image = logits_per_image.masked_fill(self.mask == 0, float('-inf'))
        logits_per_text = logits_per_text.masked_fill(self.mask == 0, float('-inf'))


        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        # Apply the mask to logits before calculating the cross-entropy loss
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"total_loss": total_loss} if output_dict else total_loss

class DisparityMaximizingClipLoss(ClipLoss):
    def __init__(self, batch_size, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False, alpha=1.0, beta=1.0):
        """
        Initialize the disparity maximizing clip loss class.

        This loss function extends the standard contrastive loss used in CLIP models by incorporating
        additional terms that aim to maximize the disparity between mismatched (negative) image and text pairs.

        Args:
            batch_size (int): Batch size, must be an even number.
            local_loss (bool): Whether to use local loss calculation.
            gather_with_grad (bool): Whether to gather features with gradients.
            cache_labels (bool): Whether to cache labels.
            rank (int): Rank of the current process in distributed training.
            world_size (int): Number of processes in distributed training.
            use_horovod (bool): Whether to use Horovod for distributed training.
            alpha (float): Weight for the contrastive loss term.
            beta (float): Weight for the dot product loss term.
        """
        super().__init__(local_loss=local_loss, gather_with_grad=gather_with_grad, cache_labels=cache_labels, rank=rank, world_size=world_size, use_horovod=use_horovod)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        """
        Forward pass to compute the disparity maximizing contrastive loss.

        This loss comprises two main components:
        1. Standard contrastive loss between matching image and text pairs.
        2. Additional terms that aim to maximize the disparity between mismatched (negative) image and text pairs.

        Args:
            image_features (torch.Tensor): Image feature vectors.
            text_features (torch.Tensor): Text feature vectors.
            logit_scale (torch.Tensor): Scaling factor for logits.
            output_dict (bool): Whether to output a dictionary or the loss value.

        Returns:
            Union[torch.Tensor, dict]: The computed loss or a dictionary containing the loss.
        """
        device = image_features.device

        # Ensure batch size is even
        half_batch_size = self.batch_size // 2

        # Split image and text features into two halves
        image_features, image_features_star = torch.split(image_features, half_batch_size)
        text_features, text_features_star = torch.split(text_features, half_batch_size)

        # Compute logits for both halves
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        logits_per_image_star, logits_per_text_star = self.get_logits(image_features_star, text_features_star, logit_scale)

        # Get ground truth labels for both sets of logits
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        labels_star = self.get_ground_truth(device, logits_per_image_star.shape[0])

        # Compute cross-entropy loss for both sets of logits
        loss_1 = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        loss_2 = (F.cross_entropy(logits_per_image_star, labels_star) + F.cross_entropy(logits_per_text_star, labels_star)) / 2
        clip_loss = loss_1 + loss_2

        # Compute dot product losses to maximize disparity
        dot_products_1 = F.softmax(image_features @ text_features_star.T, dim=1).diag()
        dot_product_loss_1 = torch.log(dot_products_1).sum(dim=0)
        dot_products_2 = F.softmax(image_features_star @ text_features.T, dim=1).diag()
        dot_product_loss_2 = torch.log(dot_products_2).sum(dim=0)
        dot_product_loss = dot_product_loss_1 + dot_product_loss_2 # TODO: Is this supposed to be negative?

        # Combine losses with weights
        total_loss = self.alpha * clip_loss + self.beta * dot_product_loss

        if output_dict:
            return {"contrastive_loss": clip_loss, "negation_loss": dot_product_loss, "total_loss": total_loss}
        else:
            return clip_loss, dot_product_loss, total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
