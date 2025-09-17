import typing as tp

from dataclasses import dataclass
from itertools import combinations, product

import torch
import torch.nn as nn

__all__ = ["VectorQuantizer", "VectorQuantizerOutput"]


@dataclass
class VectorQuantizerOutput:
    content: tp.Union[tp.List[torch.Tensor], torch.Tensor]
    additional_content: tp.Dict = None  # type: ignore
    additional_losses: tp.Dict = None  # type: ignore


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 128,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = codebook_size

        self.codebook = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.codebook.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

        self._commitment_cost = commitment_cost

    def forward(
        self,
        x,
        compute_distances_if_possible: bool = False,
        record_codebook_stats: bool = False,
    ) -> VectorQuantizerOutput:
        """Connects the module to some inputs.

        Args:
            x: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances

        """

        # Convert inputs from BCHW -> BHWC
        x = x.transpose(1, 2).contiguous()
        input_shape = x.shape
        batch_size, time, _ = input_shape
        device = x.device

        # Flatten input
        flat_input = x.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
        )

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, dtype=torch.float
        ).to(device)
        encodings.scatter_(1, encoding_indices, 1)

        # Compute distances between encoding vectors
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [
                torch.dist(items[0], items[1], 2).to(device)
                for items in combinations(flat_input, r=2)
            ]
            encoding_distances = (
                torch.tensor(_encoding_distances).to(device).view(batch_size, -1)
            )
        else:
            encoding_distances = None

        # Compute distances between embedding vectors
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [
                torch.dist(items[0], items[1], 2).to(device)
                for items in combinations(self.codebook.weight, r=2)
            ]
            embedding_distances = torch.tensor(_embedding_distances).to(device)
        else:
            embedding_distances = None

        # Sample nearest embedding
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [
                torch.dist(items[0], items[1], 2).to(device)
                for items in product(flat_input, self.codebook.weight.detach())
            ]
            frames_vs_embedding_distances = (
                torch.tensor(_frames_vs_embedding_distances)
                .to(device)
                .view(batch_size, time, -1)
            )
        else:
            frames_vs_embedding_distances = None

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = (
            self.codebook.weight[torch.argmin(distances, dim=1).detach().cpu()]
            if not self.training or record_codebook_stats
            else None
        )

        # Losses
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + commitment_loss

        quantized = (
            x + (quantized - x).detach()
        )  # Trick to prevent backpropagation of quantized
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )  # Exponential entropy

        losses = {"vq_loss": vq_loss}
        constants = {
            "constant_e_latent_loss": e_latent_loss,
            "constant_q_latent_loss": q_latent_loss,
            "constant_commitment_loss": commitment_loss,
        }
        losses.update(constants)

        return VectorQuantizerOutput(
            content=quantized.transpose(2, 1).contiguous(),
            additional_content={
                "perplexity": perplexity,
                "distances": distances.view(batch_size, time, -1),
                "encodings": encodings.view(batch_size, time, -1),
                "encoding_indices": encoding_indices.view(batch_size, time, -1),
                "encoding_distances": encoding_distances,
                "embedding_distances": embedding_distances,
                "frames_vs_embedding_distances": frames_vs_embedding_distances,
                "concatenated_quantized": concatenated_quantized,
                # "quantized": quantized.contiguous(),
            },
            additional_losses=losses,
        )

    def inference(self, embeddings) -> VectorQuantizerOutput:
        device = embeddings.device

        # embeddings_mask = embeddings #[mask]
        flat_input = embeddings.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
        )

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, dtype=torch.float
        ).to(device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        output_shape = torch.Size([1, encodings.shape[0], 256])
        quantized = torch.matmul(encodings, self.codebook.weight).view(
            output_shape
        )  # TODO input_shape
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        return VectorQuantizerOutput(
            content=quantized.transpose(2, 1).contiguous(),
            additional_content={},
            additional_losses={},
        )

    @property
    def embedding(self):
        return self.codebook
