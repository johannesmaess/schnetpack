import torch
from torch import nn
import schnetpack.properties as properties
from typing import Dict, Optional, List, Callable

__all__ = ["AtomisticRepresentation"]

class AtomisticRepresentation(nn.Module):
    """
    Base class for different neural network architectures that represent atomic interactions.
    This class generalizes shared arguments across different architectures such as SO3net, PaiNN, SchNet, and FieldSchNet.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = nn.functional.silu,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List[nn.Module]] = None,
    ):
        """
        Args:
            n_atom_basis: Number of features to describe atomic environments.
            n_interactions: Number of interaction blocks.
            radial_basis: Layer for expanding interatomic distances in a basis set.
            cutoff_fn: Function to apply cutoff in the model.
            activation: Activation function to use in the network.
            nuclear_embedding: Custom embedding for nuclear features.
            electronic_embeddings: List of electronic embeddings, such as for spin or charge.
        """
        nn.Module.__init__(self)

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff if cutoff_fn else None
        self.activation = activation

        # Initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings) # not in fs
    
    def embed(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic embeddings.
        """
        atomic_numbers = inputs[properties.Z]
        x = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)
        return x
    
    def interact(self, inputs: Dict[str, torch.Tensor], x: torch.Tensor):
        """
        Compute interaction blocks and update atomic embeddings.
        """
        raise NotImplementedError
    
    def save(self, inputs: Dict[str, torch.Tensor], x: torch.Tensor):
        """
        Save the atomic embeddings.
        """
        raise NotImplementedError
    
    def forward(self, inputs: Dict[str, torch.Tensor]):
        x = self.embed(inputs)
        x = self.interact(inputs, x)
        return self.save(inputs, x)