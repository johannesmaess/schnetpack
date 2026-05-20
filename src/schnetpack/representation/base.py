import torch
from torch import nn
import schnetpack.properties as properties
from schnetpack.nn.base import _as_module
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
        # cutoff_fn is always an nn.Module in practice (CosineCutoff/MollifierCutoff);
        # keep the constructor lenient but store a Module so the class is script-able.
        self.cutoff_fn = cutoff_fn if cutoff_fn is not None else nn.Identity()
        self.cutoff = cutoff_fn.cutoff if cutoff_fn is not None else None
        # `activation` is sometimes passed as a plain callable (e.g. F.silu);
        # wrap so subclasses that store it on `self` are script-able.
        self.activation = _as_module(activation)

        # Initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding

        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings) # not in fs
    
    def atom_embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Atom-level embedding from atomic numbers (and electronic embeddings if any).

        Subclasses may override to shape the result as the architecture's z-state
        (e.g. PaiNN pads scalar+vector slots, SO3net pads to (Lmax+1)^2 spherical
        harmonics). The base implementation returns the bare atom embedding tensor
        of shape (n_atoms, n_atom_basis), which is what SchNet consumes directly.
        """
        atomic_numbers = inputs[properties.Z]
        x = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)
        return x

    def geom_embed(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Precompute geometry-derived tensors that the per-layer interact loop needs.

        Output is an architecture-specific Dict[str, Tensor] passed to interact.
        Lifting this out of the per-layer loop is what enables hoisting it out of
        the DEQ fixed-point loop too — geometry only changes when positions change.
        """
        raise NotImplementedError

    def interact(self, geom: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Run the per-layer interaction loop given precomputed geometry features."""
        raise NotImplementedError

    def save(self, inputs: Dict[str, torch.Tensor], x: torch.Tensor):
        """Save the atomic embeddings."""
        raise NotImplementedError

    def forward(self, inputs: Dict[str, torch.Tensor]):
        x = self.atom_embed(inputs)
        geom = self.geom_embed(inputs)
        x = self.interact(geom, x)
        return self.save(inputs, x)