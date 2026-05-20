from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
import schnetpack.nn.so3 as so3
import schnetpack.properties as properties
from schnetpack.nn import ElectronicEmbedding
from schnetpack.representation.base import AtomisticRepresentation

__all__ = ["SO3net"]


class SO3net(AtomisticRepresentation):
    """
    A simple SO3-equivariant representation using spherical harmonics and
    Clebsch-Gordon tensor products.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        lmax: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        shared_interactions: bool = False,
        return_vector_representation: bool = False,
        activation: Optional[Callable] = F.silu,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            lmax: maximum angular momentum of spherical harmonics basis
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            shared_interactions:
            return_vector_representation: return l=1 features in Cartesian XYZ order
                (e.g. for DipoleMoment output module)
            nuclear_embedding: custom nuclear embedding (e.g. spk.nn.embeddings.NuclearEmbedding)
            electronic_embeddings: list of electronic embeddings. E.g. for spin and
                charge (see spk.nn.embeddings.ElectronicEmbedding)
        """
        AtomisticRepresentation.__init__(
            self,
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            activation=activation,
            nuclear_embedding=nuclear_embedding,
            electronic_embeddings=electronic_embeddings,
        )
        self.lmax = lmax
        self.return_vector_representation = return_vector_representation

        # initialize shperical harmonics
        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        # initialize filters
        self.so3convs = snn.replicate_module(
            lambda: so3.SO3Convolution(lmax, self.n_atom_basis, self.radial_basis.n_rbf),
            self.n_interactions,
            shared_interactions,
        )

        # initialize interaction blocks
        self.mixings1 = snn.replicate_module(
            lambda: nn.Linear(self.n_atom_basis, self.n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings2 = snn.replicate_module(
            lambda: nn.Linear(self.n_atom_basis, self.n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings3 = snn.replicate_module(
            lambda: nn.Linear(self.n_atom_basis, self.n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.gatings = snn.replicate_module(
            lambda: so3.SO3ParametricGatedNonlinearity(self.n_atom_basis, lmax),
            self.n_interactions,
            shared_interactions,
        )
        self.so3product = so3.SO3TensorProduct(lmax)


    def atom_embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pad the atomic embedding to (n_atoms, (Lmax+1)^2, n_z) — scalar block + zero L>0 blocks."""
        atomic_numbers = inputs[properties.Z]
        x0 = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            x0 = x0 + embedding(x0, inputs)
        x0 = x0.unsqueeze(1)
        return so3.scalar2rsh(x0, int(self.lmax))

    def geom_embed(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Precompute spherical harmonics, radial basis, and cutoff envelope."""
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        return {
            'Yij': self.sphharm(dir_ij),
            'radial_ij': self.radial_basis(d_ij),
            'cutoff_ij': self.cutoff_fn(d_ij)[..., None],
            'idx_i': inputs[properties.idx_i],
            'idx_j': inputs[properties.idx_j],
        }

    def interact(self, geom: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        Yij = geom['Yij']
        radial_ij = geom['radial_ij']
        cutoff_ij = geom['cutoff_ij']
        idx_i = geom['idx_i']
        idx_j = geom['idx_j']

        for so3conv, mixing1, mixing2, gating, mixing3 in zip(
            self.so3convs, self.mixings1, self.mixings2, self.gatings, self.mixings3
        ):
            dx = so3conv(x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
            ddx = mixing1(dx)
            dx = dx + self.so3product(dx, ddx)
            dx = mixing2(dx)
            dx = gating(dx)
            dx = mixing3(dx)
            x = x + dx

        return x

    def save(self, inputs: Dict[str, torch.Tensor], x: torch.Tensor):
        # collect results
        inputs["scalar_representation"] = x[:, 0]
        inputs["multipole_representation"] = x
        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.return_vector_representation:
            inputs["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        return inputs
