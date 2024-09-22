from typing import Callable, Dict, Union, Optional, List
import torch
from torch import nn
import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
from schnetpack.representation.base import AtomisticRepresentation

import schnetpack.nn as snn


__all__ = ["SchNet", "SchNetInteraction"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x


class SchNet(AtomisticRepresentation):
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        shared_interactions: bool = False,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            activation: activation function
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
        self.n_filters = n_filters or self.n_atom_basis

        # initialize interaction blocks
        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.activation,
            ),
            self.n_interactions,
            shared_interactions,
        )
    
    def interact(self, inputs: Dict[str, torch.Tensor], x: torch.Tensor):
        """
        Compute interaction blocks and update atomic embeddings.
        """
        # get tensors from input dictionary
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)
        
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v
        return x
    
    def save(self, inputs, x):
        inputs['scalar_representation'] = x
        return inputs
