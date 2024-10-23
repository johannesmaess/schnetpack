# define a module to compute scalar value from smallest eigenvalue of correlation matrix of scalar_representations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
import schnetpack.nn as snn
from typing import Sequence, Union, Callable, Dict, Optional
import schnetpack as spk

class Correlation(nn.Module):
    def __init__(self,
                 output_key: str = "y",
                 ):
        super(Correlation, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = self.calculate_correlation_lowest_eigenvalues(inputs) 
        return inputs

    def calculate_correlation_lowest_eigenvalues(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # compute scalar representations
        sr = inputs['scalar_representation']
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        
        # get submatrices corresponding to each molecule
        split_idx = torch.bincount(idx_m)
        sr_per_molecule = torch.split(sr, split_idx.tolist())
        
        # Note: eigvalsh also supports batched computation.
        # If the molecules in batch have the same number of atoms N (or every matrix is reduced to the size of the feature dimension F),
        # the computation can be done in parallel by `eigvalsh(sr.view(batch_size, ...))`.
        lowest_eigenvalues = torch.zeros(maxm, dtype=sr.dtype, device=sr.device)
        for i, sr_for_molecule in enumerate(sr_per_molecule):
            # compute minimal eigenvalue of correlation matrix
            c = sr_for_molecule @ sr_for_molecule.T
            min_eigval = torch.linalg.eigvalsh(c)[0]
            lowest_eigenvalues[i] = min_eigval
        return lowest_eigenvalues
    

class FCorrelation(Correlation):
    """
    n_in: input dimension
    """
    def __init__(self,
                 n_in: int,
                 n_out: int = 1,
                 n_hidden: Optional[Union[int, Sequence[int]]] = None,
                 n_layers: int = 2,
                 activation: Callable = F.silu,
                 aggregation_mode: str = "sum",
                 output_key: str = "y",
                 ):
        super(FCorrelation, self).__init__(output_key=output_key)
        self.n_in = n_in

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode


    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tmp = self.calculate_correlation_lowest_eigenvalues(inputs)
        
        # predict atomwise contributions
        y = self.outnet(tmp)
        
        inputs[self.output_key] = torch.squeeze(y, -1)
        return inputs