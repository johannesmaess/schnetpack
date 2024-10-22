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
        # compute scalar representations
        sr = inputs['scalar_representation']
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        tmp = torch.zeros(maxm, dtype=sr.dtype, device=sr.device)
        split_idx = torch.bincount(idx_m)
        
        # get submatrices
        tensors = torch.split(sr, split_idx.tolist())

        # compute minimal eigenvalue of correlation matrix
        cmat = [t @ t.T for t in tensors]
        min_eigval = [torch.linalg.eigvalsh(c)[0] for c in cmat]
        
        for i in range(maxm):
            tmp[i].add_(min_eigval[i])
        outputs = tmp

        inputs[self.output_key] = outputs 
        return inputs
    

class FCorrelation(nn.Module):
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
        super(FCorrelation, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
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
        # compute scalar representations
        sr = inputs['scalar_representation']
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        tmp = torch.zeros((maxm, self.n_in), dtype=sr.dtype, device=sr.device)
        split_idx = torch.bincount(idx_m)
        
        # get submatrices
        tensors = torch.split(sr, split_idx.tolist())

        # compute minimal eigenvalue of correlation matrix
        cmat = [t.T @ t for t in tensors] # list of n_in x n_in matrices
        min_eigvecs = [torch.linalg.eigh(c)[1][0] for c in cmat]
        
        for i in range(maxm):
            tmp[i].add_(min_eigvecs[i])
        
        # predict atomwise contributions
        y = self.outnet(tmp)
        
        inputs[self.output_key] = torch.squeeze(y, -1)
        return inputs
        
        outputs = tmp

        inputs[self.output_key] = outputs 
        return inputs
    