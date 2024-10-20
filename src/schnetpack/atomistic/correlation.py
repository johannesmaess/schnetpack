# define a module to compute scalar value from smallest eigenvalue of correlation matrix of scalar_representations

from typing import Dict
import torch
import torch.nn as nn
import schnetpack.properties as properties
import schnetpack.nn as snn

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
    