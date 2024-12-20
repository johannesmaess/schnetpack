from typing import List
import torch
import schnetpack.properties as structure
from schnetpack.datasets import ASEAtomsData

class ExtendableASEAtomsData(ASEAtomsData):
    def __init__(self, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        # Load first structure
        row = conn.get(idx + 1)

        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(row["pbc"])

        # Load follower structure to retrieve positions
        row_next = conn.get(idx + 2)
        properties_next = {}

        if load_structure:
            properties_next["_positions_next"] = (
                torch.tensor(row_next["positions"].copy()) * self.distance_conversion
            )
            # For control we also save _idx_nex
            properties_next["_idx_next"] = torch.tensor([idx + 1])
        
        properties.update(properties_next)
        return properties