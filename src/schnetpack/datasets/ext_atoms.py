from typing import List
import torch
from schnetpack.datasets import ASEAtomsData

class ExtendableASEAtomsData(ASEAtomsData):
    def _get_properties(
        self, conn, idx: int, load_properties: List[str], load_structure: bool
    ):
        # Load first structure
        properties = super()._get_properties(conn, idx, load_properties, load_structure)

        # Load follower structure to retrieve positions
        if load_structure:
            row_next = conn.get(idx + 2)
            properties["_positions_next"] = (
                torch.tensor(row_next["positions"].copy()) * self.distance_conversion
            )
            # For control we also save _idx_nex
            properties["_idx_next"] = torch.tensor([idx + 1])
        
        return properties