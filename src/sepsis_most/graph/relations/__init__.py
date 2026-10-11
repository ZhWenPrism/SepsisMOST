"""SOFAE relation matrices: temporal precedence, co-occurrence and syndrome trigger."""
from .matrices import (TRIGGER_SLACK_H, RelationMatrices, add_mods_onset, prune,
                       relation_matrices)

__all__ = ["TRIGGER_SLACK_H", "RelationMatrices", "add_mods_onset", "prune", "relation_matrices"]
