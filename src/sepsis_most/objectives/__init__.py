"""Training objective: competing-risks likelihood, ranking, load balancing,
specialisation and adverse-event terms."""
from .competing_risks import cr_nll
from .joint import LossWeights, joint_objective
from .ranking import ranking_loss
from .specialisation import specialisation_loss

__all__ = ["LossWeights", "cr_nll", "joint_objective", "ranking_loss", "specialisation_loss"]
