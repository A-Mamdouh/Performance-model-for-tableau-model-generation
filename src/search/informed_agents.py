"""Implementation of informed search agents"""

from src.search.search_agent_base import InferenceAgent
from src.search.search_node import HeuristicTableauSearchNode
import src.heuristics as H
import src.logic.tableau as T


class GreedyAgent(InferenceAgent):
    """Greedy search agent. Explores nodes based on the heuristic cost"""

    def __init__(self, heuristic: H.Heuristic, *args, **kwargs):
        self._heuristic = heuristic
        super().__init__(*args, **kwargs)

    def _create_initial_node(self) -> HeuristicTableauSearchNode:
        return HeuristicTableauSearchNode(
            priority=0,
            sentence_depth=0,
            tableau=self._create_axioms(),
            context_object=self._heuristic.get_empty_context(),
        )

    def make_search_node(
        self,
        sentence_depth: int,
        model_tableau: T.Tableau,
        parent: HeuristicTableauSearchNode,
    ) -> HeuristicTableauSearchNode:
        # Get embedding of the current branch
        # Embedding will be all literals about events that were relevant between the current model and previous model
        event_info = model_tableau.branch_event_info
        # Pass to heuristic for scoring
        new_context, branch_score = self._heuristic.score_branch(
            previous_context=parent.context_object, branch_embeddings=event_info
        )
        # Return new node
        return HeuristicTableauSearchNode(
            priority=branch_score,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
            context_object=new_context,
        )