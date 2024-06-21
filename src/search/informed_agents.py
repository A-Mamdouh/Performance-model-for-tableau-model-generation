"""Implementation of informed search agents"""

from src.search.search_agent_base import InferenceAgent
from src.search.search_node import HeuristicTableauSearchNode, TableauSearchNode
from src.heuristics.base_heuristic import Heuristic
import src.logic.tableau as T


class GreedyAgent(InferenceAgent):
    """Greedy search agent. Explores nodes based on the heuristic score"""

    def __init__(self, heuristic: Heuristic, *args, **kwargs):
        self._heuristic = heuristic
        super().__init__(*args, **kwargs)

    def _create_initial_node(self) -> HeuristicTableauSearchNode:
        return HeuristicTableauSearchNode(
            priority=0,
            sentence_depth=0,
            tableau=self._create_axioms(),
            context_object=self._heuristic.get_empty_context(),
            salience_records={},
        )

    def make_search_node(
        self,
        sentence_depth: int,
        model_tableau: T.Tableau,
        parent: HeuristicTableauSearchNode,
    ) -> HeuristicTableauSearchNode:
        # Create a normal node and pass to heuristic
        new_salience_records = TableauSearchNode.get_new_salience_record(
            model_tableau, parent
        )
        search_node = TableauSearchNode(
            0,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
            salience_records=new_salience_records,
        )
        new_context, branch_score = self._heuristic.score_node(
            previous_context=parent.context_object, search_node=search_node
        )
        # Return new node
        return HeuristicTableauSearchNode(
            priority=branch_score,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
            salience_records=new_salience_records,
            context_object=new_context,
        )