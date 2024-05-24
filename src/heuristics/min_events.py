"""This module implements the minimum events heuristic.
This heuristics favors models with less events.
"""

from typing import Tuple

from src.heuristics.base_heuristic import Heuristic
from src.heuristics.context_token import ContextToken
from src.search.search_node import TableauSearchNode


class MinEvents(Heuristic):
    """This heuristic prefers models with less entities"""

    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextToken, float]:
        context = super().score_node(previous_context, search_node)[0]
        # Set the score to be the number of events / sentence depth
        n_events = len(list(search_node.tableau.events))
        return context, n_events / max(1, search_node.sentence_depth)
