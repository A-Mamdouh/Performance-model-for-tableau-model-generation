"""This module implements the minimum events heuristic.
This heuristics favors models with less events.
"""

from typing import Tuple

from src.heuristics.base_heuristic import Heuristic
from src.heuristics.context_token import ContextToken
from src.search.search_node import TableauSearchNode


class MinEvents(Heuristic):
    """This heuristic prefers models with less entities"""

    def __init__(self, event_penalty: float = -20, model_depth_reward: float = 100):
        self.event_penalty = event_penalty
        self.model_depth_reward = model_depth_reward

    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextToken, float]:
        context = super().score_node(previous_context, search_node)[0]
        # Set the score to be the number of events / sentence depth
        n_events = len(list(search_node.tableau.branch_events))
        total_reward = (
            n_events * self.event_penalty
            + search_node.sentence_depth * self.model_depth_reward
        )
        return context, total_reward
