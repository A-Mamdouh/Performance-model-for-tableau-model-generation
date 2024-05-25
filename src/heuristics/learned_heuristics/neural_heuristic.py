"""Implementation of heuristic using neural networks"""

from typing import Tuple
from src.heuristics.base_heuristic import Heuristic
from src.heuristics.context_token import ContextToken
from src.heuristics.learned_heuristics.deep_learning_models.simple_lstm_model import (
    Model,
)
from src.search.search_node import TableauSearchNode


class NeuralHeuristic(Heuristic):
    """Mimicking minimum events non-stochastic agent"""

    def __init__(self):
        self._model = Model()

    def setup_model(self) -> None:
        """Setup model and move to gpu"""
        self._model.eval()

    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextToken, float]:
        score, new_context = self._model.from_search_node(search_node, previous_context)
        return new_context, score

    def get_empty_context(self) -> ContextToken:
        return self._model.get_initial_h()
