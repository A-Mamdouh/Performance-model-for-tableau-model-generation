"""This module implements a heuristic,
where reusing words with the highest salience scores higher"""

from typing import Tuple

from src.heuristics.base_heuristic import Heuristic
from src.heuristics.context_token import ContextToken
from src.search.search_node import TableauSearchNode


class AverageSalience(Heuristic):
    """This heuristic prefers models with higher average salience"""

    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextToken, float]:
        context = super().score_node(previous_context, search_node)[0]
        salience_records = search_node.salience_records
        average_salience = sum(salience_records.values()) / min(
            1, len(salience_records)
        )
        return context, average_salience
