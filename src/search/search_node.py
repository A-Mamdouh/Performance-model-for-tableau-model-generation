"""This class contains struct definitions for the search agent"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.heuristics.context_token import ContextToken
import src.logic.syntax as S
import src.logic.tableau as T


@dataclass
class TableauSearchNode:
    """A struct containing a maximal search node of uninforment search agents"""

    priority: int
    sentence_depth: int = field(compare=False)
    # Maximal tableau
    tableau: T.Tableau = field(compare=False)
    parent: Optional["TableauSearchNode"] = field(compare=False, default=None)
    salience_records: Dict[str, float] = field(compare=False, default_factory=dict)

    @staticmethod
    def get_new_salience_record(
        new_tableau: T.Tableau,
        previous_node: Optional["TableauSearchNode"] = None,
        salience_decay: float = 0.8,
        initial_salience: float = 100.0,
    ) -> Dict[str, float]:
        """Get a new salience record from the previous node"""
        new_literals = []
        tableau = new_tableau
        while tableau is not previous_node.tableau:
            new_literals.extend(tableau.literals)
            tableau = tableau.parent
        tableau_words = S.get_words(new_literals)
        new_record = {}
        if previous_node is not None:
            for key, value in previous_node.salience_records.items():
                new_record[key] = value * salience_decay
        for word in tableau_words:
            new_record[word] = initial_salience
        print("RECORD>>", new_record)
        return new_record

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, TableauSearchNode):
            raise TypeError(
                f"Cannot compare Tableau Search Node with type {type(other)}"
            )
        return self.priority > other.priority

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, TableauSearchNode):
            raise TypeError(
                f"Cannot compare Tableau Search Node with type {type(other)}"
            )
        return self.priority >= other.priority


@dataclass
class HeuristicTableauSearchNode(TableauSearchNode):
    """A struct containing a maximal search node of informed search agents"""

    context_object: ContextToken = field(compare=False, default=None)

