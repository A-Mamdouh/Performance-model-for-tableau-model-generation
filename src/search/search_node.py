"""This class contains struct definitions for the search agent"""

from dataclasses import dataclass, field
from typing import Optional

from src.heuristics.context_token import ContextToken
import src.logic.tableau as T


@dataclass(order=True)
class TableauSearchNode:
    """A struct containing a maximal search node of uninforment search agents"""

    priority: int
    sentence_depth: int = field(compare=False)
    # Maximal tableau
    tableau: T.Tableau = field(compare=False)
    parent: Optional["TableauSearchNode"] = field(compare=False, default=None)


@dataclass
class HeuristicTableauSearchNode(TableauSearchNode):
    """A struct containing a maximal search node of informed search agents"""

    context_object: ContextToken = field(compare=False, default=None)
