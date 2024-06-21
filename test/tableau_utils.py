"""Utility functions for tableau creation"""

from typing import Any, Dict, Iterable

import src.logic.syntax as S
import src.logic.tableau as T


def create_tableau_params(**overrides) -> Dict[str, Any]:
    """Return default parameters for a tableau"""
    types = [S.Constant.Type() for _ in range(3)]
    agents = [S.Constant.Agent() for _ in range(3)]
    events = [S.Constant.Event() for _ in range(3)]
    formulas = [S.True_, S.Or(S.True_, S.False_)]
    for event, agent, type_ in zip(events, agents, types):
        formulas.append(S.Agent(event, agent))
        formulas.append(S.Type_(event, type_))
    formulas.append(S.Eq(S.True_, S.True_))
    formulas.append(S.Not(S.Eq(S.True_, S.False_)))
    entities = [*events, *types, *agents]
    for i in range(3):
        formulas.append(
            S.And(S.Type_(events[i], types[i]), S.Agent(events[i], agents[i]))
        )
    return {
        **{"formulas": formulas, "entities": entities},
        **overrides,
    }


def create_tableau_chain(
    length: int = 4, start_empty: bool = True, **overrides
) -> Iterable[T.Tableau]:
    """Return a chain of tableaus starting with an empty tableau"""
    parent: T.Tableau = None
    output = []
    if start_empty:
        parent = T.Tableau(
            **create_tableau_params(
                **{**overrides, "parent": None, "formulas": [], "entities": []}
            )
        )
        output.append(parent)
        length -= 1
    for _ in range(length):
        parent = T.Tableau(**create_tableau_params(**{"parent": parent, **overrides}))
        output.append(parent)
    return output
