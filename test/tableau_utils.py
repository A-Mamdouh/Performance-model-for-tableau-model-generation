"""Utility functions for tableau creation"""

from typing import Any, Dict, Iterable

import src.logic.base.syntax as S
import src.logic.base.tableau as T


def create_tableau_params(**overrides) -> Dict[str, Any]:
    """Return default parameters for a tableau"""
    type_sort = S.Sort("Type")
    agent_sort = S.Sort("Agent")
    event_sort = S.Sort("Event")
    agents = [agent_sort.make_constant() for _ in range(3)]
    events = [event_sort.make_constant() for _ in range(3)]
    types = [type_sort.make_constant() for _ in range(3)]
    formulas = [S.True_, S.Or(S.True_, S.False_)]
    p1 = S.Predicate("p1", 2)
    p2 = S.Predicate("p2", 2)
    for event, agent, type_ in zip(events, agents, types):
        formulas.append(p1(event, agent))
        formulas.append(p2(event, type_))
    formulas.append(S.Eq(S.True_, S.True_))
    formulas.append(S.Not(S.Eq(S.True_, S.False_)))
    entities = [*events, *types, *agents]
    for i in range(3):
        formulas.append(S.And(p2(events[i], types[i]), p1(events[i], agents[i])))
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
