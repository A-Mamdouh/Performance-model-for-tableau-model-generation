import src.syntax as S
import src.tableau as T

from typing import Any, Dict, Iterable


def create_tableau_params(**overrides) -> Dict[str, Any]:
    """Return default parameters for a tableau"""
    types = [S.Constant.Type() for _ in range(3)]
    agents = [S.Constant.Agent() for _ in range(3)]
    events = [S.Constant.Event() for _ in range(3)]
    formulas = [S.True_, S.Or(S.True_, S.False_)]
    entities = [*events, *types, *agents]
    for i in range(3):
        formulas.append(
            S.And(S.Type_(events[i], types[i]), S.Agent(events[i], agents[i]))
        )
    return {
        **dict(formulas=formulas, entities=entities),
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
