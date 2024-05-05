from typing import Optional
import src.syntax as S
import src.tableau as T


def check_contradiction(tableau: T.Tableau) -> bool:
    """Return True if the current tableau branch is a closed branch"""
    branch_formulas = list(tableau.branch_formulas)
    # Check if the branch contains false
    if S.False_ in branch_formulas:
        return True
    # Check if the branch contains -True
    for formula in branch_formulas:
        if isinstance(formula, S.Not) and formula.formula is S.True_:
            return True
    # Check if a formula and its negation are in the branch
    for formula in branch_formulas:
        if isinstance(formula, S.Not):
            if formula.formula in branch_formulas:
                return True
    for event_info in tableau.branch_event_info:
        # At most one agent per event
        agents = list(map(lambda literal: literal.agent, event_info.positive_agents))
        if len(agents) > 1:
            return True
        # At most one type per event
        types = list(map(lambda literal: literal.type_, event_info.positive_types))
        if len(types) > 1:
            return True
    # Check equality violations
    for formula in tableau.formulas:
        if isinstance(formula, S.Eq):
            if formula.left != formula.right:
                return True
        if isinstance(formula, S.Not) and isinstance(formula.formula, S.Eq):
            if formula.formula.left == formula.formula.right:
                return True
    return False


def try_and_elim(tableau: T.Tableau) -> Optional[T.Tableau]:
    """Return child tableau with new conjuncts from the current tableau conjunctions. Only adds conjuncts if they are unique to the branch"""
    conjunctions: filter[S.And] = filter(
        lambda formula: isinstance(formula, S.And), tableau.formulas
    )
    # Collect unique conjuncts
    output_formulas = set(
        f for conj in conjunctions for f in (conj.left, conj.right)
    ).difference(set(tableau.branch_formulas))
    # If output is not empty, return new tableau
    if output_formulas:
        return T.Tableau(formulas=output_formulas, parent=tableau)
    return None


def try_double_negation(tableau: T.Tableau) -> Optional[T.Tableau]:
    """Return child tableau with a single depth of double negations removed. Only adds unique formulas to the branch. Returns None if no double negations exist"""
    double_negations: filter[S.Not] = filter(
        lambda formula: isinstance(formula, S.Not)
        and isinstance(formula.formula, S.Not),
        tableau.formulas,
    )
    # Get output formulas set and remove branch formulas
    output_formulas = set(
        map(lambda f: f.formula.formula, double_negations)
    ).difference(tableau.branch_formulas)
    # Return output if new formulas exist
    if output_formulas:
        return T.Tableau(formulas=output_formulas, parent=tableau)
    return None
