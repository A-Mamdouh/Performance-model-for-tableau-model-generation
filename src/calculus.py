"""Implementation of tableau calculus rules"""

# pylint: disable=invalid-name
import itertools
from typing import Callable, Iterable, List, Optional, Tuple

import src.syntax as S
import src.tableau as T


def check_contradiction(tableau: T.Tableau) -> bool:
    """Return True if the current tableau branch is a closed branch"""
    # pylint: disable=too-many-branches
    # pylint: disable=R0911:too-many-return-statements
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
    """Return child tableau with new conjuncts from the current tableau conjunctions.
    Only adds conjuncts if they are unique to the branch
    """
    conjunctions: Iterable[S.And] = filter(
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
    """Return child tableau with a single depth of double negations removed.
    Only adds unique formulas to the branch. Returns None if no double negations exist
    """
    double_negations: Iterable[S.Not] = filter(
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


def try_forall_elim(tableau: T.Tableau) -> Optional[T.Tableau]:
    """Return child tableau after applying forall axioms on branch to entities in this node.
    Returns None if no new branch-unique formulas are added
    """
    axioms: Iterable[S.Forall] = filter(
        lambda formula: isinstance(formula, S.Forall), tableau.branch_formulas
    )
    output_formulas: List[S.Formula] = []
    for axiom in axioms:
        # Filter node entities by sort, since the logic is sorted
        # pylint: disable=W0640:cell-var-from-loop
        applicable_entities = filter(
            lambda term: term.sort == axiom.sort, tableau.branch_entities
        )
        # Map applicable terms over the quantified partial formula
        output_formulas.extend(map(axiom.partial_formula, applicable_entities))
    output_formulas = set(output_formulas).difference(tableau.branch_formulas)
    if output_formulas:
        return T.Tableau(formulas=output_formulas, parent=tableau)
    return None


def try_focused_forall_elim(tableau: T.Tableau) -> Optional[T.Tableau]:
    """Return child tableau after applying focused forall axioms on branch to entities in this node.
    Returns None if no new branch-unique formulas are added
    """
    axioms: Iterable[S.ForallF] = filter(
        lambda formula: isinstance(formula, S.ForallF), tableau.branch_formulas
    )
    output_formulas: List[S.Formula] = []
    for axiom in axioms:
        # Filter node entities by sort, since the logic is sorted
        # pylint: disable=W0640:cell-var-from-loop
        applicable_entities = filter(
            lambda term: term.sort == axiom.sort, tableau.branch_entities
        )
        # Filter out entities that do not match the unfocused part
        applicable_entities = filter(
            lambda term: axiom.unfocused_partial(term) in tableau.branch_entities,
            applicable_entities,
        )
        # Map applicable terms over the quantified partial formula
        output_formulas.extend(map(axiom.focused_partial, applicable_entities))
    output_formulas = set(output_formulas).difference(tableau.branch_formulas)
    if output_formulas:
        return T.Tableau(formulas=output_formulas, parent=tableau)
    return None


def try_or_elim(tableau: T.Tableau) -> Optional[Iterable[T.Tableau]]:
    """Apply or elimination (also works for the de-morgan's equivalent),
    then return an iterable of the product of branches
    """
    disjunctions: Iterable[S.Not] = filter(
        lambda f: isinstance(f, S.Not) and isinstance(f.formula, S.And),
        tableau.formulas,
    )
    disjuncts: Iterable[Tuple[S.Formula, S.Formula]] = map(
        lambda f: (S.Not(f.formula.left), S.Not(f.formula.right)), disjunctions
    )
    # Remove double negations
    remove_double_negations: Callable[[S.Not], S.Formula] = (  # noqa: E731
        lambda f: f.formula.formula if isinstance(f.formula, S.Not) else f
    )
    disjuncts = map(
        lambda pair: (
            remove_double_negations(pair[0]),
            remove_double_negations(pair[1]),
        ),
        disjuncts,
    )
    # Mix branches
    branches = map(
        lambda branch: set(branch).difference(tableau.branch_formulas),
        itertools.product(*disjuncts),
    )
    # Remove empty branches
    branches = filter(lambda branch: len(branch) > 0, branches)
    # Remove repeatd branches
    branches = set(
        map(
            lambda branch_set: tuple(sorted(branch_set, key=S.Formula.__str__)),
            branches,
        )
    )
    # Map branches to tableaus
    if branches:
        return map(lambda branch: T.Tableau(branch, parent=tableau), branches)
    return None
