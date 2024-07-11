# pylint: disable=invalid-name
"""Implementation of tableau calculus rules"""

import functools
import itertools
import operator
from typing import Callable, Generator, Iterable, List, Optional, Set, Tuple

import src.logic.base.syntax as S
import src.logic.base.tableau as T


def generate_models(
    tableau: T.Tableau, axioms: Iterable[Callable[[T.Tableau], T.Tableau]]
) -> Generator[T.Tableau, None, None]:
    """Generate possible models of the"""
    # First, apply non-branching rules to the tableau until none can be applied
    maximal_non_branching = try_non_branching_rules(tableau)
    # Then, figure out the tableau to be used for branching
    maximal_non_branching = tableau
    if maximal_non_branching:
        maximal_non_branching = T.Tableau.merge(maximal_non_branching, tableau, parent=tableau)
    else:
        maximal_non_branching = tableau
    # Invoke axioms on the branching root
    maximal_non_branching = T.Tableau.merge(
        maximal_non_branching, *map(lambda axiom: axiom(maximal_non_branching), axioms), parent=tableau
    )
    # Collect branches from branching rules
    branches = try_branching_rules(maximal_non_branching)
    # If no branches are produces, resolve now
    if not branches:
        # Yield if maximal non-branching tableau is consistent
        if maximal_non_branching and is_branch_consistent(maximal_non_branching):
            yield maximal_non_branching
        return None
    # Loop over branches and expand the branches
    for branch in filter(is_branch_consistent, branches):
        # Recursively generate models
        outputs = list(generate_models(branch, axioms))
        if outputs:
            yield from outputs
        else:
            yield branch
    return None


def try_non_branching_rules(tableau: T.Tableau) -> Optional[T.Tableau]:
    """Try applying non branching rules until exhausted.
    Return None if no rules are applicable to the initial node"""
    non_branching_rules = (
        try_and_elim,
        try_double_negation,
        try_forall_elim,
        try_focused_forall_elim,
    )
    # Apply all rules to tableau and collect non-emtpy outputs
    outputs: List[T.Tableau] = list(
        filter(operator.truth, map(lambda rule: rule(tableau), non_branching_rules))
    )
    output: Optional[T.Tableau] = None
    while outputs:
        # Get non empty outputs from rules
        if output:
            outputs = *outputs, output
        output = T.Tableau.merge(*outputs, parent=tableau)
        # Reapply rules to current tableau node
        outputs = list(
            filter(operator.truth, map(lambda rule: rule(output), non_branching_rules))
        )
    return output


def try_branching_rules(tableau: T.Tableau) -> Optional[Iterable[T.Tableau]]:
    """Try applying branching rules until exhausted.
    Return None if no rules are applicable to the initial node"""
    branching_rules = (try_or_elim, try_exists_elim, try_focused_exists_elim)
    # Apply all rules to tableau and collect non-emtpy outputs

    outputs: List[List[T.Tableau]] = list(
        filter(operator.truth, map(lambda rule: rule(tableau), branching_rules))
    )
    if not outputs:
        return None
    # Now we get the merged branches of the next level of inference
    next_level: List[T.Tableau] = list(
        itertools.starmap(
            functools.partial(T.Tableau.merge, parent=tableau),
            itertools.product(*outputs),
        )
    )
    # We apply the same thing recursively
    output = set()
    for tableau_ in next_level:
        maximal = try_branching_rules(tableau_)
        if maximal:
            output.add(T.Tableau.merge(tableau_, *maximal, parent=tableau))
        else:
            output.add(tableau_)
    return output


def is_branch_consistent(tableau: T.Tableau) -> bool:
    """Return True if the current tableau branch is a closed branch"""
    # pylint: disable=too-many-branches
    # pylint: disable=R0911:too-many-return-statements
    branch_formulas = list(tableau.branch_formulas)
    # Check if the branch contains false
    if S.False_ in branch_formulas:
        return False
    # Check if the branch contains -True
    for formula in branch_formulas:
        if isinstance(formula, S.Not) and formula.formula is S.True_:
            return False
    # Check if a formula and its negation are in the branch
    for formula in branch_formulas:
        if isinstance(formula, S.Not):
            if formula.formula in branch_formulas:
                return False
    # Check equality violations
    for formula in tableau.formulas:
        if isinstance(formula, S.Eq):
            if formula.left != formula.right:
                return False
        if isinstance(formula, S.Not) and isinstance(formula.formula, S.Eq):
            if formula.formula.left == formula.formula.right:
                return False
    return True


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


def remove_double_negations(formula: S.Formula) -> S.Formula:
    """Returns formula after removing double negations,
    returns the same formula if it is not double negated"""
    if isinstance(formula, S.Not) and isinstance(formula.formula, S.Not):
        return formula.formula.formula
    return formula


def try_or_elim(tableau: T.Tableau) -> Optional[Iterable[T.Tableau]]:
    """Apply or elimination (also works for the de-morgan's equivalent),
    then return an iterable of the product of branches
    """
    # Get all disjunction formulas using the desugared form
    disjunctions: Iterable[S.Not] = filter(
        lambda f: isinstance(f, S.Not) and isinstance(f.formula, S.And),
        tableau.formulas,
    )
    disjuncts: Iterable[Tuple[S.Formula, S.Formula]] = map(
        lambda f: (S.Not(f.formula.left), S.Not(f.formula.right)), disjunctions
    )
    # Remove double negations
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
    # Remove repeated branches
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


def try_exists_elim(tableau: T.Tableau) -> Optional[Iterable[T.Tableau]]:
    """Apply exists elimination on current node
    (also works for the un-sugared not-forall-not-Formula equivalent),
    then return an iterable of the product of branches
    """
    # Get all exists formulas using the desugared form
    quantified_formulas: Iterable[S.Not] = filter(
        lambda f: isinstance(f, S.Not) and isinstance(f.formula, S.Forall),
        tableau.formulas,
    )
    # For every quantified formula, collect all of its branches in a list
    all_sub_branches: List[Set[T.Tableau]] = []
    for quantified_formula in quantified_formulas:
        inner_formula: S.Forall = quantified_formula.formula
        # Filter out applicable entities
        # pylint: disable=W0640:cell-var-from-loop
        applicable_entities: Iterable[S.Term] = filter(
            lambda e: e.sort == inner_formula.sort, tableau.branch_entities
        )
        sub_branches: Set[T.Tableau] = set(
            map(
                lambda e: T.Tableau(
                    [remove_double_negations(S.Not(inner_formula.partial_formula(e)))]
                ),
                applicable_entities,
            )
        )
        witness: S.Constant = S.Constant(quantified_formula.sort)
        witness_formula = remove_double_negations(
            S.Not(inner_formula.partial_formula(witness))
        )
        sub_branches.add(T.Tableau([witness_formula], entities=[witness]))
        # Only add nonempty subbranches. Otherwise cartesian product fails
        if sub_branches:
            # Add current formula's branches to all branches
            all_sub_branches.append(sub_branches)
    # If not sub branches were found, return None
    if not all_sub_branches:
        return None
    # The produced branches should be the cartesian product of the current inner branches
    output_branches: Set[T.Tableau] = set(
        map(
            lambda ts: T.Tableau.merge(*ts, parent=tableau),
            itertools.product(*all_sub_branches),
        )
    )
    # Map branches to tableaus
    if output_branches:
        return output_branches
    return None


def try_focused_exists_elim(tableau: T.Tableau) -> Optional[Iterable[T.Tableau]]:
    """Apply focused exists elimination on current node
    (also works for the un-sugared not-forall-not-Formula equivalent),
    then return an iterable of the product of branches
    """
    # Get all exists formulas using the desugared form
    quantified_formulas: Iterable[S.Not] = filter(
        lambda f: isinstance(f, S.Not) and isinstance(f.formula, S.ForallF),
        tableau.formulas,
    )
    # For every quantified formula, collect all of its branches in a list
    all_sub_branches: List[Set[T.Tableau]] = []
    for quantified_formula in quantified_formulas:
        inner_formula: S.ForallF = quantified_formula.formula
        # Filter out applicable entities
        # pylint: disable=W0640:cell-var-from-loop
        applicable_entities: Iterable[S.Term] = filter(
            lambda e: e.sort == inner_formula.sort, tableau.branch_entities
        )
        applicable_entities: Iterable[S.Term] = filter(
            lambda e: inner_formula.unfocused_partial(e) in tableau.branch_formulas,
            applicable_entities,
        )
        sub_branches: Set[T.Tableau] = set(
            map(
                lambda e: T.Tableau(
                    [remove_double_negations(S.Not(inner_formula.focused_partial(e)))]
                ),
                applicable_entities,
            )
        )
        witness: S.Constant = S.Constant(quantified_formula.sort)
        witness_formulas = (
            inner_formula.unfocused_partial(witness),
            S.Not(inner_formula.focused_partial(witness)),
        )
        witness_formulas = map(remove_double_negations, witness_formulas)
        sub_branches.add(T.Tableau(list(witness_formulas), entities=[witness]))
        # Add current formula's branches to all branches
        all_sub_branches.append(sub_branches)
    # If not sub branches were found, return None
    if not all_sub_branches:
        return None
    # The produced branches should be the cartesian product of the current inner branches
    output_branches: Set[T.Tableau] = set(
        itertools.starmap(
            functools.partial(T.Tableau.merge, parent=tableau),
            itertools.product(*all_sub_branches),
        )
    )
    # Map branches to tableaus
    if output_branches:
        return output_branches
    return None
