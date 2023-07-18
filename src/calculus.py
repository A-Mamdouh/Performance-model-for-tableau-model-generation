from .syntax import *
from .tableau import *
from typing import Generator, Iterable
from itertools import product


__all__ = (
    "generate_models",
    "apply_axioms",
    "check_contradictions",
    "t_and",
    "t_or",
    "t_dneg",
    "t_exists",
    "t_existsf",
    "t_forall",
    "t_forallf",
)


def generate_models(tableau: Tableau) -> Generator[Tableau, None, None]:
    model = tableau
    productions = [None]
    # First, apply axioms and non-branching rules exhaustively
    while len(productions) > 0:
        productions.clear()
        for f in model.formulas:
            productions.extend(t_no_branch(model, f))
        productions.extend(apply_axioms(model))
        if len(productions) > 0:
            model = Tableau.merge(*productions, parent=model)
    model_chain = []
    while model != tableau:
        model_chain.append(model)
        model = model.parent
    model = Tableau.merge(*model_chain, parent=tableau)
    # Then, collect branches from branching rules
    branches = []
    for f in *tableau.formulas, *model.formulas:
        b = t_branch(model, f)
        if len(b) > 0:
            branches.extend(b)
    # If no branches produced, check for contradiction and yield model then return
    if len(branches) == 0:
        if not check_contradictions(model):
            yield model
        return
    # Recursively apply model generation routine to get models
    for branch_product in product(*branches):
        for new_model in generate_models(Tableau.merge(model, *branch_product, parent=tableau)):
            yield new_model


def apply_axioms(tableau: Tableau) -> Iterable[Tableau]:
    to_merge = []
    for f in tableau.branch_formulas:
        to_merge.extend(t_forall(tableau, f))
        to_merge.extend(t_forallf(tableau, f))
    if len(to_merge) > 0:
        return Tableau.merge(*to_merge),
    return []


def check_contradictions(tableau: Tableau) -> bool:
    """ return True if the current tableau has a contradiction. Checks input tableau against the whole branch """
    for formula in tableau.formulas:
        if isinstance(formula, Eq): # a = b
            if formula.left.sort != formula.right.sort:
                return True
            if formula.left != formula.right:
                return True
        if formula == False_:
            return True # False
        if Not(formula) in tableau.branch_formulas: # a, -a
            return True
        if isinstance(formula, Not):
            if formula.formula in tableau.branch_formula: # -a, a
                return True
            if formula.formula == True_: # -True
                return True
    return False


def t_no_branch(tableau: Tableau, f: And) -> Iterable[Tableau]:
    return *t_and(tableau, f), *t_dneg(tableau, f), *t_forall(tableau, f), *t_forallf(tableau, f)

def t_branch(tableau: Tableau, f: And) -> Iterable[Iterable[Tableau]]:
    branch_productions_list = [
        [*t_or(tableau, f)],
        [*t_exists(tableau, f)],
        [*t_existsf(tableau, f)]
    ]
    return list(filter(lambda x: len(x) > 0, branch_productions_list))


def t_and(tableau: Tableau, f: And) -> Iterable[Tableau]:
    if isinstance(f, And):
        formulas = [_f for _f in (f.left, f.right) if _f not in tableau.branch_formulas]
        if len(formulas) > 0:
            return (Tableau(formulas, parent=tableau),)
    return []


def t_or(tableau: Tableau, f: Not) -> Iterable[Tableau]:
    if isinstance(f, Not) and isinstance(f.formula, And):
        f = f.formula
        formulas = (Not(branch) for branch in (f.left, f.right))
        formulas = filter(lambda f_: f_ not in tableau.branch_formulas, )
        return (Tableau([f_,], parent=tableau) for f_ in formulas)
    return []


def t_dneg(tableau: Tableau, f: Not) -> Iterable[Tableau]:
    if isinstance(f, Not) and isinstance(f.formula, Not):
        formula = f.formula.formula
        if formula not in tableau.branch_formulas:
            return (Tableau([formula], parent=tableau),)
    return []


def t_exists(tableau: Tableau, f:Not) -> Iterable[Tableau]:
    if isinstance(f, Not) and isinstance(f.formula, Forall):
        qf = f.formula
        witness = Constant(qf.sort)
        formulas = (Not(qf.partial_formula(c)) for c in (*tableau.branch_entities, witness) if c.sort == qf.sort)
        formulas = list(filter(lambda f_: f_ not in tableau.branch_formulas, formulas))
        if len(formulas) > 0:
            return (Tableau([f_,], parent=tableau) for f_ in formulas)
    return []


def t_existsf(tableau: Tableau, f: Not) -> Iterable[Tableau]:
    if isinstance(f, Not) and isinstance(f.formula, ForallF):
        qf = f.formula
        witness = Constant(qf.sort)
        formulas = ( Not(qf.focused_partial(c))
            for c in tableau.branch_entities
            if c.sort == qf.sort and qf.unfocused_partial(c) in tableau.branch_formulas
        ) 
        formulas = list(filter(lambda f_: f_ not in tableau.branch_formulas, (*formulas, Not(qf.focused_partial(witness)))))
        if len(formulas) > 0:
            return (Tableau([f_], parent=tableau) for f_ in formulas)
    return []


def t_forall(tableau: Tableau, f: Forall) -> Iterable[Tableau]:
    if isinstance(f, Forall):
        formulas = (f.partial_formula(c) for c in tableau.branch_entities if c.sort == f.sort)
        formulas = list(filter(lambda f_: f_ not in tableau.branch_formulas, formulas))
        if len(formulas) > 0:
            return (Tableau(formulas, parent=tableau),)
    return []


def t_forallf(tableau: Tableau, f: ForallF) -> Iterable[Tableau]:
    if isinstance(f, ForallF):
        relation_entities = [c for c in tableau.branch_entities if f.unfocused_partial(c) in tableau.branch_formulas]
        if len(relation_entities) > 0:
            formulas = (f.focused_partial(c) for c in relation_entities if c.sort == f.sort)
            formulas = filter(lambda f_: f_ not in tableau.branch_formulas, formulas)
            return (Tableau(formulas, parent=tableau),)
    return []
    # TODO: Discuss this rule later with Michael and Fredrick
        # return (
        #     Tableau([Forall(lambda x: Not(f.unfocused_partial(x)), f.sort)], parent=tableau),
        #     Tableau([], parent=tableau)
        # )
        


"""
Calculus
______________________________
# And
a & b           |=  a, b;
# Not
## Or
-(a & b)        |=  -a  ; -b;
## DNeg
--a             |= a;
## Contra1
-a, a           |= False;
## Contra2
-True           |= False;
## Exists
-A_x.f[x]       |= -f[c1]; ... ; -f[cn]; -f[c_new];                 (forall c1, ..., cn \in H. Repeatable every new entity. c_new added to H)
-A_x.R[x]:f[x]  |= -f[c1]; ... ; -f[cn]; R[c_new], -f[c_new];       (forall c1, ..., cn \in H if R[c_x] \in M. Repeatable every new entity. c_new added to H)
# F
False           |= False;
# Forall
A_x.f[x]        |= f[c1], ... , f[cn] ;           (forall c1, ..., cn \in H. Repeatable every new entity)
A_x.R[x]:f[x]   |=                                (forall c1, ..., cn \in H if R[c] \in M. Repeatable every new entity)
                    Case 1 (R[c] is not empty) :  f[c1], ... , f[cn] ;
                    case 2 (R[c] is empty)     :  A_x.-R[x]; EMPTY;
# Equality
a = b           |= False;
"""
