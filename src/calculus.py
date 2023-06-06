from tableau import *
from language import *
from typing import List, Type


## Not rule
def contradiction(t: Tableau, f1: TaggedFormula) -> BottomNode | None:
    # Type assertions
    assert isinstance(f1, TaggedFormula)
    assert isinstance(t, Tableau)
    # Create a formula with opposite tag for checking the rule
    f2 = TaggedFormula(f1.form, not f1.tag)
    if not t.contains_all(f1, f2):
        return None
    # Return a new bottom node with the contradiction rule
    return BottomNode(f"contradiction: <{f1}, {f2}>", t)


def contradiction_constant(t: Tableau, f: TaggedFormula) -> BottomNode | None:
    # Type assertions
    assert isinstance(f, TaggedFormula)
    assert isinstance(t, Tableau)
    assert f.form in [Top, Bot]
    # Sanity checks
    if (f.form is Top and f.tag is True) or (f.form is Bot and f.tag is False):
        return None
    if not f in t:
        return None
    # Return bottom node
    return BottomNode(f"contradiction: <{f}>", t)


## Helper rule for connective destruction rules
def _destruct(
    t: Tableau,
    f: TaggedFormula,
    tag: bool,
    branch: bool,
    FCls: Type[And | Or],
    rule_name: str,
) -> Tableau | List[Tableau] | None:
    # Type assertions
    assert isinstance(f, TaggedFormula)
    assert isinstance(f.form, FCls)
    assert isinstance(t, Tableau)
    # Sanity checks
    if not f.tag is tag:
        return None
    if not f in t:
        return None
    # Get the left and right parts of the connective
    ltag = TaggedFormula(f.form.left, f.tag)
    rtag = TaggedFormula(f.form.right, f.tag)
    # If not branching, make new tableau
    if not branch:
        new_formulas = frozenset([x for x in [ltag, rtag] if not x in t])
        return Tableau(new_formulas, f"{rule_name}: <{f}>", t)
    # Handle branching
    lform = set()
    if not ltag in t:
        lform.add(ltag)
    rform = set()
    if not rtag in t:
        rform.add(rtag)
    return [
        Tableau(frozenset(lform), f"{rule_name}_l: <{f}>"),
        Tableau(frozenset(rform), f"{rule_name}_r: <{f}>"),
    ]


## And rules
def destruct_and_t(t: Tableau, f: TaggedFormula) -> Tableau | None:
    return _destruct(t, f, True, False, And, "destruct_and_t")


def destruct_and_f(t: Tableau, f: TaggedFormula) -> List[Tableau] | None:
    return _destruct(t, f, False, True, And, "destruct_and_f")


## Or rules
def destruct_or_t(t: Tableau, f: TaggedFormula) -> List[Tableau] | None:
    return _destruct(t, f, True, True, Or, "destruct_or_t")


def destruct_or_f(t: Tableau, f: TaggedFormula) -> List[Tableau] | None:
    return _destruct(t, f, False, False, Or, "destruct_or_f")


## Main script
if __name__ == "__main__":
    # Quick test to the logic of the calculus and implementation of the tableau
    root_tableau = Tableau(frozenset([TaggedFormula(Top, True)]))
    second_tableau = Tableau(
        frozenset([TaggedFormula(Top, False)]), "init", root_tableau
    )
    bottom_node = contradiction(second_tableau, TaggedFormula(Top, True))
    print(bottom_node)


"""
TODO:
    - Add quantifier rules
    - Add inference algorithm
"""
