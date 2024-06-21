"""
Testing syntax implementation
"""

import src.logic.syntax as S


class TestOperatorHeirarchy:
    """Check if operator built on not and _and_ are also accessible in their simplified forms"""

    def test_or_is_not_and(self):
        """Check that Or(l, r) is an instance of Not(And(Not(l), Not(r)))"""
        f = S.Or(S.True_, S.False_)
        # Check if f = Or(True, False) can be read as Not(And(Not(True), Not(False)))
        assert isinstance(f, S.Not)
        assert isinstance(f.formula, S.And)
        inner_formula: S.And = f.formula
        # pylint: disable=E1101
        assert isinstance(inner_formula.left, S.Not)
        assert inner_formula.left.formula is f.left
        assert isinstance(inner_formula.right, S.Not)
        assert inner_formula.right.formula is f.right

    def test_impl_is_not_and(self):
        """Check that Implicaties(l, r) is an instance of Or(Not(l), r)"""
        f = S.Implies(S.True_, S.False_)
        # Check if f = Implies(True, False) can be read as Or(Not(True), False)
        assert isinstance(f, S.Or)
        assert isinstance(f.left, S.Not)
        assert f.left.formula is f.pre
        assert f.right is f.post
