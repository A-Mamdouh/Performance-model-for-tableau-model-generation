from src import syntax as S


class TestOperatorHeirarchy:
    def test_or_is_not_and(self):
        """Check that Or(l, r) is an instance of Not(And(Not(l), Not(r)))"""
        f = S.Or(S.True_, S.False_)
        # Check if f = Or(True, False) can be read as Not(And(Not(True), Not(False)))
        assert isinstance(f, S.Not)
        assert isinstance(f.formula, S.And)
        assert isinstance(f.formula.left, S.Not)
        assert f.formula.left.formula is f.left
        assert isinstance(f.formula.right, S.Not)
        assert f.formula.right.formula is f.right

    def test_impl_is_not_and(self):
        """Check that Implicaties(l, r) is an instance of Or(Not(l), r)"""
        f = S.Implies(S.True_, S.False_)
        # Check if f = Implies(True, False) can be read as Or(Not(True), False)
        assert isinstance(f, S.Or)
        assert isinstance(f.left, S.Not)
        assert f.left.formula is f.pre
        assert f.right is f.post

    # TODO: Test equality and hashing
    # TODO: Test quantifiers
