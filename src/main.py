import language_signature as S

a = S.Constant("a")
x = S.Variable("X")
mother = S.FunctionSymbol("mother", 1)
mother_of_a = mother(x)
_and = S.LogicalOperator("and", 2)

print(a, x, mother, mother_of_a, _and(S._true, S._false))
