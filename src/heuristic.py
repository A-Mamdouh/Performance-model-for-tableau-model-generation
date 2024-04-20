from . import syntax
from . import tableau


class Heuristic:
    def scoreFormula(self, formula: syntax.Formula, tableau: tableau.Tableau) -> float:
        return 0.0
