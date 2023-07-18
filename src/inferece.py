from typing import Generator, Optional
from .narrator import *
from .tableau import *
from .calculus import *


__all__ = ("Agent",)


class Agent:

    def __init__(self):
        self.tableau = self._get_axioms()

    def listen(self, narrator: Narrator, tableau: Optional[Tableau] = None) -> Generator[Tableau, None, None]:
        if tableau is None:
            tableau = self.tableau
        sentence: Sentence = next(iter(narrator))
        formula = sentence.get_formula(self.get_focus())
        for model in self._get_models(formula, tableau):
            pass
    
    def _get_focus(self) -> Sentence.Focus:
        return Sentence.Focus.FULL
    
    @staticmethod
    def get_axioms() -> Tableau:
        
