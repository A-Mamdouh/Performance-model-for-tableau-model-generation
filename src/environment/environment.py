"""Universe of discourse"""

from src.logic.base.syntax import Constant, Predicate, Sort
from src.logic.simple_event_semantics.tableau import Tableau


class Sorts:
    """Namespace for sorts"""

    agent = Sort("agent")
    location = Sort("location")
    event = Sort("event")
    action = Sort("action")


class Constants:
    """Namespace for individuals in the environment"""

    # Locations
    library: Constant = Sorts.location.make_constant("library")
    kitchen: Constant = Sorts.location.make_constant("kitchen")
    lounge: Constant = Sorts.location.make_constant("lounge")
    garden: Constant = Sorts.location.make_constant("garden")
    bedroom: Constant = Sorts.location.make_constant("bedroom")
    # Actors
    alex: Constant = Sorts.agent.make_constant("alex")
    bob: Constant = Sorts.agent.make_constant("bob")
    charlie: Constant = Sorts.agent.make_constant("charlie")
    diana: Constant = Sorts.agent.make_constant("diana")
    # actions
    see: Constant = Sorts.action.make_constant("see")
    enter: Constant = Sorts.action.make_constant("enter")
    take_knife: Constant = Sorts.action.make_constant("take_knife")
    murder: Constant = Sorts.action.make_constant("murder")
    hear: Constant = Sorts.action.make_constant("hear")

    @staticmethod
    def get(constant_name: str, sort: Sort | None = None) -> Constant:
        """Get a constant from the environment using its string name"""
        for term in Constants.__dict__.values():
            if not isinstance(term, Constant):
                continue
            if term.name == constant_name:
                if sort is None or term.sort == sort:
                    return term
        raise KeyError(f"No constant with name {constant_name} in this environment")


class Predicates:
    """Namespace for predicates"""

    subject = Predicate("subject", 2, sorts=[Sorts.event, Sorts.agent])
    action = Predicate("action", 2, sorts=[Sorts.event, Sorts.action])
    object = Predicate("object", 2, sorts=[Sorts.event, Sorts.agent])
    location = Predicate("location", 2, sorts=[Sorts.event, Sorts.location])


class Axioms:
    """Axioms of our environment"""

    def axiom_going_to_kitchen_is_suspicious(self, tableau: Tableau) -> Tableau:
        """If someone goes to the kitchen where the knife is, then they are a suspect"""
    
    def axiom_going_to_the_murder_location_is_suspicious(self, tableau: Tableau) -> Tableau:
        """If someone goes to the murder location, then they are a suspect"""
    
    def axiom_the_murderer_is_a_suspect(self, tableau: Tableau) -> Tableau:
        "Axiom that simple means murderer(M) -> suspect(M)"
    
    def axiom_a_suspect_is_the_murderer(self, tableau: Tableau) -> Tableau:
        """suspect(X) -> murderer(X) | -murderer(X).
            a -> T is a tautology, but in this case is useful for model generation
        """

