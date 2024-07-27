"""Universe of discourse"""

from src.logic.base.syntax import Constant, Predicate, Sort


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