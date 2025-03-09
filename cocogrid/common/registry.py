"""Common registry components."""


class AlreadyRegisteredError(Exception):
    """Raised when an entity is already registered."""

    def __init__(self, registry: str, id: str) -> None:
        """Initialize the error with registry and entity ID."""
        super().__init__(f"Registry {registry} already contains an entity with ID {id}.")


class NotRegisteredError(Exception):
    """Raised when an entity is not found in a registry."""

    def __init__(self, registry: str, id: str) -> None:
        """Initialize the error with registry and entity ID."""
        super().__init__(f"Registry {registry} does not contain an entity with ID {id}.")
