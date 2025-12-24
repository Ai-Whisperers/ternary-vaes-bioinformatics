import abc


class BaseFactory(abc.ABC):
    """Abstract base class for all factories."""

    @classmethod
    @abc.abstractmethod
    def build(cls, **kwargs):
        """Builds an instance (in memory)."""
        pass

    @classmethod
    def create(cls, **kwargs):
        """Creates an instance (persisted/mocked)."""
        instance = cls.build(**kwargs)
        # Add persistence logic here if needed
        return instance
