from abc import ABC, abstractmethod


class Metadata(ABC):

    @staticmethod
    @abstractmethod
    def printable_fields():
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls):
        pass

    def to_dict(self):
        return {
            attribute: getattr(self, attribute) for attribute in self.printable_fields()
        }

    def __repr__(self):
        attribute_str = ", ".join([f"{k}={v}" for k, v in self.to_dict().items()])
        return f"{type(self).__name__}({attribute_str})"

    def __str__(self):
        attribute_str = ",\n    ".join([f"{k}={v}" for k, v in self.to_dict().items()])
        return f"{type(self).__name__}(\n    {attribute_str})"
