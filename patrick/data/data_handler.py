import json
from abc import ABC, abstractmethod
from xml.etree.ElementTree import Element


class DataHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_xml(cls, data_xml: Element):
        pass

    @staticmethod
    @abstractmethod
    def _printable_fields():
        pass

    def to_dict(self):
        return {
            attribute: getattr(self, f"_{attribute}")
            for attribute in self._printable_fields()
        }

    def __repr__(self):
        attribute_str = ", ".join([f"{k}={v}" for k, v in self.to_dict().items()])
        return f"{type(self).__name__}({attribute_str})"

    def __str__(self):
        attribute_str = ",\n    ".join([f"{k}={v}" for k, v in self.to_dict().items()])
        return f"{type(self).__name__}(\n    {attribute_str})"
