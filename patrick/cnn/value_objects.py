from enum import Enum


class Framework(Enum):
    torch = "torch"


class DataSource(Enum):
    local = "local"


class ComputingDevice(Enum):
    cpu = "cpu"
    gpu = "gpu"
