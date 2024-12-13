from patrick.core import ComputingDevice, DataSource, Framework
from patrick.interfaces import Repository
from patrick.interfaces.cnn import NetBuilder, TorchNetBuilder
from patrick.repositories.local import (
    LocalModelRepository,
    LocalMovieRepository,
    LocalTorchNetRepository,
)


def model_repository_factory(data_source: DataSource) -> Repository:
    class_dict = {"local": LocalModelRepository}
    return class_dict[data_source]()


def net_repository_factory(
    data_source: DataSource, framework: Framework, device: ComputingDevice
) -> Repository:
    class_dict = {"local": {"torch": LocalTorchNetRepository}}
    concrete_class = class_dict[data_source][framework]
    return concrete_class(device)


def movie_repository_factory(data_source: DataSource, name: str) -> Repository:
    class_dict = {"local": LocalMovieRepository}
    return class_dict[data_source](name)


def net_builder_factory(
    framework: Framework,
    device: ComputingDevice,
    net_repository: Repository,
) -> NetBuilder:
    class_dict = {"torch": TorchNetBuilder}
    return class_dict[framework](device, net_repository)
