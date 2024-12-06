from __future__ import annotations

from patrick.core import Model, Movie
from patrick.core.value_objects import ComputingDevice, DataSource, Framework
from patrick.interfaces.cnn import NetBuilder
from patrick.interfaces.model import ModelBuilder
from patrick.interfaces.repository import Repository
from patrick.repositories.local import LocalModelRepository, LocalTorchNetRepository


def model_repository_factory(data_source: DataSource) -> Repository:
    class_dict = {"local": LocalModelRepository}
    return class_dict[data_source]()


def net_repository_factory(data_source: DataSource, framework: Framework) -> Repository:
    class_dict = {"local": {"torch": LocalTorchNetRepository}}
    concrete_class = class_dict[data_source][framework]
    return concrete_class()


def load_model(
    model_name: str,
    data_source: DataSource,
    framework: Framework,
    device: ComputingDevice,
) -> Model:

    net_repository = net_repository_factory(data_source, framework)
    net_builder = NetBuilder(device, net_repository)

    model_repository = model_repository_factory(data_source)
    model_repository._net_builder = net_builder

    model_builder = ModelBuilder(model_name, model_repository)

    return model_builder.build()


def load_movie(movie_name: str, data_source: DataSource) -> Movie:
    pass


def compute_predictions(model: Model, movie: Movie) -> Movie:
    pass


def save_movie(movie: Movie) -> None:
    pass


if __name__ == "__main__":

    movie_name = "blob"
    model_name = "model_architecture_yymmdd_HHMMSS"
    data_source = "local"
    framework = "torch"
    computing_device = "gpu"

    model = load_model(
        model_name=model_name,
        data_source=data_source,
        framework=framework,
        device=computing_device,
    )

    movie = load_movie(movie_name=movie_name, data_source=data_source)

    analysed_movie = compute_predictions(model, movie)

    save_movie(analysed_movie)
