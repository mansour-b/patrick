from patrick.core import DataSource
from patrick.interfaces import Repository
from patrick.repositories.local import (
    LocalFrameRepository,
    LocalMovieRepository,
    LocalNNModelRepository,
    OSFMovieRepository,
    OSFNNModelRepository,
)


def repository_factory(data_source: DataSource, name: str) -> Repository:

    repo_class_dict = {
        "local": {
            "input_frames": LocalFrameRepository,
            "output_frames": LocalFrameRepository,
            "input_movies": LocalMovieRepository,
            "output_movies": LocalMovieRepository,
            "models": LocalNNModelRepository,
        },
        "osf": {
            "input_movies": OSFMovieRepository,
            "models": OSFNNModelRepository,
        },
    }
    repo_class = repo_class_dict[data_source][name]
    return repo_class(name)
