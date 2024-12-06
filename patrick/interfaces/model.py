from patrick.interfaces.builder import Builder
from patrick.interfaces.repository import Repository


class ModelBuilder(Builder):
    _model_repository: Repository
