def load_model():
    pass


def load_movie():
    pass


def compute_predictions():
    pass


def save_movie():
    pass


if __name__ == "__main__":
    movie_name = "blob"

    model_name = "model_architecture_yymmdd_HHMMSS"

    data_source = "local"

    framework = "torch"

    computing_device = "gpu"

    model = load_model(model_name)

    movie = load_movie(movie_name)

    analysed_movie = compute_predictions(model, movie)

    save_movie(analysed_movie)
