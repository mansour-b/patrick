from patrick import Frame, Model, Movie


def detect_structures_on_frame(frame: Frame, model: Model) -> Frame:
    detections = model.predict(frame)
    return Frame(
        name=frame.name,
        width=frame.width,
        height=frame.height,
        annotation_list=detections,
    )


def detect_structures_in_movie(movie: Movie, model: Model) -> Movie:
    return Movie(
        frame_list=[
            detect_structures_on_frame(frame, model) for frame in movie.frame_list
        ],
        track_list=[],
    )
