from patrick import Movie, Tracker


def track_structures(movie: Movie, tracker: Tracker) -> Movie:
    tracks = tracker.make_tracks(movie)
    return Movie(frame_list=movie.frame_list, track_list=tracks)
