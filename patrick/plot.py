import time

import imageio
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from patrick import PATRICK_DIR_PATH
from patrick.data.annotation import Annotation, Polyline
from patrick.data.image import Image


def make_movie(image_list: list[Image], image_dir_name: str, fps: int):

    time_str = time.strftime("%y%m%d_%H%M%S")
    movie_path = PATRICK_DIR_PATH / f"misc/image_dir_name_{time_str}"

    with imageio.get_writer(movie_path, fps=fps) as writer:
        for image in image_list:
            fig = plot_image(image, image_dir_name, return_figure=True)
            writer.append_data(mplfig_to_npimage(fig))


def plot_image(
    image: Image,
    image_dir_name: str = None,
    cmap: str = "gray",
    annotation_color: str = "tab:red",
    return_figure: bool = False,
):
    image_array = image.get_image_array(image_dir_name)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_array, cmap=cmap)
    ax.axis("off")
    for annotation in image._annotations:
        plot_annotation(ax, annotation, color=annotation_color)
    fig.show()
    if return_figure:
        return fig


def plot_annotation(ax, annotation: Annotation, color: str):
    annotation_type_dict = {"polyline": plot_polyline}
    plot_function = annotation_type_dict[annotation.type]
    plot_function(ax, annotation, color)


def plot_polyline(ax, polyline: Polyline, color: str):
    point_list = polyline._point_list
    for i, point_1 in enumerate(point_list[:-1]):
        point_2 = point_list[i + 1]
        x1, y1 = point_1
        x2, y2 = point_2
        ax.plot([x1, x2], [y1, y2], color=color)
