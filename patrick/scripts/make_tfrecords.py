from argparse import ArgumentParser

from patrick.data.tfrecord import make_tfrecords

parser = ArgumentParser()

parser.add_argument("--experiment", help="Name of the simulation, e.g., 'blob_i")
parser.add_argument("--image_width", type=int, default=512, help="Image width")
parser.add_argument("--image_height", type=int, default=512, help="Image height")

args = parser.parse_args()

make_tfrecords(
    args.experiment, image_width=args.image_width, image_height=args.image_height
)
