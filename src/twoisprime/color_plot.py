# -*- coding: utf-8 -*-
"""
This is a script that analyses and displays color information of an input image.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


import argparse
import sys
import logging

from twoisprime import __version__

__author__ = "twoisprime"
__copyright__ = "twoisprime"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def color_plot(image_path, resolution=100):
    """Plot color information in HSV space

    Args:
      image_path (str): path to image file
      resolution (int): resize value that determines the amount of pixels to consider
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    small = cv2.resize(img, (resolution, resolution), interpolation = cv2.INTER_AREA)
    pixel_colors = small.reshape((np.shape(small)[0]*np.shape(small)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Color information script.")
    parser.add_argument(
        "--version",
        action="version",
        version="twoisprime {ver}".format(ver=__version__))
    parser.add_argument(
        "-i",
        "--image",
        dest="image",
        help="path to image file",
        type=str,
        metavar="STR")
    parser.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        help="resize resolution",
        type=int,
        metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Processing...")
    color_plot(args.image, args.resolution)
    _logger.info("Done")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
