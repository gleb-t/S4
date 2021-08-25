"""
A collection of common functions used for rendering various ensemble figures.
"""
import itertools
import os
import glob
import re
import shutil
import subprocess
from typing import Callable, List, Optional

from deprecation import deprecated
from matplotlib.colors import ListedColormap

import PythonExtras.cairo_extras
import PythonExtras.file_tools as file_tools
import PythonExtras.volume_tools as volume_tools
import PythonExtras.rendering_tools as rendering_tools
from PythonExtras.rendering_tools import ImageRenderer
from PythonExtras.volume_tools import TTFFunc


# A colormap for flow data that I roughly scraped from the paper
# "A Fluid Flow Data Set for Machine Learning and its Application to Neural Flow Map Interpolation".
coldHotFlowCmap = ListedColormap([[0.239, 0.262, 0.611], [0.192, 0.219, 0.588], [0.196, 0.223, 0.592], [0.196, 0.227, 0.592], [0.196, 0.231, 0.592], [0.200, 0.235, 0.596], [0.200, 0.239, 0.596], [0.200, 0.243, 0.596], [0.203, 0.247, 0.600], [0.203, 0.250, 0.603], [0.203, 0.254, 0.603], [0.203, 0.258, 0.607], [0.207, 0.262, 0.611], [0.207, 0.266, 0.611], [0.207, 0.270, 0.615], [0.211, 0.274, 0.615], [0.211, 0.278, 0.615], [0.211, 0.282, 0.615], [0.215, 0.290, 0.619], [0.219, 0.301, 0.627], [0.219, 0.305, 0.631], [0.219, 0.309, 0.631], [0.223, 0.313, 0.635], [0.223, 0.321, 0.635], [0.231, 0.329, 0.643], [0.231, 0.333, 0.643], [0.231, 0.341, 0.647],
                                  [0.231, 0.345, 0.650], [0.235, 0.352, 0.654], [0.235, 0.356, 0.654], [0.239, 0.364, 0.658], [0.239, 0.368, 0.658], [0.243, 0.380, 0.662], [0.247, 0.388, 0.670], [0.247, 0.392, 0.670], [0.250, 0.400, 0.674], [0.250, 0.407, 0.678], [0.254, 0.415, 0.686], [0.258, 0.423, 0.686], [0.258, 0.427, 0.686], [0.262, 0.435, 0.690], [0.266, 0.443, 0.698], [0.266, 0.450, 0.701], [0.274, 0.462, 0.705], [0.278, 0.470, 0.709], [0.286, 0.478, 0.713], [0.294, 0.486, 0.717], [0.298, 0.490, 0.721], [0.305, 0.501, 0.729], [0.309, 0.505, 0.729], [0.313, 0.513, 0.733], [0.325, 0.525, 0.737], [0.333, 0.533, 0.741], [0.337, 0.537, 0.745],
                                  [0.345, 0.545, 0.749], [0.349, 0.556, 0.752], [0.360, 0.564, 0.760], [0.368, 0.576, 0.764], [0.372, 0.580, 0.768], [0.380, 0.592, 0.772], [0.384, 0.596, 0.776], [0.396, 0.607, 0.780], [0.403, 0.615, 0.788], [0.407, 0.623, 0.788], [0.419, 0.639, 0.796], [0.427, 0.647, 0.800], [0.439, 0.658, 0.807], [0.443, 0.662, 0.811], [0.447, 0.670, 0.811], [0.454, 0.678, 0.815], [0.462, 0.686, 0.819], [0.474, 0.694, 0.823], [0.486, 0.701, 0.831], [0.490, 0.705, 0.835], [0.498, 0.713, 0.835], [0.513, 0.725, 0.843], [0.525, 0.733, 0.847], [0.529, 0.737, 0.850], [0.541, 0.745, 0.854], [0.552, 0.756, 0.858], [0.560, 0.764, 0.866],
                                  [0.576, 0.776, 0.870], [0.584, 0.780, 0.874], [0.596, 0.792, 0.878], [0.607, 0.800, 0.882], [0.619, 0.807, 0.890], [0.631, 0.815, 0.890], [0.635, 0.819, 0.894], [0.647, 0.831, 0.901], [0.658, 0.839, 0.905], [0.670, 0.847, 0.909], [0.682, 0.854, 0.913], [0.686, 0.858, 0.917], [0.698, 0.862, 0.921], [0.709, 0.870, 0.925], [0.721, 0.874, 0.929], [0.733, 0.878, 0.929], [0.737, 0.882, 0.929], [0.752, 0.886, 0.937], [0.760, 0.894, 0.941], [0.776, 0.898, 0.941], [0.792, 0.905, 0.945], [0.796, 0.909, 0.949], [0.807, 0.913, 0.949], [0.819, 0.921, 0.956], [0.831, 0.925, 0.956], [0.835, 0.929, 0.960], [0.847, 0.937, 0.960],
                                  [0.858, 0.941, 0.964], [0.870, 0.945, 0.960], [0.882, 0.952, 0.952], [0.886, 0.952, 0.945], [0.894, 0.956, 0.933], [0.901, 0.960, 0.921], [0.909, 0.964, 0.905], [0.913, 0.964, 0.890], [0.921, 0.968, 0.882], [0.925, 0.968, 0.874], [0.933, 0.972, 0.862], [0.941, 0.976, 0.847], [0.949, 0.980, 0.831], [0.952, 0.980, 0.827], [0.956, 0.984, 0.811], [0.968, 0.984, 0.800], [0.976, 0.988, 0.784], [0.984, 0.992, 0.768], [0.988, 0.992, 0.760], [0.992, 0.996, 0.749], [0.996, 0.988, 0.737], [0.996, 0.984, 0.725], [0.996, 0.980, 0.717], [0.996, 0.968, 0.705], [0.996, 0.964, 0.698], [0.996, 0.956, 0.686], [0.996, 0.949, 0.674],
                                  [0.996, 0.945, 0.666], [0.996, 0.937, 0.654], [0.996, 0.929, 0.643], [0.996, 0.921, 0.631], [0.996, 0.913, 0.619], [0.996, 0.909, 0.615], [0.996, 0.901, 0.603], [0.996, 0.894, 0.592], [0.996, 0.886, 0.580], [0.996, 0.878, 0.568], [0.996, 0.874, 0.560], [0.992, 0.862, 0.549], [0.992, 0.850, 0.537], [0.992, 0.839, 0.525], [0.992, 0.823, 0.513], [0.992, 0.819, 0.505], [0.992, 0.807, 0.494], [0.992, 0.792, 0.482], [0.992, 0.784, 0.474], [0.992, 0.776, 0.470], [0.992, 0.764, 0.458], [0.992, 0.752, 0.447], [0.992, 0.737, 0.431], [0.992, 0.725, 0.419], [0.992, 0.717, 0.415], [0.992, 0.709, 0.403], [0.992, 0.694, 0.392],
                                  [0.992, 0.682, 0.380], [0.988, 0.670, 0.376], [0.988, 0.662, 0.372], [0.988, 0.647, 0.364], [0.984, 0.631, 0.356], [0.980, 0.615, 0.349], [0.980, 0.603, 0.341], [0.980, 0.596, 0.337], [0.976, 0.584, 0.333], [0.972, 0.568, 0.325], [0.972, 0.556, 0.317], [0.968, 0.537, 0.309], [0.968, 0.529, 0.309], [0.964, 0.513, 0.301], [0.964, 0.501, 0.298], [0.960, 0.490, 0.290], [0.960, 0.474, 0.282], [0.960, 0.470, 0.278], [0.956, 0.458, 0.274], [0.956, 0.443, 0.266], [0.952, 0.427, 0.262], [0.952, 0.423, 0.258], [0.945, 0.411, 0.250], [0.941, 0.396, 0.247], [0.933, 0.380, 0.239], [0.929, 0.368, 0.235], [0.925, 0.364, 0.231],
                                  [0.917, 0.349, 0.227], [0.913, 0.341, 0.223], [0.909, 0.333, 0.219], [0.905, 0.325, 0.211], [0.901, 0.317, 0.207], [0.898, 0.301, 0.203], [0.890, 0.294, 0.200], [0.882, 0.282, 0.192], [0.882, 0.274, 0.192], [0.878, 0.266, 0.188], [0.874, 0.258, 0.184], [0.870, 0.247, 0.180], [0.862, 0.235, 0.172], [0.862, 0.227, 0.168], [0.858, 0.223, 0.164], [0.850, 0.211, 0.160], [0.847, 0.200, 0.156], [0.843, 0.192, 0.152], [0.843, 0.188, 0.152], [0.835, 0.180, 0.149], [0.827, 0.172, 0.149], [0.819, 0.168, 0.149], [0.819, 0.164, 0.149], [0.815, 0.160, 0.149], [0.807, 0.152, 0.149], [0.803, 0.149, 0.149], [0.796, 0.141, 0.149],
                                  [0.788, 0.133, 0.149], [0.784, 0.129, 0.149], [0.776, 0.125, 0.149], [0.772, 0.121, 0.149], [0.768, 0.117, 0.149], [0.764, 0.113, 0.149], [0.764, 0.113, 0.149], [0.752, 0.101, 0.149], [0.749, 0.098, 0.149], [0.745, 0.094, 0.149], [0.741, 0.090, 0.149], [0.741, 0.090, 0.149], [0.733, 0.082, 0.149], [0.729, 0.074, 0.149], [0.721, 0.070, 0.149], [0.721, 0.070, 0.149], [0.717, 0.066, 0.149], [0.713, 0.062, 0.149], [0.713, 0.062, 0.149], [0.705, 0.058, 0.149], [0.705, 0.058, 0.149], [0.698, 0.050, 0.149], [0.698, 0.050, 0.149], [0.698, 0.050, 0.149], [0.694, 0.047, 0.149], [0.690, 0.043, 0.149], [0.690, 0.043, 0.149],
                                  [0.690, 0.043, 0.149], [0.686, 0.039, 0.149], [0.682, 0.035, 0.149], [0.682, 0.035, 0.149], [0.678, 0.031, 0.149], [0.674, 0.027, 0.149], [0.674, 0.027, 0.149], [0.674, 0.027, 0.149], [0.674, 0.027, 0.149], [0.674, 0.027, 0.149], [0.674, 0.027, 0.149], [0.666, 0.019, 0.149]])


coldHotCylinderCmap = ListedColormap([[0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.238, 0.261, 0.611], [0.241, 0.273, 0.616], [0.244, 0.285, 0.622], [0.246, 0.297, 0.627], [0.249, 0.309, 0.633], [0.251, 0.321, 0.638], [0.254, 0.332, 0.644], [0.257, 0.344, 0.650], [0.259, 0.356, 0.655], [0.262, 0.368, 0.661], [0.264, 0.380, 0.666], [0.267, 0.392, 0.672], [0.269, 0.404, 0.678], [0.272, 0.415, 0.683], [0.275, 0.427, 0.689], [0.277, 0.439, 0.694], [0.280, 0.451, 0.700],
                                      [0.282, 0.463, 0.706], [0.285, 0.475, 0.711], [0.310, 0.496, 0.716], [0.342, 0.520, 0.721], [0.374, 0.544, 0.725], [0.406, 0.569, 0.730], [0.438, 0.593, 0.735], [0.470, 0.617, 0.739], [0.502, 0.641, 0.744], [0.534, 0.665, 0.749], [0.566, 0.689, 0.754], [0.598, 0.713, 0.758], [0.630, 0.738, 0.763], [0.662, 0.762, 0.768], [0.694, 0.786, 0.772], [0.726, 0.810, 0.777], [0.758, 0.834, 0.782], [0.790, 0.858, 0.786], [0.822, 0.883, 0.791], [0.854, 0.907, 0.796], [0.886, 0.931, 0.800], [0.918, 0.955, 0.805], [0.950, 0.979, 0.810], [0.955, 0.959, 0.787], [0.955, 0.930, 0.758], [0.955, 0.901, 0.729], [0.955, 0.871, 0.700],
                                      [0.954, 0.842, 0.671], [0.954, 0.813, 0.642], [0.954, 0.783, 0.613], [0.954, 0.754, 0.584], [0.954, 0.725, 0.555], [0.953, 0.695, 0.527], [0.953, 0.666, 0.498], [0.953, 0.637, 0.469], [0.953, 0.607, 0.440], [0.953, 0.578, 0.411], [0.952, 0.549, 0.382], [0.952, 0.519, 0.353], [0.952, 0.490, 0.324], [0.952, 0.461, 0.295], [0.952, 0.431, 0.266], [0.942, 0.408, 0.254], [0.927, 0.388, 0.248], [0.913, 0.368, 0.243], [0.899, 0.348, 0.237], [0.884, 0.328, 0.232], [0.870, 0.308, 0.227], [0.856, 0.287, 0.221], [0.842, 0.267, 0.216], [0.827, 0.247, 0.210], [0.813, 0.227, 0.205], [0.799, 0.207, 0.199], [0.785, 0.187, 0.194],
                                      [0.770, 0.166, 0.188], [0.756, 0.146, 0.183], [0.742, 0.126, 0.178], [0.727, 0.106, 0.172], [0.713, 0.086, 0.167], [0.699, 0.066, 0.161], [0.685, 0.046, 0.156], [0.670, 0.025, 0.150], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149],
                                      [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149], [0.666, 0.018, 0.149]])


class MemberImageRenderer:

    def __init__(self, imageRenderer: Optional[ImageRenderer] = None):
        self.renderer = imageRenderer or ImageRenderer()
        self.dataCache = {}

    def render_member_image(self, memberPath: str, frameIndex: int, tfFunc: TTFFunc):
        if memberPath not in self.dataCache:
            self.dataCache[memberPath] = volume_tools.load_volume_data_from_dat(memberPath)

        # This is suboptimal, we should first check if a rendered image is cached and only then slice the member data.
        imageHash = '{}_{}'.format(memberPath, frameIndex)
        memberData = self.dataCache[memberPath]
        frameData = memberData[frameIndex] if frameIndex < memberData.shape[0] else memberData[-1]

        return self.renderer.render_image(frameData, imageHash, tfFunc)


@deprecated("Use MemberImageRenderer instead.")
def get_member_image(memberPath, frameIndex, tfFunc: TTFFunc):
    # WE ASSUME HERE THAT TF DOESN'T CHANGE DYNAMICALLY. (same func == same TF)
    # Cache member images, data and tf in this function's attribute. Not elegant, but simple.
    if 'imageCache' not in get_member_image.__dict__:
        get_member_image.imageCache = {}
        get_member_image.dataCache = {}

    if 'tfFunc' not in get_member_image.__dict__ or tfFunc != get_member_image.tfFunc:
        get_member_image.tfCache = rendering_tools.precompute_tf_uint8(tfFunc)
        get_member_image.tfFunc = tfFunc
        get_member_image.imageCache = {}  # Reset the image cache, since the TF has changed.
        print("Recomputing the TF cache.")

    imageKey = (memberPath, frameIndex)
    if imageKey not in get_member_image.imageCache:
        if memberPath not in get_member_image.dataCache:
            get_member_image.dataCache[memberPath] = volume_tools.load_volume_data_from_dat(memberPath)

        data = get_member_image.dataCache[memberPath]
        # Take the last frame, if the frame index is too large.
        data = data[frameIndex] if frameIndex < data.shape[0] else data[-1]
        # Drop the temporal and Z dimensions to get a 2D+C image.
        image = rendering_tools.render_2d_volume_to_rgba(data, get_member_image.tfCache)[0, 0]
        # Store to the cache.
        get_member_image.imageCache[imageKey] = image

    return get_member_image.imageCache[imageKey]


def get_member_image_surface(memberPath: str, frameIndex: int, tf: Callable):
    """
    Renders a frame of an ensemble member into a cairo image surface.
    :return:
    """
    image = get_member_image(memberPath, frameIndex, tf)
    return PythonExtras.cairo_extras.image_rgba_uint8_to_cairo(image)


def get_params_from_member_name(memberName):
    chunks = re.split(r'[_,\-\s]', memberName)
    if len(chunks) <= 1:
        labels = []
    else:
        labels = [c for c in chunks if len(c) <= 8]

    return [l.lstrip('0 ') for l in labels]


def get_ensemble_labeler(ensembleName: str) -> Callable[[str], List[str]]:
    import regex
    import itertools

    def _build_labeler(pattern: str) -> Callable[[str], List[str]]:
        def _labeler(name: str) -> List[str]:
            match = regex.match(pattern, name)
            if match:
                # Fetch all the captures from all the groups.
                capturesPerGroup = (match.captures(i) for i in range(1, len(match.groups()) + 1))
                return list(itertools.chain(*capturesPerGroup))
            else:
                raise ValueError("Failed to parse the name '{}' using the pattern '{}'.".format(name, pattern))

        return _labeler

    if ensembleName == 'wetting':
        return _build_labeler(r'(Ca)=([\d\-]+),(M)=([\d]+)')
    elif ensembleName == 'droplet-splash':
        return _build_labeler(r'([^_]+)_view.*')
    else:
        raise ValueError("Unknown ensemble name: '{}'.".format(ensembleName))


def get_max_member_size(memberDataDirPath, tf=None):
    impostorMaxWidth, impostorMaxHeight = 0, 0
    if tf is None:
        tf = lambda x: (1.0, 1.0, 1.0, 1.0)
    # Check the first 10 members to save time.
    for impostorFilename in itertools.islice(glob.glob(os.path.join(memberDataDirPath, '*.dat')), 10):
        impostorPath = os.path.join(memberDataDirPath, impostorFilename)
        with get_member_image_surface(impostorPath, 0, tf) as someImpostor:
            impostorWidthOrig = someImpostor.get_width()
            impostorHeightOrig = someImpostor.get_height()
            if impostorWidthOrig > impostorMaxWidth:
                impostorMaxWidth = impostorWidthOrig
            if impostorHeightOrig > impostorMaxHeight:
                impostorMaxHeight = impostorHeightOrig

    return impostorMaxHeight, impostorMaxWidth


def get_max_member_duration(memberDataDirPath: str) -> int:
    memberPaths = glob.iglob(os.path.join(memberDataDirPath, '*.dat'))
    memberDurations = (volume_tools.VolumeMetadata.load_from_dat(p).frameNumber for p in memberPaths)

    return max(memberDurations)


def get_ensemble_tf(ensembleName: str) -> TTFFunc:
    import matplotlib.pyplot as plt

    viridisColormap = plt.get_cmap('viridis')
    if ensembleName == 'cylinder':
        # return lambda x: viridisColormap(int(x * 2))
        # return lambda x: coldHotFlowCmap(int(x * 3))
        return lambda x: coldHotCylinderCmap(int(x))
    elif ensembleName == 'cylinder-300':
        # return lambda x: viridisColormap(int(x * 3))
        # return lambda x: coldHotFlowCmap(int((x - 30) * 5))
        return lambda x: coldHotCylinderCmap(int(x))
    elif ensembleName == 'hotroom2D':
        return lambda x: viridisColormap(int((x - 220) * 255 / (255 - 220)))
    elif ensembleName == 'parabolicCircles':
        # Generate a 'saw' TF to highlight isolines.
        def _tf(x):
            if int(x / 8) % 2 == 0:
                return viridisColormap(0)
            else:
                return viridisColormap(x)

        return _tf
    elif ensembleName == 'wetting':
        return lambda x: viridisColormap(int(x))
    elif ensembleName == 'droplet-splash':
        grayColormap = plt.get_cmap('gray')
        return lambda x: grayColormap(int(x))
    else:
        raise RuntimeError("Unknown ensemble name: '{}'".format(ensembleName))


def get_tf(tfName: str) -> TTFFunc:
    if tfName == 'bw':
        return lambda x: (0., 0., 0., 1.) if x == 0 else (1., 1., 1., 1.)
    else:
        raise RuntimeError("Unknown TF name: '{}'".format(tfName))


def render_figure_video(outputDirPath: str,
                        outputVideoPath: str,
                        renderFrameFunc: Callable[[int, str], None],
                        frameNumber: int):
    file_tools.prepare_output_dir(outputDirPath)

    for f in range(frameNumber):
        print("Rendering frame {}/{}".format(f + 1, frameNumber))
        outputImagePath = os.path.join(outputDirPath, 'frame-{:06d}.png'.format(f))
        renderFrameFunc(f, outputImagePath)

    compile_video(outputDirPath, outputVideoPath)


def compile_video(outputDirPath, outputVideoPath, filenameFormat: str = 'frame-%06d.png'):
    if not os.path.exists('ffmpeg.exe'):
        print('FFMpeg not in PATH. Skipped video compilation.')
        return

    print("Compiling a video.")
    tempVideoPath = os.path.join(outputDirPath, 'video_temp.avi')
    subprocess.check_output(
        ['ffmpeg.exe',
         '-f', 'image2',
         '-framerate', '10',
         '-i', filenameFormat,
         '-y',
         '-b:v', '20M',
         tempVideoPath],
        cwd=outputDirPath
    )
    print("Copying the video.")
    file_tools.create_dir(os.path.dirname(outputVideoPath))
    shutil.copy(tempVideoPath, outputVideoPath)
