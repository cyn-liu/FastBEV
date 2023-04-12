from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
from matplotlib.axes import Axes
from torch import Tensor
import cv2
from matplotlib.patches import Rectangle
from einops import rearrange

from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.structures.points.base_points import BasePoints
from mmdet3d.structures.points import LiDARPoints

__all__ = ['get_visualize_sample']


def get_visualize_sample(
        pointclouds: Union[LiDARPoints, Tensor],
        boxes_3d: LiDARInstance3DBoxes,
        labels: np.ndarray,
        show_range: List[float] = [-50, -50, 50, 50],
        vis_config: Dict = None,
        title: str = 'LIDARTOP') -> np.ndarray:
    # Init axes.
    fig, ax = plt.subplots(figsize=(24, 24))

    if pointclouds is None:
        ax.set_facecolor('gray')  # black
    else:
        # Show point cloud.
        if isinstance(pointclouds, BasePoints):
            points = np.transpose(pointclouds.coord.numpy())
        elif isinstance(pointclouds, Tensor):
            points = np.transpose(pointclouds[:, :3].cpu().numpy())
        else:
            assert isinstance(pointclouds, np.ndarray), f"pointclouds must be one of \
                [BasePoints, Tensor, np.ndarray], but got {type(pointclouds)}"
            points = rearrange(pointclouds[:, :3], 'n c -> c n')

        pts = view_points(points, np.eye(4), normalize=False)
        dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / max(show_range))
        ax.scatter(pts[0, :], pts[1, :], c=colors, s=0.2)

    if boxes_3d is not None:
        render(boxes_3d, labels, ax, vis_config, view=np.eye(4))

    ax.set_xlim(show_range[0] - 3, show_range[2] + 3)
    ax.set_ylim(show_range[1] - 3, show_range[3] + 3)
    rect = Rectangle((-1, -1.5), 2, 3, linewidth=4, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.title(title)
    # if savepath is not None:
    #     plt.savefig(savepath)
    #     plt.close()
    # else:
    #     plt.show()

    fig.canvas.draw()
    plt_image_array = np.array(fig.canvas.renderer.buffer_rgba())
    rgb_image = cv2.cvtColor(plt_image_array, cv2.COLOR_RGBA2RGB)
    return rgb_image

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3, f"Invalid points shape: {points.shape}"

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def render(boxes_3d: LiDARInstance3DBoxes,
           labels: np.ndarray,
           axis: Axes,
           vis_config: Dict,
           view: np.ndarray = np.eye(3),
           normalize: bool = False):
    n_corners = np.transpose(boxes_3d.corners.cpu().numpy(), (0, 2, 1))

    for idx, corner in enumerate(n_corners):
        corners = view_points(corner, view, normalize=normalize)[:2, :]
        label = labels[idx]
        color = '#{:02X}{:02X}{:02X}'.format(*[min(max(0, c), 255) for c in vis_config[label]['color']])
        linewidth = vis_config[label]['thickness'] + 1

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=color, linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], color)
        draw_rect(corners.T[4:], color)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=color, linewidth=linewidth)