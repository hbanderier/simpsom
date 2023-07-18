from typing import Union, Collection, Tuple
from nptyping import NDArray

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from matplotlib.ticker import MaxNLocator
from itertools import product
from pylettes import Distinct20
from scipy.interpolate import LinearNDInterpolator


def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def tile(
    polygons: str,
    coor: Tuple[float],
    color: Tuple[float],
    edgecolor: Tuple[float] = None,
    alpha: float = .1,
    linewidth: float = 1.,
) -> RegularPolygon:
    """Set the tile shape for plotting.

    Args:
        polygons (str): type of polygons, case-insensitive
        coor (tuple[float, float]): positon of the tile in the plot figure.
        color (tuple[float,...]): color tuple.
        edgecolor (tuple[float,...]): border color tuple.

    Returns:
        (matplotlib patch object): the tile to add to the plot.
    """
    if polygons.lower() == "rectangle":
        return RegularPolygon(
            coor,
            numVertices=4,
            radius=0.95 / np.sqrt(2),
            orientation=np.radians(45),
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    elif polygons.lower() == "hexagons":
        return RegularPolygon(
            coor,
            numVertices=6,
            radius=0.95 / np.sqrt(3),
            orientation=np.radians(0),
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    else:
        raise NotImplementedError("Only hexagons and rectangles")


def draw_polygons(
    polygons: str,
    fig: Figure,
    centers: Collection[float],
    feature: Collection[float],
    ax: Axes = None,
    cmap: ListedColormap | None | str = None,
    norm: Normalize | None = None,
    edgecolors: Tuple[float] | Collection[Tuple] = None,
    alphas: Collection[float] | float | int = None,
    linewidths: Collection[float] | float | int = 1.,
) -> Axes:
    """Draw a grid based on the selected tiling, nodes positions and color the tiles according to a given feature.
    
    Args:
        polygons_class (str): type of polygons, case-insensitive
        fig (matplotlib figure object): the figure on which the grid will be plotted.
        centers (list, float): array containing couples of coordinates for each cell
            to be plotted in the Hexagonal tiling space.
        feature (list, float): array contaning informations on the weigths of each cell,
            to be plotted as colors.
        cmap (ListedColormap): a custom color map.

    Returns:
        ax (matplotlib axis object): the axis on which the hexagonal grid has been plotted.
    """
    if ax is None:
        ax = fig.add_subplot(111, aspect="equal")
    centers = np.asarray(centers)
    xpoints = centers[:, 0]
    ypoints = centers[:, 1]
    patches = []

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    elif cmap is None:
        cmap = plt.get_cmap("viridis")
    cmap.set_bad(color="#ffffff", alpha=1.0)
    
    if np.isnan(feature).all():
        edgecolors = "#555555"
    
    if isinstance(edgecolors, str) or (edgecolors is None) or (len(edgecolors) == 3):
        edgecolors = [edgecolors] * len(feature)
        
    if alphas is None:
        alphas = 1.
        
    if isinstance(alphas, int | float):
        alphas = [alphas] * len(feature)
        
    if linewidths is None:
        linewidths = 1.
        
    if isinstance(linewidths, int | float):
        linewidths = [linewidths] * len(feature)

    for x, y, f, ec, alpha, linewidth in zip(xpoints, ypoints, feature, edgecolors, alphas, linewidths):
        if norm is not None:
            color = cmap(norm(f))
        else:
            color = cmap(f)
        patches.append(tile(polygons, (x, y), color=color, edgecolor=ec, alpha=alpha, linewidth=linewidth))

    pc = PatchCollection(patches, match_original=True, cmap=cmap, norm=norm)
    pc.set_array(np.array(feature))
    ax.add_collection(pc)

    dy = 1 / np.sqrt(3) if polygons == 'hexagons' else 1 / np.sqrt(2)
    ax.set_xlim(xpoints[0] - 0.5, xpoints[-1] + 0.5)
    ax.set_ylim(ypoints[0] - dy,  ypoints[-1] + dy)
    ax.axis('off')
    
    return ax


def plot_map(
    centers: Collection[np.ndarray],
    feature: Collection[np.ndarray],
    polygons: str,
    show: bool = True,
    print_out: bool = False,
    file_name: str = "./som_plot.png",
    fig: Figure = None,
    ax: Axes = None,
    draw_cbar: bool = True,
    cbar_kwargs: dict = None,
    **kwargs: Tuple
) -> Tuple[Figure, Axes]:
    """Plot a 2D SOM

    Args:
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        feature (list or array): The SOM node feature defining the color map
            (e.g. node.weights, node.diff)
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.
            - norm (Normalize): a Normalizer

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 4)
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12
    if "cbar_label" in kwargs:
        cbar_kwargs["label"] = kwargs["cbar_label"] # backwards compatibility baby

    if fig is None:
        fig, ax = plt.subplots(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))
    ax = draw_polygons(
        polygons,
        fig,
        centers,
        feature,
        ax,
        cmap=kwargs["cmap"] if "cmap" in kwargs else plt.get_cmap("viridis"),
        norm=kwargs["norm"] if "norm" in kwargs else None,
        edgecolors=kwargs["edgecolors"] if "edgecolors" in kwargs else None,
        alphas=kwargs["alphas"] if "alphas" in kwargs else None,
        linewidths=kwargs["linewidths"] if "linewidths" in kwargs else None,
    )
    if 'title' in kwargs:
        ax.set_title(kwargs["title"], size=kwargs["fontsize"] * 1.15)


    if not np.isnan(feature).all() and (draw_cbar or cbar_kwargs):
        cbar = plt.colorbar(ax.collections[0], ax=ax, **cbar_kwargs)
        cbar.ax.tick_params(labelsize=kwargs["fontsize"] * 0.85)
        cbar.outline.set_visible(False)

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax


def add_cluster(
    fig: Figure,
    ax: Axes,
    coords: NDArray,
    clu_labs: NDArray,
    polygons: str = 'hexagons',
    cmap: str | Colormap | list | NDArray = None,
) -> Tuple[Figure, Axes]:
    unique_labs = np.unique(clu_labs)
    sym = np.any(unique_labs < 0)

    if cmap is None:
            cmap = "PiYG" if sym else "Greens"
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    nabove = np.sum(unique_labs > 0)
    if isinstance(cmap, list | NDArray):
        colors = cmap
    else:
        if sym:
            nbelow = np.sum(unique_labs < 0)
            cab = np.linspace(1, 0.66, nabove)
            cbe = np.linspace(0.33, 0, nbelow)
            if 0 in unique_labs:
                zerocol = [0.5]
            else:
                zerocol = []
            colors = [*cbe, *zerocol, *cab]
        else:
            if 0 in unique_labs:
                zerocol = [0.0]
            else:
                zerocol = []
            colors = np.linspace(1.0, 0.33, nabove)
            colors = [*zerocol, *colors]
        colors = cmap(colors)
    if polygons == 'rectangle':
        dx, dy = coords[1, :] - coords[0, :]
        gen = [(sgnx * dx / 2.2, sgny * dy / 2.2) for sgnx, sgny in product([-1, 0, 1], [-1, 0, 1])]
    elif polygons == 'hexagons':
        l = 0.85 / np.sqrt(3)
        gen = [(l * degcos(theta), l * degsin(theta)) for theta in range(30, 360, 60)]
    for coord, val in zip(coords, clu_labs):
        newcs = [[coord[0] + cx, coord[1] + cy] for cx, cy in gen]
        coords = np.append(coords, newcs, axis=0)
        clu_labs = np.append(clu_labs, [val] * len(newcs))
    minx, miny = np.amin(coords, axis=0)
    maxx, maxy = np.amax(coords, axis=0)
    x = np.linspace(minx - 1, maxx + 1, 101)
    y = np.linspace(miny - 1, maxy + 1, 101)

    for i, lab in enumerate(np.unique(clu_labs)):
        interp = LinearNDInterpolator(coords, clu_labs == lab)
        r = interp(*np.meshgrid(x, y)) 
        if lab == 0:
            ax.contourf(x, y, r, levels=[0.8, 1], colors='black', alpha=0.6)
        else:
            ax.contour(x, y, r, levels=[0.8], colors=[colors[i]], linewidths=4)
    return fig, ax


def line_plot(
    y_val: Union[np.ndarray, list],
    x_val: Union[np.ndarray, list] = None,
    show: bool = True,
    print_out: bool = False,
    file_name: str = "./line_plot.png",
    **kwargs: Tuple[int]
) -> Tuple[Figure, plt.Axes]:
    """A simple line plot with maplotlib.

    Args:
        y_val (array or list): values along the y axis.
        x_val (array or list): values along the x axis,
            if none, these will be inferred from the shape of y_val.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - xlabel (str): x-axis label,
            - ylabel (str): y-axis label,
            - logx (bool): if True set x-axis to logarithmic scale,
            - logy (bool): if True set y-axis to logarithmic scale,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "Line plot"
    if "xlabel" not in kwargs.keys():
        kwargs["xlabel"] = "x"
    if "ylabel" not in kwargs.keys():
        kwargs["ylabel"] = "y"
    if "logx" not in kwargs.keys():
        kwargs["logx"] = False
    if "logy" not in kwargs.keys():
        kwargs["logy"] = False
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    fig = plt.figure(figsize=(kwargs["figsize"][0], kwargs["figsize"][1]))
    ax = fig.add_subplot(111, aspect="equal")
    plt.sca(ax)
    plt.grid(False)

    if x_val is None:
        x_val = range(len(y_val))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(x_val, y_val, marker="o")

    plt.xticks(fontsize=kwargs["fontsize"] * 0.85)
    plt.yticks(fontsize=kwargs["fontsize"] * 0.85)

    if kwargs["logy"]:
        ax.set_yscale("log")

    if kwargs["logx"]:
        ax.set_xscale("log")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.xlabel(kwargs["xlabel"], fontsize=kwargs["fontsize"])
    plt.ylabel(kwargs["ylabel"], fontsize=kwargs["fontsize"])

    plt.title(kwargs["title"], size=kwargs["fontsize"] * 1.15)

    ax.set_aspect("auto")
    fig.tight_layout()

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax


def scatter_on_map(
    datagroups: Collection[np.ndarray],
    centers: Collection[np.ndarray],
    polygons: str,
    color_val: bool = None,
    show: bool = True,
    print_out: bool = False,
    file_name: str = "./som_scatter.png",
    **kwargs: Tuple[int]
) -> Tuple[Figure, plt.Axes]:
    """Scatter plot with points projected onto a 2D SOM.

    Args:
        datagroups (list[array,...]): Coordinates of the projected points.
            This must be a nested list/array of arrays, where each element of
            the list is a group that will be plotted separately.
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        color_val (array): The feature value to use as color map, if None
                the map will be plotted as white.
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "Projection onto SOM"
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    if color_val is None:
        color_val = np.full(len(centers), np.nan)

    fig, ax = plot_map(
        centers, color_val, polygons, show=False, print_out=False, **kwargs
    )

    for i, group in enumerate(datagroups):
        print(group[:, 0])
        ax.scatter(
            group[:, 0],
            group[:, 1],
            color=Distinct20()[i % 20],
            edgecolor="#ffffff",
            linewidth=1,
            label="{:d}".format(i),
        )

    plt.legend(
        bbox_to_anchor=(-0.025, 1),
        fontsize=kwargs["fontsize"] * 0.85,
        frameon=False,
        title="Groups",
        ncol=int(len(datagroups) / 10.0) + 1,
        title_fontsize=kwargs["fontsize"],
    )

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax
