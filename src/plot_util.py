# Plotting code adapted from
#  https://github.com/michaelaerni/iclr23-InductiveBiasesHarmlessInterpolation/blob/main/src/plot_util.py

import math

import matplotlib
import matplotlib.colors
import matplotlib.container
import matplotlib.patches
import matplotlib.pyplot

TEX_PT_PER_IN = 72.27
DEFAULT_PPI = 300.0  # points per PIXEL! (display in e.g., jupyter)
PT_PER_BP = 803.0 / 800.0  # big point to point

GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5))
SILVER_RATIO = 1.0 + math.sqrt(2)

# Reference font sizes from LaTeX (all in pt)
_FONT_SIZE_MAIN_PT = 9.0
_FONT_SIZE_SMALL_PT = 8.0
_FONT_SIZE_FOOTNOTE_PT = 7.0
_FONT_SIZE_SCRIPT_PT = 6.0
_FONT_SIZE_SUBLARGE_PT = 10.0
_FONT_SIZE_LARGE_PT = 10.95

# Figure widths
FIGURE_WIDTH_ONECOL_PT = 230.0
FIGURE_WIDTH_TWOCOL_PT = 480.0
FIGURE_WIDTH_REDUCED_PT = 166.0  # three figures spanning full width
FIGURE_SIZE_ONECOL_IN = (
    FIGURE_WIDTH_ONECOL_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_ONECOL_PT / TEX_PT_PER_IN) / GOLDEN_RATIO,
)
FIGURE_SIZE_TWOCOL_IN = (
    FIGURE_WIDTH_TWOCOL_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_TWOCOL_PT / TEX_PT_PER_IN) / SILVER_RATIO,
)
FIGURE_SIZE_REDUCED_IN = (
    FIGURE_WIDTH_REDUCED_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_REDUCED_PT / TEX_PT_PER_IN) * 3.0 / 4.0,  # different ratio for small figures
)

LINE_WIDTH_PT = 1.25
BAR_WIDTH = 1.0 / GOLDEN_RATIO

DEFAULT_COLORMAP = matplotlib.colors.ListedColormap(
    colors=("#139FCD", "#FFD166", "#CE123E", "#052B38", "#06EFB1"), name="cvd_friendly"
)

MARKER_MAP = ("o", "d", "x", "+")

LINESTYLE_MAP = ("solid", "dashed", "dotted", (0, (3, 1, 1, 1, 1, 1)))  # dash dot dot

SHADING_ALPHA = 0.3


def setup_matplotlib():
    matplotlib.pyplot.rcdefaults()

    # Use colormap which works for people with CVD and greyscale printouts
    matplotlib.colormaps.register(cmap=DEFAULT_COLORMAP, force=True)

    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "image.cmap": DEFAULT_COLORMAP.name,
            "axes.prop_cycle": matplotlib.rcsetup.cycler("color", DEFAULT_COLORMAP.colors),
            "font.family": "sans-serif",
            "font.sans-serif": ["Open Sans"],
            "font.size": _FONT_SIZE_MAIN_PT,
            "figure.dpi": DEFAULT_PPI,
            # Axis labels, titles (if any) and legend labels are one smaller than main text
            "axes.titlesize": _FONT_SIZE_SMALL_PT,
            "axes.labelsize": _FONT_SIZE_SMALL_PT,
            "legend.fontsize": _FONT_SIZE_SMALL_PT,
            # Ticks are two smaller than main text
            "xtick.labelsize": _FONT_SIZE_FOOTNOTE_PT,
            "ytick.labelsize": _FONT_SIZE_FOOTNOTE_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "patch.linewidth": LINE_WIDTH_PT,
            "lines.markersize": 5,
            # "scatter.edgecolors": "black",
            # "errorbar.capsize": 2,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderpad": 0.1,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.2,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.5,
            "legend.loc": "center",
            "savefig.dpi": "figure",
            "savefig.pad_inches": 0.0,
            "savefig.transparent": True,
            "figure.constrained_layout.use": True,
            "figure.figsize": FIGURE_SIZE_ONECOL_IN,
            # "axes.grid": True,
            # "axes.grid.which": "major",
            "grid.color": "#c0c0c0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.25,
            "grid.alpha": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            # Disable minor ticks by default for increase clarity
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        }
    )


def resize_figure_with_legend(fig: matplotlib.figure.Figure) -> None:
    (legend,) = filter(lambda child: isinstance(child, matplotlib.legend.Legend), fig.get_children())
    # FIXME: Calculation might not be completely correct, but correct enough
    fig.set_size_inches(
        fig.get_size_inches()[0],
        fig.get_size_inches()[1] + legend.get_window_extent().height / fig.dpi,
    )


def extend_line_y_clipping(ax: matplotlib.axes.Axes, extension_factor: float = 2.0) -> None:
    for line in ax.get_lines():
        line.set_clip_box(
            matplotlib.transforms.TransformedBbox(
                matplotlib.transforms.Bbox([[0, 0], [extension_factor, extension_factor]]),
                ax.transAxes,
            )
        )


def proxy_patch() -> matplotlib.patches.Patch:
    return matplotlib.patches.Patch(color="none")
