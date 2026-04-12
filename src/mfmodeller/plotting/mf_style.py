"""
A template to stylise plotly, such that it fits certain scientific conventions better.
This file needs to be within the path, and you import it at the beginning of
the document and then set the default template to the one desired as:
`import publishable
pio.templates.default = "scientific"`
There is a helper function that does this for you in this script.

"""

import plotly.graph_objects as go
import plotly.io as pio

FIG_WIDTH = 612.40416  # Matches text width of my personal document exactly.
FIG_HEIGHT = 0.75 * FIG_WIDTH  # Sets it to a 4/3 aspect ratio.

MARKER_SIZE = 8
MARKER_LINE_WIDTH = 1.5
LINE_WIDTH = 2

african_violet = "rgb(178,132,190)"  # A colour I frequently want

STANDARD_COLOURWAY = [
    "rgb(0,0,0)",
    "rgb(0,119,187)",
    "rgb(238,51,119)",
    "rgb(0,153,136)",
    "rgb(51,187,238)",
    "rgb(238,119,51)",
    "rgb(204,51,17)",
    "rgb(187,187,187)",
]

STANDARD_COLOURWAY_TRANSPARENTS = [
    "rgb(0,0,0)",
    "rgba(47,27,93,1.0)",
    "rgba(47,27,93,0.725)",
    "rgba(47,27,93,0.60)",
    "rgba(47,27,93,0.475)",
    "rgba(47,27,93,0.35)",
    "rgba(47,27,93,0.1)",
]
margin_dict = {"t": 7.5, "b": 22, "l": 45, "r": 7.5}

pio.templates["scientific"] = go.layout.Template(
    layout={
        "colorway": STANDARD_COLOURWAY,
        "scene": {
            "xaxis": {
                "backgroundcolor": "white",
                "gridwidth": 2,
                "linecolor": "rgb(0,0,0)",
                "showbackground": True,
                "showgrid": True,
                "showline": True,
                "mirror": "allticks",
                "ticks": "inside",
                "nticks": 6,
                "tickwidth": 1.2,
                "zeroline": False,
                "zerolinecolor": "rgb(0,0,0)",
            },
            "yaxis": {
                "backgroundcolor": "white",
                "gridwidth": 2,
                "linecolor": "rgb(0,0,0)",
                "showbackground": True,
                "showgrid": True,
                "showline": True,
                "mirror": "allticks",
                "ticks": "inside",
                "nticks": 6,
                "tickwidth": 1.2,
                "zeroline": False,
                "zerolinecolor": "rgb(0,0,0)",
            },
            "zaxis": {
                "backgroundcolor": "white",
                "gridwidth": 2,
                "linecolor": "rgb(0,0,0)",
                "showbackground": True,
                "showgrid": True,
                "showline": True,
                "mirror": "allticks",
                "ticks": "inside",
                "nticks": 6,
                "tickwidth": 1.2,
                "zeroline": False,
                "zerolinecolor": "rgb(0,0,0)",
            },
        },
        "xaxis": {
            "automargin": True,
            "linecolor": "rgb(0,0,0)",
            "linewidth": 2,
            "showgrid": True,
            "showline": True,
            "mirror": "allticks",
            "ticks": "inside",
            "nticks": 6,
            "tickwidth": 1.2,
            "title": {"standoff": 10},
            "zeroline": False,
            "griddash": "dot",
            "gridwidth": 0.5,
            "gridcolor": "rgba(0,0,0,0.5)",
        },
        "yaxis": {
            "automargin": True,
            "linecolor": "rgb(0,0,0)",
            "linewidth": 2,
            "showgrid": True,
            "showline": True,
            "mirror": "allticks",
            "ticks": "inside",
            "nticks": 6,
            "tickwidth": 1.2,
            "title": {"standoff": 10},
            "griddash": "dot",
            "gridwidth": 0.5,
            "gridcolor": "rgba(0,0,0,0.5)",
            "zeroline": False,
        },
        "font": {"family": "CMU Serif", "size": 11, "color": "black"},
        "width": FIG_WIDTH,  # Matches text width of latex document exactly.
        "height": FIG_HEIGHT,  # Sets it to a 4/3 aspect ratio.
        "margin": margin_dict,
        "legend": {
            "orientation": "h",
            "y": -0.3,
            "x": 0.5,
            "yanchor": "bottom",
            "xanchor": "center",
            "bordercolor": "black",
            "borderwidth": 1.4,
        },
    }
)


def intialise_plotly_style() -> None:
    import plotly.io as pio

    pio.templates.default = "scientific"
    return
