from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    data: np.array,
    min: int,
    max: int,
    name: str,
) -> go.Scatter3d:
    """Creates a 3D scatter plot with points colored by data values.

    Args:
        x (np.ndarray): Array of points along the x-axis
        y (np.ndarray): Array of points along the y-axis
        z (np.ndarray): Array of points along the z-axis
        data (np.array): Values used to color the points
        min (int): Minimum value for the color scale
        max (int): Maximum value for the color scale
        name (str): Label for the colorbar

    Returns:
        go.Scatter3d: Plotly 3D scatter trace object
    """
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=8,
            color=data,
            colorscale="Jet",
            colorbar=dict(title=name),
            cmin=min,
            cmax=max,
        ),
    )


def plot_3d_scatter(x: np.ndarray, y: np.ndarray, z: np.ndarray, style):
    """Creates a simple 3D scatter plot with consistent styling.

    Args:
        x (np.ndarray): X-axis coordinates
        y (np.ndarray): Y-axis coordinates
        z (np.ndarray): Z-axis coordinates
        style (dict): Additional plotly styling options to override defaults

    Returns:
        go.Figure: Plotly figure containing the 3D scatter plot
    """
    fig = go.Figure(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
        )
    )
    default_style = {
        "data_0_marker": dict(size=4, color="black"),
        "data_0_showlegend": False,
        "layout_autosize": True,
        "layout_scene_aspectmode": "data",
        # "layout_font": dict(family="Times new roman", size=45),
    }
    fig.update(**default_style)
    fig.update(**style)
    return fig


def slider_surface_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    volume: np.array,
    min: float,
    max: float,
    name: str,
):
    """Creates an animated 3D surface plot with a slider control.

    Generates a series of 2D surfaces at different z-levels that can be animated
    using a slider control. Each surface is colored based on the values in volume.

    Args:
        x (np.ndarray): Array of points along the x-axis
        y (np.ndarray): Array of points along the y-axis
        z (np.ndarray): Array of z-levels for each surface
        volume (np.array): 3D array of values used to color each surface
        min (float): Minimum value for the color scale
        max (float): Maximum value for the color scale
        name (str): Label for the colorbar

    Returns:
        go.Figure: Plotly figure containing the animated 3D surface plot with slider controls
    """
    r, c = volume[0].shape

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=z[k] * np.ones((r, c)),
                    surfacecolor=volume[k],
                    cmin=min,
                    cmax=max,
                    x=x,
                    y=y,
                ),
                name=f"{k}",
            )
            for k in range(len(z))
        ]
    )

    fig.add_trace(
        go.Surface(
            z=z[0] * np.ones((r, c)),
            surfacecolor=volume[0],
            colorscale="Jet",
            cmin=min,
            cmax=max,
            colorbar=dict(thickness=20, ticklen=4, title=name),
            x=x,
            y=y,
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 0, "t": 0},
            "len": 1,
            "x": 0,
            "y": 0,
            "active": len(z) - 1,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f"{z[k]:.4}",
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    z_eps = z[1] - z[0]
    fig.update_layout(
        scene=dict(
            zaxis=dict(range=[z[0] - z_eps, z[-1] + z_eps], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(3000 / len(z))],
                        "label": "&#9654;",
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "type": "buttons",
                "x": 1,
                "y": 0.01,
            }
        ],
        sliders=sliders,
    )

    return fig
