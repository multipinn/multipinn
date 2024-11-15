from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def update_fig_2d(fig: go.Figure) -> go.Figure:
    """Updates 2D figure layout with consistent styling.

    Applies standard styling including font sizes, grid lines, axis lines and labels.

    Args:
        fig (go.Figure): Plotly figure to update

    Returns:
        go.Figure: Updated figure with consistent styling
    """
    text_size = 45
    h_size = 1_000
    w_size = 1_000

    fig.update_layout(
        autosize=False,
        width=w_size,
        height=h_size,
        font=dict(family="Times new roman", size=text_size),
        margin=dict(l=0, r=0, b=0, t=20, pad=0),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=7,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=5,
        gridcolor="LightPink",
        zeroline=True,
        zerolinewidth=5,
        zerolinecolor="LightPink",
        title_text="<i>x<i>",
        title_font_size=text_size,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=7,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=5,
        gridcolor="LightPink",
        zeroline=True,
        zerolinewidth=5,
        zerolinecolor="LightPink",
        title_text=f"<i>t<i>",
        title_standoff=50,
        title_font_size=text_size,
    )

    return fig


def plot_2d_heatmap(x: np.ndarray, y: np.ndarray, z: np.ndarray, style: dict):
    """Creates a 2D heatmap visualization.

    Args:
        x (np.ndarray): X-axis coordinates
        y (np.ndarray): Y-axis coordinates
        z (np.ndarray): Values to plot as colors
        style (dict): Additional plotly styling options to override defaults

    Returns:
        go.Figure: Plotly figure containing the heatmap
    """
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
        )
    )
    default_style = {
        "data_0_colorscale": "Thermal",
        "data_0_showlegend": False,
        "data_0_colorbar_thickness": 60,
        "layout_autosize": False,
        "layout_width": 1200,
        "layout_height": 800,
        "layout_font": dict(family="Times new roman", size=45),
        "layout_xaxis": go.layout.XAxis(
            title="<i>x<i>",
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            color="black",
        ),
        "layout_yaxis": go.layout.YAxis(
            title="<i>y<i>",
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            color="black",
        ),
    }
    fig.update(**default_style)
    fig.update(**style)
    return fig


def scatter_2d(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    min: int,
    max: int,
    name: str,
) -> go.Scatter:
    """Creates a 2D scatter plot with points colored by data values.

    Args:
        x (np.ndarray): X-axis coordinates
        y (np.ndarray): Y-axis coordinates
        data (np.ndarray): Values used to color the points
        min (int): Minimum value for the color scale
        max (int): Maximum value for the color scale
        name (str): Label for the colorbar

    Returns:
        go.Scatter: Plotly scatter trace object
    """
    return go.Scatter(
        x=x,
        y=y,
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


def plot_2d_scatter(x: np.ndarray, y: np.ndarray, style):
    """Creates a simple 2D scatter plot.

    Args:
        x (np.ndarray): X-axis coordinates
        y (np.ndarray): Y-axis coordinates
        style (dict): Additional plotly styling options to override defaults

    Returns:
        go.Figure: Plotly figure containing the scatter plot
    """
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
        )
    )
    default_style = {
        "data_0_marker": dict(size=4, color="black"),
        "data_0_showlegend": False,
        "layout_autosize": True,
        "layout_xaxis": dict(scaleanchor="y", scaleratio=1),
        "layout_yaxis": dict(scaleanchor="x", scaleratio=1),
        # "layout_font": dict(family="Times new roman", size=45),
    }
    fig.update(**default_style)
    fig.update(**style)
    return fig


def update_2d_figures_axes(
    fig: go.Figure, x_min: float, y_min: float, x_max: float, y_max: float
) -> go.Figure:
    """Updates axes ranges and applies consistent styling.

    Args:
        fig (go.Figure): Plotly figure to update
        x_min (float): Minimum x-axis value
        y_min (float): Minimum y-axis value
        x_max (float): Maximum x-axis value
        y_max (float): Maximum y-axis value

    Returns:
        go.Figure: Updated figure with new axis ranges and styling
    """
    fig = go.Figure(data=fig)
    fig = update_fig_2d(fig)

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig
