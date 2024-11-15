import itertools
from typing import List, Sequence, Union

import numpy as np
import plotly.graph_objects as go


def SliderPlot(keys: Sequence[Union[str, int]], data, coord: List = None):
    assert len(keys) == len(data.shape)
    assert "x" in keys
    assert "y" in keys
    if coord is None:
        coord = []
        for s in data.shape:
            coord.append(np.arange(s))
    else:
        assert len(keys) == len(coord)
        for i, e in enumerate(coord):
            if e is None:
                coord[i] = np.arange(data.shape[i])
    sliders = []
    for i, key in enumerate(keys):
        if isinstance(key, str):
            if key == "x":
                assert "x_dim" not in locals()
                x_dim = i
            elif key == "y":
                assert "y_dim" not in locals()
                y_dim = i
            else:
                sliders.append(i)
        else:
            assert 0 <= key < data.shape[i]
    assert sliders

    fig = go.Figure()
    titles = []
    names = []
    for index in itertools.product(*[range(data.shape[s]) for s in sliders]):
        slices = []
        k = 0
        for key in keys:
            if isinstance(key, str):
                if key == "x" or key == "y":
                    slices.append(slice(None))
                else:
                    slices.append(index[k])
                    k += 1
            else:
                slices.append(key)
        z = data[tuple(slices)]
        if x_dim < y_dim:
            z = z.T
        fig.add_trace(
            go.Heatmap(
                visible=False,
                x=coord[x_dim],
                y=coord[y_dim],
                z=z,
                showlegend=False,
                colorscale="Jet",
            )
        )
        title = ""
        name = ""
        for i in range(len(index)):
            title += f"{keys[sliders[i]]}: {coord[sliders[i]][index[i]]} "
            name += f"{index[i]} "
        titles.append(title[:-1])
        names.append(name[:-1])

    steps = []
    n = len(fig.data)
    for i in range(n):
        step = dict(
            method="update",
            args=[{"visible": [False] * n}, {"title": titles[i]}],
            label=names[i],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    fig.data[0].visible = True
    fig.update_layout(sliders=[dict(active=0, steps=steps)], title=titles[0])
    fig.show()


if __name__ == "__main__":
    data = np.arange(3 * 10 * 4 * 20 * 2).reshape((3, 10, 4, 20, 2))
    coord = [
        None,
        np.linspace(-10, 10, 10),
        np.linspace(0, 0.6, 4),
        np.linspace(5, 6, 20),
        None,
    ]

    SliderPlot(("slider_text", "y", "other_value", "x", 1), data, coord)
    SliderPlot((2, "x", "y", "X value", 0), data, coord)
