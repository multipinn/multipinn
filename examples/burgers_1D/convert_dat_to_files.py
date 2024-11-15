import numpy as np

if __name__ == "__main__":
    data = np.genfromtxt("burgers1d.dat", comments="%", dtype="float32")
    point_x = data[:, :1]
    values = data[:, 1:]
    points = []
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        point_t = np.ones((point_x.shape[0], 1)) * t
        points.append(np.column_stack((point_x, point_t)))

    np.savetxt("points.dat", np.vstack(points), delimiter=",", fmt="%.18e")
    np.savetxt(
        "values.dat",
        np.hstack(values.T).reshape(-1, 1),
        newline=",\n",
        delimiter=",",
        fmt="%.18e",
    )
