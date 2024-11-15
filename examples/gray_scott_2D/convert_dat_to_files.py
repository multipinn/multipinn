import numpy as np

if __name__ == "__main__":
    data = np.genfromtxt("grayscott.dat", comments="%", dtype="float32")
    points = data[:, :3]
    values = data[:, 3:]

    np.savetxt("points.dat", points, delimiter=",", fmt="%.18e")
    np.savetxt("values.dat", values, delimiter=",", fmt="%.18e")
