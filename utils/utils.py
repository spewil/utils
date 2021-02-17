import numpy as np


def normal(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) /
                  (2 * np.power(sig, 2.))) * (1 / (sig * np.sqrt(2 * np.pi)))


def mahalanobis(vec, inv_cov):
    prod = vec.reshape(-1, 1).T @ inv_cov @ vec.reshape(-1, 1)
    return np.sqrt(prod)[0, 0]


def polar_to_cartesian(coord):
    return [coord[0] * np.cos(coord[1]), coord[0] * np.sin(coord[1])]


def cartesian_to_polar(coord):
    output = [0, 0]
    output[0] = np.sqrt(coord[0]**2 + coord[1]**2)
    # change behavior depending on quadrant
    output[1] = np.arccos(coord[0] / output[0])
    if coord[1] >= 0:
        output[1] = np.arccos(coord[0] / output[0])
    else:
        if coord[0] >= 0:
            output[1] = -np.arccos(coord[0] / output[0])
        else:
            output[1] = np.pi + np.arccos(-coord[0] / output[0])
    return output


def radians(degrees):
    return (np.pi / 180) * degrees


def quadratic(x, a, b, start=-250):
    return a * (x - start)**2 + b


def generate_arc(num_points, radius=250, start=0, end=180, polar=False):
    start_angle_rad = (np.pi / 180) * start
    end_angle_rad = (np.pi / 180) * end
    angles = np.linspace(end_angle_rad,
                         start_angle_rad,
                         num_points,
                         endpoint=True)
    radii = np.ones(num_points) * radius
    polar_coords = np.array(list(zip(radii, angles)))
    if polar:
        return polar_coords
    else:
        return np.array([polar_to_cartesian(p) for p in polar_coords])


def unique(data: list):
    ul = []
    for x in data:
        if x not in ul:
            ul.append(x)
    return ul


# invert trajectories if it increases their correlation coeff
def align_trajectory(trajectory, reference):
    corr = np.dot(trajectory, reference)
    corr_flipped = np.dot(-trajectory, reference)
    if corr > corr_flipped:
        return trajectory
    else:
        return -trajectory


def random_matrices():
    # A problem with a low condition number is said to be well-conditioned, while a problem with a high condition number is said to be ill-conditioned.

    size = 5
    conds = []
    for _ in range(1000):
        A = np.random.normal(size=(size, size))
        # print(A)
        # print(np.linalg.norm(A, axis=0))
        A_normed = A / np.linalg.norm(A, axis=0)
        # print(A_normed)
        # print(np.sqrt(np.sum(A_normed[:, 1]**2)))
        conds.append(np.linalg.cond(A_normed))

    print(np.min(conds), np.max(conds), np.median(conds), np.mean(conds))
    # plt.hist(conds, bins=100)
    # plt.show()
