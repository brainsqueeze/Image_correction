#  __author__ = 'Dave'
import numpy as np


def boundary_separation(array, idx):
    """
    Computes the distance between boundary points in the 4, partitioned
    closed sets on S1.

    :param array: list of each sub-array for the non-empty quadrants (tuple)
    :param idx: index of the current array (int)
    :return: (float)
    """
    l = array[idx]
    l_next = array[(idx + 1) % len(array)]

    if l_next[0] > l[-1]:
        return l_next[0] - l[-1]
    else:
        return l_next[0] - (l[-1] - (2 * np.pi))


def min_arc_length(array):
    """
    Computes the shortest path connecting all points in the input array
    constrained to the unit circle.  Equivalently, this is computing the
    minimum closed set containing all elements in the standard S1 topology.

    :param array: array of angles (list)
    :return: shortest path length (float)
    """
    pi = np.pi
    q1, q2, q3, q4 = [], [], [], []

    # split circle into 4 quadrants, populate each quadrant depending on incoming value
    # modulo on 2pi is handled to avoid winding numbers
    for angle in iter(array):
        mod_angle = angle % (2 * pi)

        if 0 <= mod_angle < pi / 2:
            q1.append(mod_angle)
        elif pi / 2 <= mod_angle < pi:
            q2.append(mod_angle)
        elif pi <= mod_angle < (3 * pi / 2):
            q3.append(mod_angle)
        else:
            q4.append(mod_angle)

    # sort each quadrant, separately, as ascending angle
    q1.sort()
    q2.sort()
    q3.sort()
    q4.sort()

    # get the maximum arc length between adjacent points, for each quadrant respectively
    max_length = max(len(q1), len(q2), len(q3), len(q4))
    l1, l2, l3, l4 = 0, 0, 0, 0
    for i in xrange(max_length - 1):
        if i + 1 < len(q1):
            d = q1[i + 1] - q1[i]
            l1 = d if d > l1 else l1
        if i + 1 < len(q2):
            d = q2[i + 1] - q2[i]
            l2 = d if d > l2 else l2
        if i + 1 < len(q3):
            d = q3[i + 1] - q3[i]
            l3 = d if d > l3 else l3
        if i + 1 < len(q4):
            d = q4[i + 1] - q4[i]
            l4 = d if d > l4 else l4

    l = max(l1, l2, l3, l4)

    # get the max distance between boundary points of the adjacent quadrants
    t = [arr for arr in (q1, q2, q3, q4) if arr]
    bd = [boundary_separation(t, i) for i in xrange(len(t))]
    l = max(bd + [l])  # max distance from either boundary points or the max within the quadrants

    return (2 * pi) - l


def translate(arr, n_permute):
    """
    :param arr: (list)
    :param n_permute: (int) order of the permutation
    :return: (list) permuted array
    """
    return [item for item in arr[n_permute:]] + [item for item in arr[:n_permute]]


def loss(m1, m2):
    """
    :param m1: slope of line 1 (float)
    :param m2: slope of line 2 (float)
    :return: (float)
    """
    if not type(m1) == np.ndarray or not type(m2) == np.ndarray:
        raise ValueError('TypeError: m1 and m2 must be numpy arrays.')
    return np.arctan((m1 / m2) - 1.)


def health(population, metric='L2', order=1):
    """
    :param population: loss functions for set of objects (numpy array)
    :param metric: metric space measure (string, optional, 'L2' by default, accepts: 'L1', 'minkowski',
    'vector_length', 'arc_length')
    :param order: (float, optional, only used with Minkowski p-adic measure, 1 by default)
    :return: metric space norm of population (float)
    """
    population = np.array(population)
    if metric == 'L2':
        return np.sqrt(np.dot(population, population))
    elif metric == 'L1':
        return np.sum(abs(population))
    elif metric == 'minkowski':
        t = np.sum(abs(population)**order)
        return t**(1. / order)
    elif metric == 'vector_length':
        v = [reduce(lambda x, y: np.cos(x) + np.cos(y), population),
             reduce(lambda x, y: np.sin(x) + np.sin(y), population)]
        v = np.array(v)
        return np.sqrt(np.dot(v, v)) / float(len(v))
    elif metric == 'arc_length':
        return min_arc_length(population)
