import numpy as np
import numpy.linalg as la
from scipy.misc import factorial
from scipy.sparse import csr_matrix
from modepy import XiaoGimbutasSimplexQuadrature


# TODO: Frames computed incorrectly. I don't think this affects
# the inner product computation since we only use the frame to
# compute the inner product of our elements.


def whitney_innerproduct(sc, u, v):
    # build geometric data
    metrics = compute_metrics(sc.vertices, sc[-1].simplices)
    volumes = compute_volume_from_metric(metrics)
    wedge_inners = compute_wedge_inners(u.wedges, v.wedges, metrics)

    # build matrix entry data
    coeffs = compute_pairwise_coeffs(u, v)
    powers = compute_pairwise_powers(u, v)
    integrals = barycentric_integrals(powers, volumes)
    data = (coeffs * integrals * wedge_inners).ravel()

    # build matrix row / col data
    indices1, indices2 = np.meshgrid(u.indices, v.indices)
    rows = u.ordering[:, indices1.ravel()].ravel()
    cols = v.ordering[:, indices2.ravel()].ravel()

    return csr_matrix((data, (rows, cols)))


# TODO: Leave quadrature method open. This shouldn't
# be forced on the user nor should we control it.
def projection(sc, u, f):
    dimension = sc.complex_dimension()
    metrics = compute_metrics(sc.vertices, sc[-1].simplices)
    volumes = compute_volume_from_metric(metrics)

    nodes, weights = quadrature_nodes_and_weights(order=6, dimension=2)

    num_simplices = sc[-1].simplices.shape[0]
    num_nodes = nodes.shape[1]

    frames = compute_frames(sc.vertices, sc[-1].simplices)

    half_metrics = np.array([np.dot(metric, frame.T)
                             for metric, frame in zip(metrics, frames)])

    nodes_on_simplices = np.hstack(np.dot(frames, nodes))

    barycentric_coordinates = barycentric_coordinates_from_nodes(nodes)

    monomials = np.prod(barycentric_coordinates.T ** u.powers[:, None, :], axis=-1)

    f_wedges = np.array([wedges for (_, wedges) in f])
    wedge_inners = compute_wedge_inners(u.wedges, f_wedges, metrics)

    f_values = np.empty((num_simplices, len(f), num_nodes))

    for i, (function, f_wedges) in enumerate(f):
        f_values[:, i, :] = function(*nodes_on_simplices).reshape(num_simplices,
                                                                  num_nodes)

    values = (monomials[None, :, None] * f_values[:, None, :]) * weights

    print np.sum(values, axis=-1) * u.coeffs[:, None]


def quadrature_nodes_and_weights(order, dimension):
    quadrature = XiaoGimbutasSimplexQuadrature(order=order, dims=dimension)
    nodes = (quadrature.nodes + 1.0) / 2.0
    weights = quadrature.weights / 2.0
    return nodes, weights


def compute_metrics(vertices, simplices):
    frames = compute_frames(vertices, simplices)
    return np.array(map(pullback_metric, frames))


def compute_dual_metrics(metrics):
    return np.array(map(la.inv, metrics))


def pullback_metric(frame):
    return np.dot(frame.T, frame)


def compute_frames(vertices, simplices):
    points = vertices[simplices]
    return np.swapaxes(points[:, 1:] - points[:, 0, None], 1, 2)
    # points = vertices[simplices]
    # return points[:, 1:] - points[:, 0, None]


def compute_volume_from_metric(metric):
    dimension = metric.shape[-1]
    return np.sqrt(la.det(metric)) / factorial(dimension)


def compute_wedge_inners(wedges1, wedges2, metrics):
    dimension = metrics.shape[-1]
    dual_metrics = compute_dual_metrics(metrics)
    diffs = barycentric_differentials(dimension)

    barycentric_inners = np.array([np.dot(diffs.T, np.dot(dual_metric, diffs))
                                   for dual_metric in dual_metrics])

    wedge_inners = np.empty((len(metrics), len(wedges1), len(wedges2)))

    for i1, w1 in enumerate(wedges1):
        minors = barycentric_inners[:, w1]
        for i2, w2 in enumerate(wedges2):
            wedge_inners[:, i1, i2] = la.det(minors[:, :, w2])

    return wedge_inners


def barycentric_differentials(n):
    """
    Provides us with columns representing the barycentric
    differentials This uses the relation

    dlambda0 = (-dlambda1) + ... + (-dlambdan)

    and takes the remaining dlambdas to be basis elements.
    """
    return np.column_stack([-np.ones(n), np.eye(n)])


def barycentric_integrals(powers, volumes):
    """
    Integral on standard n-simplex of barycentric monomials
    using the formula that the integral

          a0       a1           an
    lambda0  lambda1  ... lambdan

    on the standard simplex is

        a0! a1! ... an!
    -----------------------
    (a0 + a1 + ... an + n)!

    """
    n = powers.shape[-1] - 1
    integrals_on_ref = (np.prod(factorial(powers), axis=-1) /
                        factorial(np.sum(powers, axis=-1) + n))
    return integrals_on_ref * factorial(n) * volumes[:, None, None]


def compute_pairwise_coeffs(u, v):
    return u.coeffs[:, None] * v.coeffs


def compute_pairwise_powers(u, v):
    return u.powers[:, None] * v.powers


def barycentric_coordinates_from_nodes(nodes):
    return np.vstack([1.0 - np.sum(nodes, axis=0),
                      nodes])


def barycentric_monomial(powers, nodes):
    return np.prod(nodes ** powers[:, None, :], axis=-1)
