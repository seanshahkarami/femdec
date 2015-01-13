import numpy as np
import numpy.linalg as la
from itertools import combinations
from itertools import combinations_with_replacement
from pydec import simplicial_complex
from pydec.fem import whitney_innerproduct
from scipy.misc import factorial
from scipy.sparse import csr_matrix


def barycentric_integral(powers, volumes):
    """
    Integral on standard n-simplex of barycentric monomials:

         a0       a1          an
    lambda   lambda  ... lambda

    """
    n = powers.shape[-1] - 1
    integrals_on_ref = (np.prod(factorial(powers), axis=-1) /
                        factorial(np.sum(powers, axis=-1) + n))
    return integrals_on_ref * factorial(n) * volumes[:, None, None]


class whitney_element(object):

    def __init__(self, coeffs, powers, wedges):
        self.coeffs = coeffs
        self.powers = powers
        self.wedges = wedges


class whitney_elements(object):

    def __init__(self, indices, coeffs, powers, wedges, ordering):
        self.indices = indices
        self.coeffs = coeffs
        self.powers = powers
        self.wedges = wedges
        self.ordering = ordering

    def d(self):
        # d_indices = []
        d_coeffs = []
        d_powers = []
        # d_wedges = []

        for coeff, power, wedge in zip(self.coeffs, self.powers, self.wedges):
            for i in xrange(self.powers.shape[-1]):
                if power[i] > 0 and i not in wedge:
                    d_coeff = coeff * power[i]
                    d_power = power.copy()
                    d_power[i] -= 1
                    # d_wedge = np.empty(wedge)

                    # d_wedges[count, 0] = i
                    # d_wedges[count, 1:] = wedge
                    d_coeffs.append(d_coeff)
                    d_powers.append(d_power)


def whitney_basis(sc, k, r):
    n = sc.complex_dimension()
    signatures = afk_signatures(n, k, r)

    elements = [element_from_signature(n, k, sig) for sig in signatures]

    coeffs = np.hstack([e.coeffs for e in elements])
    powers = np.vstack([e.powers for e in elements])
    wedges = np.vstack([e.wedges for e in elements])

    return whitney_elements(sc, k, [], coeffs, powers, wedges)


class whitney_signature(object):

    def __init__(self, alpha, sigma):
        self.alpha = np.array(alpha, dtype=np.int)
        self.sigma = np.array(sigma, dtype=np.int)

    def key_on_simplex(self, simplex):
        return simplex[self.alpha].tostring() + \
            simplex[self.sigma].tostring()


def afk_signatures(n, k, r):
    reference_simplex = np.arange(n+1)
    return [whitney_signature(alpha, sigma)
            for sigma in combinations(reference_simplex, k+1)
            for alpha in combinations_with_replacement(sigma, r-1)]


def element_from_signature(n, k, sig):
    coeffs = np.empty(k+1, dtype=np.float)
    powers = np.empty((k+1, n+1), dtype=np.int)
    wedges = np.empty((k+1, k), dtype=np.int)

    coeffs[::2] = 1.0
    coeffs[1::2] = -1.0

    powers[:] = np.bincount(sig.alpha, minlength=n+1)

    for i in xrange(k+1):
        powers[i, sig.sigma[i]] += 1
        wedges[i, :i] = sig.sigma[:i]
        wedges[i, i:] = sig.sigma[i+1:]

    return whitney_element(coeffs, powers, wedges)


# class geometric_data(object):

#     def __init__(self, sc):
#         points = sc.vertices[sc.simplices]
#         self.frames = frames


import time


def barycentric_diffs(n):
    return np.column_stack([-np.ones(n), np.eye(n)])


def grads_from_diffs(metric, diffs):
    return la.solve(metric, diffs)


def cartesian_product(*args):
    return map(np.ravel, np.meshgrid(*args))


def inner(sc, u, v):
    points = sc.vertices[sc.simplices]
    frames = points[:, 1:] - points[:, 0, None]

    metrics = np.array([np.inner(frame, frame) for frame in frames])
    dual_metrics = np.array(map(la.inv, metrics))

    volumes = np.sqrt(la.det(metrics)) / factorial(sc.complex_dimension())
    # volumes[sc[-1].simplex_parity == 0] *= -1.0

    print volumes

    coeffs = u.coeffs[:, None] * v.coeffs
    powers = u.powers[:, None] + v.powers
    integrals = barycentric_integral(powers, volumes)

    # integrals_upper = ...
    # integrals_lower = ... (all the same power...so same result.)

    diffs = barycentric_diffs(sc.complex_dimension())

    inners = np.array([np.dot(diffs.T, np.dot(dual_metric, diffs))
                       for dual_metric in dual_metrics])

    dets = np.empty((len(sc[-1].simplices), len(u.wedges), len(v.wedges)))

    for i1, w1 in enumerate(u.wedges):
        minors = inners[:, w1]
        for i2, w2 in enumerate(v.wedges):
            dets[:, i1, i2] = la.det(minors[:, :, w2])

    indices1, indices2 = np.meshgrid(u.indices, v.indices)
    rows = u.ordering[:, indices1.ravel()].ravel()
    cols = v.ordering[:, indices2.ravel()].ravel()
    data = (coeffs * integrals * dets).ravel()

    return csr_matrix((data, (rows, cols)))


def test_orientation():
    vertices = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],

        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    simplices = np.array([
        [0, 1, 2],
        [3, 5, 4],
    ])

    sc = simplicial_complex(vertices, simplices)

    indices = np.array([0, 0, 1, 1, 2, 2])

    coeffs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    powers = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])

    wedges = np.array([
        [1],
        [0],
        [2],
        [0],
        [2],
        [1],
    ])

    ordering = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])

    u = whitney_elements(indices, coeffs, powers, wedges, ordering)
    K = inner(sc, u, u).todense()
    assert np.allclose(K[:3, :3], -K[3:, 3:])


def test_square():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    simplices = np.array([
        [0, 1, 2],
        [1, 2, 3],
    ])

    sc = simplicial_complex(vertices, simplices)

    indices = np.array([0, 0, 1, 1, 2, 2])

    coeffs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    powers = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])

    wedges = np.array([
        [1],
        [0],
        [2],
        [0],
        [2],
        [1],
    ])

    sigmas = np.array([
        [0, 1],
        [0, 2],
        [1, 2],
    ])

    lookup = {}
    ordering = []

    for simplex in simplices:
        simplex_ordering = []
        for sigma in sigmas:
            key = simplex[sigma].tostring()
            if key not in lookup:
                lookup[key] = len(lookup)
            simplex_ordering.append(lookup[key])
        ordering.append(simplex_ordering)

    ordering = np.array(ordering)

    # consider factoring out a fem data {} object which
    # we can build into an extendable framework.

    start = time.time()
    u = whitney_elements(indices, coeffs, powers, wedges, ordering)
    K1 = inner(sc, u, u).todense()
    print time.time() - start

    start = time.time()
    K2 = whitney_innerproduct(sc, k=1).todense()
    print time.time() - start


def main():
    # test_orientation()
    # test_square()

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    simplices = np.array([
        [0, 1, 2],
        [1, 2, 3],
    ])

    sc = simplicial_complex(vertices, simplices)

    indices = np.array([0, 0, 1, 1, 2, 2])

    coeffs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    powers = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])

    wedges = np.array([
        [1],
        [0],
        [2],
        [0],
        [2],
        [1],
    ])

    sigmas = np.array([
        [0, 1],
        [0, 2],
        [1, 2],
    ])

    lookup = {}
    ordering = []

    for simplex in simplices:
        simplex_ordering = []
        for sigma in sigmas:
            key = simplex[sigma].tostring()
            if key not in lookup:
                lookup[key] = len(lookup)
            simplex_ordering.append(lookup[key])
        ordering.append(simplex_ordering)

    ordering = np.array(ordering)

    # consider factoring out a fem data {} object which
    # we can build into an extendable framework.

    start = time.time()
    u = whitney_elements(indices, coeffs, powers, wedges, ordering)
    K1 = inner(sc, u, u).todense()
    print time.time() - start

    start = time.time()
    K2 = whitney_innerproduct(sc, k=1).todense()
    print time.time() - start


    # from modepy import XiaoGimbutasSimplexQuadrature

    # quadrature = XiaoGimbutasSimplexQuadrature(order=5, dims=2)

    # nodes = (quadrature.nodes + 1.0) / 2.0
    # nodes = np.vstack([-np.sum(nodes, axis=0), nodes])
    # weights = quadrature.weights / 2.0
    # l0, l1, l2 = nodes

    # C = np.array([1.0, -1.0])

    # P = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    # ])

    # CC = C[:, None] * C
    # PP = P[:, None] + P

    # print np.sum(np.prod(nodes ** PP[:, :, None], axis=1)*CC[:, None]*weights, axis=0)

    # print np.sum((l1*l1*weights +
    #        -l1*l2*weights +
    #        -l1*l2*weights +
    #        l2*l2*weights))


if __name__ == '__main__':
    main()
