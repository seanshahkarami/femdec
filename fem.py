import numpy as np
import numpy.linalg as la
from itertools import combinations
from itertools import combinations_with_replacement
from pydec import simplicial_complex
from pydec.fem import whitney_innerproduct
from scipy.misc import factorial
from scipy.sparse import csr_matrix
# import time


# TODO: This should probably be refactored to NOT use volumes! (Jan 15 2015)
def barycentric_beta(powers, volumes):
    """
    Integral on standard n-simplex of barycentric monomials:

         a0       a1          an
    lambda   lambda  ... lambda

    The naming of this function `beta` comes from the fact that
    the standard beta function agrees with this in one-dimension,
    so you may argue this is a generalization to simplices.
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


def barycentric_differentials(n):
    return np.column_stack([-np.ones(n), np.eye(n)])


def grads_from_diffs(metric, diffs):
    return la.solve(metric, diffs)


def pullback_metric(frame):
    return np.inner(frame, frame)


def inner(sc, u, v):
    vertices = sc.vertices
    simplices = sc[-1].simplices

    points = vertices[simplices]
    frames = points[:, 1:] - points[:, 0, None]

    metrics = np.array(map(pullback_metric, frames))
    dual_metrics = np.array(map(la.inv, metrics))

    volumes = np.sqrt(la.det(metrics)) / factorial(sc.complex_dimension())
    # volumes[sc[-1].simplex_parity == 1] *= -1.0

    coeffs = u.coeffs[:, None] * v.coeffs
    powers = u.powers[:, None] + v.powers
    integrals = barycentric_beta(powers, volumes)

    diffs = barycentric_differentials(sc.complex_dimension())

    inners = np.array([np.dot(diffs.T, np.dot(dual_metric, diffs))
                       for dual_metric in dual_metrics])

    dets = np.empty((len(simplices), len(u.wedges), len(v.wedges)))

    # this next chuck vectorizes along simplices so
    # we only end up doing a small number of loops
    # through pairs of elements
    for i1, w1 in enumerate(u.wedges):
        minors = inners[:, w1]
        for i2, w2 in enumerate(v.wedges):
            dets[:, i1, i2] = la.det(minors[:, :, w2])

    indices1, indices2 = np.meshgrid(u.indices, v.indices)
    rows = u.ordering[:, indices1.ravel()].ravel()
    cols = v.ordering[:, indices2.ravel()].ravel()
    data = (coeffs * integrals * dets).ravel()

    return csr_matrix((data, (rows, cols)))


def unique_rows(a):
    a = np.ascontiguousarray(a)
    b = a.view(np.dtype((np.void, a.itemsize * a.shape[-1])))
    return np.unique(b, return_inverse=True)


def main():
    vertices = np.loadtxt('square.node', skiprows=1, usecols=(1, 2), dtype=np.float)
    simplices = np.loadtxt('square.ele', skiprows=1, usecols=(1, 2, 3), dtype=np.int)

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

    alphas = np.array([
        [],
        [],
        [],
    ], dtype=np.int)

    sigmas = np.array([
        [0, 1],
        [0, 2],
        [1, 2],
    ], dtype=np.int)

    signatures = np.column_stack([alphas, sigmas])

    sigmas_on_simplices = simplices[:, sigmas].reshape(-1, sigmas.shape[-1])
    sigmas_on_simplices.sort()

    if alphas.shape[-1] == 0:
        signatures_on_simplices = sigmas_on_simplices
    else:
        alphas_on_simplices = simplices[:, alphas].reshape(-1, alphas.shape[-1])
        alphas_on_simplices.sort()
        signatures_on_simplices = np.column_stack([
            alphas_on_simplices,
            sigmas_on_simplices
        ])

    print signatures_on_simplices

    # print sigmas_on_simplices

    unique, inverse = unique_rows(signatures_on_simplices)
    ordering = inverse.reshape(simplices.shape[0], signatures.shape[0])

    # consider factoring out a fem data {} object which
    # we can build into an extendable framework.

    import matplotlib.pyplot as plt

    u = whitney_elements(indices, coeffs, powers, wedges, ordering)
    K1 = inner(sc, u, u).todense()
    plt.figure()
    plt.title('My Inner Product')
    plt.imshow(K1, interpolation='nearest')

    K2 = whitney_innerproduct(sc, k=1).todense()
    plt.figure()
    plt.title('Current Implementation')
    plt.imshow(K2, interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    main()
