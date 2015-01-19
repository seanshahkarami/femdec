import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement
from pydec import simplicial_complex
from innerproduct import whitney_innerproduct
from innerproduct import projection
from innerproduct import quadrature_nodes_and_weights
from itertools import izip


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
        self.dimension = np.max(ordering)+1
        self.num_terms = len(coeffs)

    def __iter__(self):
        return izip(self.indices,
                    self.coeffs,
                    self.powers,
                    self.wedges)

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


def unique_rows(a):
    a = np.ascontiguousarray(a)
    b = a.view(np.dtype((np.void, a.itemsize * a.shape[-1])))
    return np.unique(b, return_inverse=True)


def compute_ordering(simplices, alphas, sigmas):
    num_simplices = simplices.shape[0]

    sigmas_on_simplices = simplices[:, sigmas].reshape(-1, sigmas.shape[-1])
    sigmas_on_simplices.sort()

    if alphas is None or alphas.shape[-1] == 0:
        signatures_on_simplices = sigmas_on_simplices
    else:
        alphas_on_simplices = simplices[:, alphas].reshape(-1, alphas.shape[-1])
        alphas_on_simplices.sort()
        signatures_on_simplices = np.column_stack([
            alphas_on_simplices,
            sigmas_on_simplices
        ])

    unique, inverse = unique_rows(signatures_on_simplices)
    ordering = inverse.reshape(num_simplices, -1)

    return ordering


def read_mesh():
    vertices = np.loadtxt('square.node', skiprows=1, usecols=(1, 2), dtype=np.float)
    vertices = np.column_stack([vertices, np.zeros(vertices.shape[0])])
    simplices = np.loadtxt('square.ele', skiprows=1, usecols=(1, 2, 3), dtype=np.int)
    return simplicial_complex(vertices, simplices)


def main():
    sc = read_mesh()

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

    ordering = compute_ordering(sc[-1].simplices, alphas, sigmas)

    def f1(x, y, z): return x
    def f2(x, y, z): return y
    f = [(f1, np.array([0])), (f2, np.array([1]))]

    u = whitney_elements(indices, coeffs, powers, wedges, ordering)
    # K = whitney_innerproduct(sc, u, u)
    projection(sc, u, f)


if __name__ == '__main__':
    main()
