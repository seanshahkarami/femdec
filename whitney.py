import numpy as np
import numpy.linalg as la
from itertools import combinations
from itertools import combinations_with_replacement
from scipy.sparse import coo_matrix
from scipy.special import gamma
from quadrature import Quadrature


class whitney_term(object):

    def __init__(self, coeff, power, wedge):
        self.coeff = coeff
        self.power = np.asarray(power)
        self.wedge = np.asarray(wedge)


class whitney_element(object):

    def __init__(self, coeffs, powers, wedges):
        self.coeffs = coeffs
        self.powers = powers
        self.wedges = wedges


class WhitneySignature(object):

    def __init__(self, alpha, sigma):
        self.alpha = np.asarray(alpha, dtype=np.int)
        self.sigma = np.asarray(sigma, dtype=np.int)

    def key_for_simplex(self, simplex):
        return np.sort(simplex[self.alpha]).tostring() + \
            np.sort(simplex[self.sigma]).tostring()


class whitney_elements(object):

    def __init__(self, complex, k, r, elements, ordering):
        self.complex = complex
        self.k = k
        self.r = r

        self.ordering = ordering

        self.elements = elements
        self.index = np.ravel(self.elements[0])
        self.coeff = np.ravel(self.elements[1])
        self.power = ravel_2d(self.elements[2])
        self.wedge = ravel_2d(self.elements[3])

    def __len__(self):
        return len(self.ordering)


def whitney_basis(sc, k, r):
    assert 1 <= k <= sc.complex_dimension()
    assert 1 <= r

    signatures = afw_signatures(sc, k, r)

    elements = build_elements(sc, signatures, k, r)
    ordering = build_ordering(sc, signatures)

    return whitney_elements(sc, k, r, elements, ordering)


def afw_signatures(sc, k, r):
    reference_simplex = np.arange(sc.complex_dimension()+1)

    signatures = []

    for sigma in combinations(reference_simplex, k+1):
        for alpha in combinations_with_replacement(sigma, r-1):
            signatures.append(WhitneySignature(alpha, sigma))

    return signatures


def build_ordering(sc, signatures):
    simplices = np.sort(sc[-1].simplices)

    ordering = np.empty((len(simplices), len(signatures)), dtype=np.int)

    indices = {}

    for simplex, ordering_on_simplex in zip(simplices, ordering):
        for i, sig in enumerate(signatures):
            key = sig.key_for_simplex(simplex)
            if key not in indices:
                indices[key] = len(indices)
            ordering_on_simplex[i] = indices[key]

    return ordering


# basis = whitney_basis(sc, k, r, basis='afw')
# d_basis = basis.exterior_derivative()


def build_elements(sc, signatures, k, r):
    index = np.empty((len(signatures), k+1), dtype=np.int)
    coeff = np.empty((len(signatures), k+1), dtype=np.float)
    power = np.empty((len(signatures), k+1, sc.complex_dimension()+1), dtype=np.int)
    wedge = np.empty((len(signatures), k+1, k), dtype=np.int)

    index[:] = np.arange(len(signatures))[:, None]

    coeff[:, ::2] = 1.0
    coeff[:, 1::2] = -1.0

    for i, sig in enumerate(signatures):
        power[i, :] = np.bincount(sig.alpha, minlength=power.shape[-1])

        for j in xrange(k+1):
            power[i, j, sig.sigma[j]] += 1
            wedge[i, j, :j] = sig.sigma[:j]
            wedge[i, j, j:] = sig.sigma[j+1:]

    return index, coeff, power, wedge


def exterior_derivative(element):
    # may want to think of this as a term list...
    coeffs, powers, wedges = element

    # Preallocate space for exterior derivative terms.
    capacity = wedges.shape[0] * powers.shape[1]

    d_coeffs = np.empty(capacity, dtype=coeffs.dtype)
    d_powers = np.empty((capacity, powers.shape[1]), dtype=powers.dtype)
    d_wedges = np.empty((capacity, wedges.shape[1]+1), dtype=wedges.dtype)

    count = 0

    for coeff, power, wedge in zip(coeffs, powers, wedges):
        for i in xrange(powers.shape[1]):
            if power[i] > 0 and i not in wedge:
                d_coeffs[count] = coeff * power[i]

                d_powers[count, :] = power
                d_powers[count, i] -= 1

                d_wedges[count, 0] = i
                d_wedges[count, 1:] = wedge

                count += 1

    return d_coeffs[:count], d_powers[:count], d_wedges[:count]


def inner(u, v):
    vertices = u.complex.vertices
    simplices = u.complex.simplices

    coeff = np.ravel(u.coeff[:, None] * v.coeff)
    power = ravel_2d(u.power[:, None] + v.power)

    index1, index2 = map(np.ravel, np.meshgrid(u.index, v.index))

    rows = u.ordering[:, index1]
    cols = v.ordering[:, index2]
    data = np.ones_like(rows, dtype=np.float)

    for i, simplex in enumerate(simplices):
        pass

    data *= coeff
    data *= simplex_beta(power+1)

    return coo_matrix((data.ravel(), (rows.ravel(), cols.ravel()))).tocsr()


def projection(u, f):
    vertices = u.complex.vertices
    simplices = u.complex.simplices

    quadrature = Quadrature(u.complex, order=4)

    # f(quadrature.nodes.reshape(...)).reshape(...)

    points = vertices[simplices]
    frames = points[:, 1:] - points[:, 0, np.newaxis]

    for func, wedge in f:
        total = 0.0
        for simplex in simplices:
            points = vertices[simplex]
            T = points[1:] - points[0]
            volume = np.sqrt(la.det(np.inner(T, T))) / 2.0
            nodes = np.dot(T, quadrature.nodes) + points[0, :, None]
            total += np.sum(func(nodes) * quadrature.weights) * volume
        print total

    # u = WhitneyElements(sc, k=2, r=2)
    # q = Quadrature(sc, order=8)


def ravel_2d(a):
    return a.reshape(-1, a.shape[-1])


def simplex_beta(a):
    return np.prod(gamma(a), axis=-1) / gamma(np.sum(a, axis=-1))
