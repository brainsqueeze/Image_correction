#  __author__ = 'Dave'
import numpy as np


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
        return np.sqrt(np.dot(v, v))
    elif metric == 'arc_length':
        # find shortest unit arc length which passes through all unit circle points defined the population of
        # angles as well as passing through theta = 0.
        # if 0 not in population:
        #     population = np.concatenate((population, [0]))

        min_arc = 2 * np.pi
        population = sorted(population, reverse=True)
        pop_try = population[:]
        for i in xrange(len(population)):
            top = pop_try.pop(0)
            pop_try.append(top)
            # pop_try = translate(population, i)
            delta_s = sum(abs(np.diff(pop_try)))
            if delta_s < min_arc:
                min_arc = delta_s

        return min_arc


class Population:

    def __init__(self):
        self.father = None
        self.mother = None
        self.population = []
        self.children = []
        self.size = 0

    def exterminate(self):
        self.__init__()

    def _mutate(self, p_mutate, mutation):
        """
        Performs mutations on the children, stochastically, based on the mutation probability.
        :param p_mutate: (float) likelihood of a mutation occuring for each item in the sequence.
        :param mutation: (function object) user-defined function which determines the nature of the mutation.
        """
        self.children = mutation(self.children, p_mutate)

    def add_parents(self, sample, fitness, n_parents):
        """
        Parents are selected according to fitness probability.
        :param sample: (numpy array)
        :param fitness: (array)
        :param n_parents: (int)
        """
        self.size = len(sample)
        fitness = [np.pi - abs(val) for val in fitness]
        probabilities = [val / float(sum(fitness)) for val in fitness]
        idx, ind = 0, []
        for _ in xrange(n_parents):
            not_accepted = True
            while not_accepted:
                idx = int(self.size * np.random.random())
                if np.random.random() < float(probabilities[idx] / max(probabilities)):
                    not_accepted = False
            ind.append(idx)
        pop = sample[ind]

        self.father = pop[:n_parents / 2]
        self.mother = pop[n_parents / 2:n_parents]

    def add_children(self, mutation, p_mutate=0.01):
        """
        :param mutation: (function object) user-defined function which determines the nature of the mutation, can only
        accept the population and p_mutate as parameters.
        :param p_mutate: (float) likelihood of a mutation occuring for each item in the sequence.
        """

        if self.father is not None and self.mother is not None:
            self.population = np.concatenate((self.father, self.mother))
            parents = np.concatenate((self.father, self.mother))
            parents_length = len(parents)

            while len(self.children) < self.size - parents_length:
                male = np.random.randint(0, parents_length - 1)
                female = np.random.randint(0, parents_length - 1)
                if male != female:
                    child = []
                    male = parents[male:]
                    female = parents[:female]
                    half = len(male) / 2

                    if len(male) > 0 and len(female) > 0:
                        child = np.concatenate((male[half:], female[:half]))
                    elif len(male) > 0 and len(female) == 0:
                        child = male[half:]
                    elif len(male) == 0 and len(female) > 0:
                        child = female[:half]

                    if len(self.children) == 0 and len(child) > 0:
                        self.children = child
                    else:
                        self.children = np.concatenate((self.children, child))

            self._mutate(p_mutate, mutation=mutation)
            self.population = np.concatenate((self.population, self.children))
            self.population = self.population[:self.size]
            if len(self.population) < self.size:
                diff = self.size - len(self.population)
                gap = np.random.choice(self.population, size=diff)
                self.population = np.concatenate((self.population, gap))
        else:
            raise ValueError('Need to add parents.')
