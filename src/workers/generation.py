from __future__ import division
import numpy as np
from numbers import Number


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

    def add_parents(self, sample, fitness, max_parent_per_capita=1.0):
        """
        Parents are selected according to fitness probability.
        :param sample: (numpy array)
        :param fitness: (array)
        :param max_parent_per_capita: (float in interval [0., 1.0])
        """

        assert isinstance(max_parent_per_capita, Number) and 0 <= max_parent_per_capita <= 1.0
        self.size = len(sample)
        max_parent_size = int(max_parent_per_capita * self.size)

        probabilities = np.cos(fitness) ** 2
        r = np.random.random(size=self.size)
        parents = sample[r < probabilities]

        parent_size = min(parents.shape[0], max_parent_size)
        split = parent_size // 2

        self.father = parents[:split]
        self.mother = parents[split: parent_size]

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

            if parents_length < 2 or parents_length == self.size:
                print(self.population)

            while len(self.children) < self.size - parents_length:
                male = np.random.randint(0, parents_length - 1)
                female = np.random.randint(0, parents_length - 1)
                if male != female:
                    child = []
                    male = parents[male:]
                    female = parents[:female]
                    half = len(male) // 2

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
