from workers.correct import CorrectImage
from workers.optimize import loss, health, Population

import numpy as np
import matplotlib.pyplot as plt


def evolve(image_class, pop_class, pairs, mutation_probability):
    """
    :param image_class: (class object)
    :param pop_class: (class object)
    :param pairs: (numpy array with dimensions (n_pairs, 2, 2, 2)) pairs of lines
    :param mutation_probability: (float)
    :return: (numpy array) loss functions for the population
    """
    pop_class.exterminate()
    slopes = map(image_class.slope, pairs)
    loss_func = map(lambda s: loss(np.array(s[0]), np.array(s[1])), slopes)

    pop_class.add_parents(pairs, loss_func, 200)
    pop_class.add_children(image_class.mutation, p_mutate=mutation_probability)
    pop_slopes = map(image_class.slope, pop_class.population)
    pop_loss = map(lambda s: loss(np.array(s[0]), np.array(s[1])), pop_slopes)
    return pop_loss


def draw(pairs):
    plt.clf()
    x = pairs.shape
    pairs = pairs.reshape(x[0] * x[1], x[2], x[3])
    for idx, line in enumerate(pairs):
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show()


def main():
    pop = Population()
    pic = CorrectImage()

    pic.add_path('data')
    pic.add_image('initial.png')

    pic.hough_transform(vary=False, plot=False)  # set vary True to change edge filters, plot True for visualizations
    pair = pic.line_pair(800)

    total = 0
    mean = []
    avg_angle, count = 0, 0
    for _ in xrange(500):
        loss_func = evolve(pic, pop, pairs=pair, mutation_probability=0.01)

        pop_health = health(loss_func, metric='arc_length')
        total += pop_health
        mean.append(total / (_ + 1))

        pair = pop.population

        if _ % 100 == 0:
            print "Generation {0} health: \t{1}".format(_, pop_health), total / (_ + 1)
            # draw(pair)

        if (_ + 1) > 300:
            count += 1
            avg_angle += abs(np.mean(loss_func))
    avg_angle /= count
    print(avg_angle)

    plt.clf()
    plt.plot(range(500), mean)
    plt.savefig('plots/generational_performance.png')
    plt.show()

if __name__ == '__main__':
    main()
