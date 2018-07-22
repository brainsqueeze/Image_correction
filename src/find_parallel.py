from __future__ import print_function

from .workers.correct import CorrectImage
from .workers.optimization_utils import loss, health
from .workers.generation import Population

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import warp, PiecewiseAffineTransform


def evolve(image_class, pop_class, pairs, mutation_probability):
    """
    :param image_class: (class object)
    :param pop_class: (class object)
    :param pairs: (numpy array with dimensions (n_pairs, 2, 2, 2)) pairs of lines
    :param mutation_probability: (float)
    :return: (numpy array) loss functions for the population
    """

    pop_class.exterminate()
    slopes = image_class.slope(pairs)
    loss_func = loss(m1=slopes[:, 0], m2=slopes[:, 1])

    pop_class.add_parents(
        sample=pairs,
        fitness=loss_func,
        max_parent_per_capita=0.3
    )
    pop_class.add_children(mutation=image_class.mutation, p_mutate=mutation_probability)
    pop_slopes = image_class.slope(pop_class.population)
    pop_loss = loss(m1=pop_slopes[:, 0], m2=pop_slopes[:, 1])
    return pop_loss


def draw(pairs):
    plt.clf()
    x = pairs.shape
    pairs = pairs.reshape(x[0] * x[1], x[2], x[3])
    for idx, line in enumerate(pairs):
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show()


def visualize_arc_length(health_measure):
    plt.clf()
    x = np.arange(0, 1.01, 0.01)
    plt.scatter(np.cos(health_measure), np.sin(health_measure))
    plt.plot(x, np.sqrt(1 - x ** 2))
    plt.plot(x, -np.sqrt(1 - x ** 2))
    plt.xlim((-1.2, 1.2))
    plt.ylim((-1.2, 1.2))
    plt.show()


def affine_transform(img):
    rows, cols = img.shape[0], img.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 20)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1]  # - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    print(src[:, 1])
    print(src[:, 0])
    dst_cols = src[:, 0] - np.sin((src[:, 0] / np.max(src[:, 0])) * np.pi) * np.max(src[:, 0])
    print(dst_cols)

    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = rows
    out_cols = cols
    out = warp(img, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(out)
    ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    ax.axis((0, out_cols, out_rows, 0))
    plt.savefig('plots/piecewise_affine.png')
    plt.show()


def main():
    num_epochs = 10000
    pop = Population()
    pic = CorrectImage()

    pic.add_path('data')
    pic.add_image('initial.png')
    pic.hough_transform(vary=False, plot=False)  # set vary True to change edge filters, plot True for visualizations
    pair = pic.line_pair(800)

    total = 0
    mean = []
    avg_angle, count = 0, 0
    for epoch in range(num_epochs):
        loss_func = evolve(pic, pop, pairs=pair, mutation_probability=0.01)

        pop_health = health(loss_func, metric='arc_length')
        total += pop_health
        mean.append(total / (epoch + 1))

        pair = pop.population

        if (epoch + 1) % 100 == 0:
            print("Generation", epoch + 1, "health:", pop_health, "epoch averaged:", total / (epoch + 1))
            # visualize_arc_length(loss_func)
            # draw(pair)

        count += 1
        avg_angle += abs(np.mean(loss_func))

    avg_angle /= count
    print("Average angle of distortion:", avg_angle * (180 / np.pi))

    plt.clf()
    plt.plot(range(num_epochs), mean)
    plt.xlabel('Generations')
    plt.ylabel('Loss function')
    plt.savefig('plots/generational_performance.png')


if __name__ == '__main__':
    main()
