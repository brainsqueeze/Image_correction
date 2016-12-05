from workers.correct import CorrectImage
from workers.optimize import loss, health, Population

import numpy as np


def main():
    # initialization
    pic = CorrectImage()
    pic.add_path('data')
    pic.add_image('initial.png')

    pic.hough_transform(vary=False, plot=False)  # set vary True to change edge filters, plot True for visualizations
    pair = pic.line_pair(16)
    slopes = map(pic.slope, pair)
    loss_func = map(lambda s: loss(np.array(s[0]), np.array(s[1])), slopes)

    # print pair
    print loss_func
    print health(loss_func, metric='arc_length'), '\n'

    pop = Population()
    pop.add_parents(pair, loss_func, 4)
    pop.add_children(pic.mutation, p_mutate=0.01)

if __name__ == '__main__':
    main()
