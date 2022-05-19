import numpy


def color_distance(c1: numpy.ndarray, c2: numpy.ndarray):
    # https://en.wikipedia.org/wiki/Color_difference
    # Cast to int
    c1 = [i.item() for i in c1]
    c2 = [i.item() for i in c2]

    r_bar = 0.5 * (c1[0] + c2[0])
    d_r = c1[0] - c2[0]
    d_g = c1[1] - c2[1]
    d_b = c1[2] - c2[2]

    if r_bar < 128:
        return (2 * d_r ** 2) + (4 * d_g ** 2) + (3 * d_b ** 2)
    else:
        return (3 * d_r ** 2) + (4 * d_g ** 2) + (2 * d_b ** 2)
