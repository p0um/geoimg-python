import random
from typing import Tuple, List
from math import log2
from tqdm import trange, tqdm
from multiprocessing import Pool

from shapes import Shape

import cv2
import numpy

# GLOBALS
# TODO Refactor into config file/cli args?

# Represents the number of shapes that will be used for the first iteration. Every shape will be randomly
# generated.
INITIAL_SHAPE_COUNT = 100

# Represents the number of shapes that will be drawn to the canvas after each iteration
# The number of drawn shapes may be less if some shapes give us a worse image
BEST_TO_DRAW = 5

# Represents the number of shapes that are kept after each iteration. For example, with a value of 300,
# if each iteration includes 500 shapes, the top 300 will be kept, and the bottom 200 will be deleted.
SURVIVOR_COUNT = 300

# Represents the number of offsprings each survivor shape gets. Offsprings have small variations from
# their parent shapes
OFFSPRING_COUNT = 5

# Represents the number of new shapes that will be added before each iteration. Those new shapes
# are randomly generated using the same method as the initial shapes (used in first iteration)
# Those shapes are not based off parent shapes and can vary wildly.
NEW_SHAPE_COUNT = 150


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


def get_score(img1, img2):
    height = len(img1)
    width = len(img1[0])

    score = 0
    for row in range(height):
        for col in range(width):
            score += color_distance(img1[row][col], img2[row][col])
    return score


def get_box_score(img1, img2, shape: Shape):
    height = len(img1)
    width = len(img1[0])

    bbox = shape.get_bounding_box()
    score = 0
    for row in bbox.row_range(height):
        for col in bbox.col_range(width):
            score += color_distance(img1[row][col], img2[row][col])
    return score


def get_weighted_random():
    # Weight based on 0.5^x, which favors smaller numbers
    return 5 * log2(-1/(random.random()-1))  # Scale to increase variance


def generate_children(current_score: int, shapes: List[Tuple[int, Shape]], canvas):
    new_shapes = list()
    for i in range(SURVIVOR_COUNT):
        # Copy survivor shapes to new array
        new_shapes[i] = current_score, shapes[i][1]

        # Generate offsprings from survivors
        for _ in range(OFFSPRING_COUNT):
            child = shapes[i][1].generate_child(get_weighted_random, canvas)
            new_shapes.append((current_score, child))

    # Generate new random shapes
    for i in range(NEW_SHAPE_COUNT):
        new_shapes.append((current_score, Shape.get_random_shape(canvas)))
    return new_shapes


def pool_job(score: int, shape: Shape, canvas: numpy.ndarray, source_img: numpy.ndarray):
    source_width, source_height = len(source_img[0]), len(source_img)

    canvas_copy = canvas.copy()
    if score == 0 or shape.get_bounding_box().get_size(source_width, source_height) >= source_img.size // 6:
        # No previous score or very large shape, calculate score over whole image
        shape.draw(canvas_copy)
        return get_score(source_img, canvas_copy)  # Update score for shape in list
    else:
        # Small shape, compute score of region overlapped by bounding box before and after shape
        prev_score = get_box_score(source_img, canvas_copy, shape)
        shape.draw(canvas_copy)
        new_score = get_box_score(source_img, canvas_copy, shape)
        return score + (new_score - prev_score)  # Increment previous score by bbox diff


def main():
    source_img = cv2.imread('sources/apples.jpg')
    source_width, source_height = len(source_img[0]), len(source_img)
    cv2.imshow('Original', source_img)
    cv2.waitKey(1)  # Wait 1ms to render image

    # Initialize the initial canvas to have the average color of the source image
    mean_color = numpy.mean(source_img, axis=(0, 1)).astype(numpy.uint8)
    canvas = numpy.zeros((source_height, source_width, 3), dtype=numpy.uint8)
    canvas[:] = mean_color

    shapes: List[Tuple[int, Shape]] = [(0, Shape.get_random_shape(canvas)) for _ in range(INITIAL_SHAPE_COUNT)]
    current_score = 0
    drawn_shapes = 0
    first_iter = True
    while drawn_shapes < 50000:
        # Generate shapes
        if not first_iter:
            shapes = generate_children(current_score, shapes, canvas)

        first_iter = False  # Set to false for future loops

        # Calculate best shape to add to image
        with Pool(processes=20) as p:
            args = list()
            for i in range(len(shapes)):
                score, shape = shapes[i]
                args.append((score, shape, canvas, source_img))

            score_list = p.starmap(pool_job, tqdm(args, desc=str(drawn_shapes), total=len(shapes), unit='shapes'))

        for i, score in enumerate(score_list):
            shapes[i] = score, shapes[i][1]  # Assign score (index 0) to every object

        shapes = sorted(shapes)  # Sort based on first index of tuple (score)

        # Since lower score is better, draw first shapes
        for i in range(BEST_TO_DRAW-1, -1, -1):
            # In the top n shapes, draw from worst to best to avoid overwriting best shape with lower quality one
            if shapes[i][0] < current_score:
                # Shape improves score
                shapes[i][1].draw(canvas)
                drawn_shapes += 1

        shapes = shapes[:SURVIVOR_COUNT]  # Delete low-scoring shapes
        cv2.imshow('Canvas', canvas)
        cv2.waitKey(1)  # Wait 1ms to render image
        current_score = get_score(source_img, canvas)

    cv2.waitKey(0)  # Wait until keypress


if __name__ == '__main__':
    main()
