from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, List
from math import floor, ceil

from bounding import BoundingBox
from color import color_distance


def write_pixel(array, row, col, color, source_img) -> int:
    canvas_pixel = array[row][col]
    source_pixel = source_img[row][col]
    prev_score = color_distance(canvas_pixel, source_pixel)

    for i in range(3):
        # Input is rgb, cv2 needs bgr, so invert color order
        array[row][col][i] = color[::-1][i]

    canvas_pixel = array[row][col]
    new_score = color_distance(canvas_pixel, source_pixel)
    return new_score - prev_score


class Shape(ABC):
    @staticmethod
    @abstractmethod
    def _generate_random(array) -> Shape:
        pass

    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        pass

    @abstractmethod
    def draw(self, array, source_img) -> int:
        """
        Draws the current shape into the canvas array and computes the
        score difference from the newly-drawn pixel.

        :param array: Canvas to draw the shape to
        :param source_img: Source image
        :return: an integer representing the difference between the old score and the new score
        """
        pass

    @abstractmethod
    def generate_child(self, random_dist: Callable, array) -> Shape:
        pass

    @staticmethod
    def get_random_shape(array) -> Shape:
        return random.choice([cls for cls in Shape.__subclasses__()])._generate_random(array)

    def __lt__(self, other):
        # lt only used when sorting the shapes array, order is irrelevant if both shapes have same score
        return True


class Circle(Shape):
    def __init__(self, row: int, col: int, radius: float, color: List[int]):
        self.row = row
        self.col = col
        self.radius = radius
        self.color = color
        self.bounding_box = BoundingBox(floor(row-radius), ceil(row+radius), floor(col-radius), ceil(col+radius))

    @staticmethod
    def _generate_random(array) -> Circle:
        width, height = len(array[0]), len(array)

        row = random.randint(0, height-1)
        col = random.randint(0, width-1)
        radius = random.randint(1, max(width, height)//2)
        color = get_random_color()
        return Circle(row, col, radius, color)

    def get_bounding_box(self) -> BoundingBox:
        return self.bounding_box

    def draw(self, array, source_img) -> int:
        width, height = len(array[0]), len(array)
        score_diff = 0

        r_square = self.radius ** 2

        # TODO redo using bounding box range?
        r_min = max(0, floor(self.row - self.radius))
        r_max = min(height, ceil(self.row + self.radius + 1))
        c_min = max(0, floor(self.col - self.radius))
        c_max = min(width, ceil(self.col + self.radius + 1))
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                distance_square = (r - self.row) ** 2 + (c - self.col) ** 2
                if distance_square <= r_square:
                    score_diff += write_pixel(array, r, c, self.color, source_img)
        return score_diff

    def generate_child(self, random_dist: Callable, array) -> Circle:
        width, height = len(array[0]), len(array)

        row = get_valid_random_int(self.row, 0, height, random_dist)
        col = get_valid_random_int(self.col, 0, width, random_dist)
        radius = get_valid_random_float(self.radius, 4, max(width, height), random_dist)
        color = get_color_variation(self.color, random_dist)

        return Circle(row, col, radius, color)


class Rectangle(Shape):
    def __init__(self, row: int, col: int, w: int, h: int, color: List[int]):
        self.row = row
        self.col = col
        self.w = w
        self.h = h
        self.color = color
        self.bounding_box = BoundingBox(row - h, row + h, col - w, col + w)

    @staticmethod
    def _generate_random(array) -> Shape:
        width, height = len(array[0]), len(array)

        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)
        w = random.randint(1, width//2)
        h = random.randint(1, height//2)
        color = get_random_color()
        return Rectangle(row, col, w, h, color)

    def get_bounding_box(self) -> BoundingBox:
        return self.bounding_box

    def draw(self, array, source_img) -> int:
        score_diff = 0
        for row in self.bounding_box.row_range(len(array)):
            for col in self.bounding_box.col_range(len(array[0])):
                score_diff += write_pixel(array, row, col, self.color, source_img)
        return score_diff

    def generate_child(self, random_dist: Callable, array) -> Shape:
        width, height = len(array[0]), len(array)

        row = get_valid_random_int(self.row, 0, height, random_dist)
        col = get_valid_random_int(self.col, 0, width, random_dist)
        w = get_valid_random_int(self.w, 1, width, random_dist)
        h = get_valid_random_int(self.h, 1, height, random_dist)
        color = get_color_variation(self.color, random_dist)

        return Rectangle(row, col, w, h, color)


# Util functions
def get_valid_random_float(curr, min_val, max_val, random_dist: Callable) -> float:
    # Multiply by either -1 or 1 to have offset in both directions
    offset = random_dist() * [-1, 1][random.randrange(2)]
    while not (min_val <= curr + offset < max_val):
        offset = random_dist() * [-1, 1][random.randrange(2)]
    return curr + offset


def get_valid_random_int(curr, min_val, max_val, random_dist: Callable) -> int:
    # Multiply by either -1 or 1 to have offset in both directions
    offset = round(random_dist() * [-1, 1][random.randrange(2)])
    while not (min_val <= curr + offset < max_val):
        offset = round(random_dist() * [-1, 1][random.randrange(2)])
    return curr + offset


def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def get_color_variation(curr_color, random_dist: Callable):
    new_color = [0, 0, 0]
    for i in range(3):
        new_color[i] = curr_color[i] + random_dist() * [-1, 1][random.randrange(2)]  # Add offset
        new_color[i] %= 256  # Cap to [0, 256) range
        # Note: Using mod to restrict range of random number is suboptimal, but should still
        # give us usable result that are not too far off
        # (256 looping down to 0 will be a big difference however, and might be a problem, to check)
    return new_color
