
class BoundingBox:
    def __init__(self, min_row: int, max_row: int, min_col: int, max_col: int):
        self.min_row = max(0, min_row)
        self.max_row = max_row
        self.min_col = max(0, min_col)
        self.max_col = max_col

    def row_range(self, height):
        end = min(height, self.max_row + 1)
        return range(self.min_row, end)

    def col_range(self, width):
        end = min(width, self.max_col + 1)
        return range(self.min_col, end)

    def get_size(self, width, height):
        box_height = min(height, self.max_row) - self.min_row
        box_width = min(width, self.max_col) - self.min_col
        return box_height * box_width
