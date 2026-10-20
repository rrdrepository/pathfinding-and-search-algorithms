"""Problem 2: Console-based 9x9 Sudoku Solver using backtracking.

Enter nine rows. Each row must contain nine single digits separated by spaces.
Use 0 for an empty cell. Example:
5 3 0 0 7 0 0 0 0
"""

import time


GRID_SIZE = 9
BOX_SIZE = 3


def print_board(board):
    """Display a Sudoku board with separators for each 3x3 subgrid."""
    divider = "+-------+-------+-------+"

    for row_index, row in enumerate(board):
        if row_index % BOX_SIZE == 0:
            print(divider)

        row_display = []
        for column_index, value in enumerate(row):
            if column_index % BOX_SIZE == 0:
                row_display.append("|")
            row_display.append(str(value) if value != 0 else ".")
        row_display.append("|")
        print(" ".join(row_display))

    print(divider)


def find_empty_cell(board):
    """Return the first empty cell as (row, column), or None if there is none."""
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            if board[row][column] == 0:
                return row, column
    return None


def is_valid(board, row, column, number):
    """Check whether a number can be placed at board[row][column]."""
    # Check the row and column.
    for index in range(GRID_SIZE):
        if board[row][index] == number or board[index][column] == number:
            return False

    # Check the 3x3 subgrid containing the cell.
    start_row = (row // BOX_SIZE) * BOX_SIZE
    start_column = (column // BOX_SIZE) * BOX_SIZE
    for box_row in range(start_row, start_row + BOX_SIZE):
        for box_column in range(start_column, start_column + BOX_SIZE):
            if board[box_row][box_column] == number:
                return False

    return True


def find_duplicates(values):
    """Return the repeated non-zero values in a row, column, or subgrid."""
    seen = set()
    duplicates = set()

    for value in values:
        if value != 0:
            if value in seen:
                duplicates.add(value)
            seen.add(value)

    return sorted(duplicates)


def validate_initial_board(board):
    """Check the starting clues for duplicates in rows, columns, and 3x3 boxes."""
    for row in range(GRID_SIZE):
        duplicates = find_duplicates(board[row])
        if duplicates:
            return False, f"Duplicate digit(s) {duplicates} found in row {row + 1}."

    for column in range(GRID_SIZE):
        column_values = [board[row][column] for row in range(GRID_SIZE)]
        duplicates = find_duplicates(column_values)
        if duplicates:
            return False, f"Duplicate digit(s) {duplicates} found in column {column + 1}."

    for start_row in range(0, GRID_SIZE, BOX_SIZE):
        for start_column in range(0, GRID_SIZE, BOX_SIZE):
            box_values = []
            for row in range(start_row, start_row + BOX_SIZE):
                for column in range(start_column, start_column + BOX_SIZE):
                    box_values.append(board[row][column])

            duplicates = find_duplicates(box_values)
            if duplicates:
                box_number = (start_row // BOX_SIZE) * BOX_SIZE + (start_column // BOX_SIZE) + 1
                return False, f"Duplicate digit(s) {duplicates} found in 3x3 box {box_number}."

    return True, "The initial board is valid."


def solve_sudoku(board, statistics):
    """Solve the board recursively using backtracking.

    The function changes the board in place and returns True when a solution is
    found. A placement is counted whenever a valid candidate is placed. A
    backtrack is counted whenever that placement must later be removed.
    """
    empty_cell = find_empty_cell(board)

    # Base case: no empty cells remain, so the board is solved.
    if empty_cell is None:
        return True

    row, column = empty_cell
    for number in range(1, GRID_SIZE + 1):
        if is_valid(board, row, column, number):
            board[row][column] = number
            statistics["placements_attempted"] += 1

            if solve_sudoku(board, statistics):
                return True

            # This valid-looking choice led to a dead end, so undo it.
            board[row][column] = 0
            statistics["backtracks"] += 1

    return False


def read_row(row_number):
    """Read one valid Sudoku row, re-prompting until it is correctly entered."""
    while True:
        entered_row = input(f"Row {row_number} (nine digits, separated by spaces): ").strip()
        values = entered_row.split()

        if len(values) != GRID_SIZE:
            print("Invalid row. Enter exactly nine digits separated by spaces.")
            continue

        if any(len(value) != 1 or value not in "0123456789" for value in values):
            print("Invalid row. Use only single digits from 0 to 9; use 0 for an empty cell.")
            continue

        return [int(value) for value in values]


def read_board():
    """Read all nine rows of a Sudoku puzzle from the console."""
    print("Enter the Sudoku puzzle one row at a time.")
    print("Use 0 for an empty cell and separate every digit with a space.")

    board = []
    for row_number in range(1, GRID_SIZE + 1):
        board.append(read_row(row_number))
    return board


def main():
    """Run the Sudoku solver program."""
    print("=" * 48)
    print("       Sudoku Solver Using Backtracking")
    print("=" * 48)

    board = read_board()
    board_is_valid, validation_message = validate_initial_board(board)

    if not board_is_valid:
        print("\nThe initial puzzle is invalid.")
        print(validation_message)
        print("Please run the program again and enter a valid puzzle.")
        return

    print("\nInitial Sudoku board:")
    print_board(board)

    statistics = {"placements_attempted": 0, "backtracks": 0}
    start_time = time.perf_counter()
    solved = solve_sudoku(board, statistics)
    elapsed_time = time.perf_counter() - start_time

    if solved:
        print("\nSudoku solved successfully:")
        print_board(board)
    else:
        print("\nNo valid solution exists for this Sudoku puzzle.")
        print("The initial board is shown below:")
        print_board(board)

    print("Search statistics:")
    print(f"Placements attempted: {statistics['placements_attempted']}")
    print(f"Backtrack operations: {statistics['backtracks']}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    main()
