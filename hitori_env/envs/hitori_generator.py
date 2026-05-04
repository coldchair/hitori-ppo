import numpy as np
from collections import deque
import random


def _check_connectivity(size: int, is_shaded: np.ndarray) -> bool:
    """
    Checks if all 'white' (False) cells in the grid are connected.
    Uses Breadth-First Search (BFS).
    """
    white_cells = np.argwhere(~is_shaded)
    if len(white_cells) == 0:
        # An all-black grid is not a valid puzzle, but it's not "disconnected"
        # We'll treat it as valid here, but generator should avoid this.
        return True

    total_white = len(white_cells)

    # Start BFS from the first white cell
    start_node = tuple(white_cells[0])
    q = deque([start_node])
    visited = {start_node}
    count = 0

    while q:
        r, c = q.popleft()
        count += 1

        # Check all four neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Check if neighbor is in bounds
            if 0 <= nr < size and 0 <= nc < size:
                neighbor = (nr, nc)
                # If neighbor is white and not visited
                if not is_shaded[nr, nc] and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)

    # All white cells are connected if BFS visited all of them
    return count == total_white


def _generate_latin_square(size: int) -> np.ndarray:
    """
    Generates a randomized Latin Square (no repeats in rows/cols).
    This will serve as our "solution" grid.
    """
    # Create a shuffled base row (e.g., [3, 1, 2])
    nums = np.arange(1, size + 1)
    np.random.shuffle(nums)

    # Create the square by rolling this base row
    grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        grid[i, :] = np.roll(nums, -i)

    # Further randomize by shuffling rows and columns
    np.random.shuffle(grid)  # Shuffles rows
    grid = grid.T
    np.random.shuffle(grid)  # Shuffles columns
    grid = grid.T

    return grid


def _generate_shading_pattern(size: int) -> np.ndarray:
    """
    Generates a valid shading pattern where:
    1. No two shaded cells are adjacent.
    2. All unshaded (white) cells are connected.
    """
    is_shaded = np.zeros((size, size), dtype=bool)
    num_cells = size * size

    # Aim to shade 15-30% of cells. More cells = harder puzzle.
    target_shaded_count = int(num_cells * np.random.uniform(0.15, 0.30))
    shaded_count = 0

    # Get a shuffled list of all possible cells to try shading
    possible_cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(possible_cells)

    for r, c in possible_cells:
        if shaded_count >= target_shaded_count:
            break

        # --- Check Rule 1: No adjacent shaded cells ---
        has_adjacent = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and is_shaded[nr, nc]:
                has_adjacent = True
                break

        if has_adjacent:
            continue  # Can't shade this one

        # --- Check Rule 2: White cells remain connected ---
        is_shaded[r, c] = True  # Temporarily shade it

        if not _check_connectivity(size, is_shaded):
            # Shading this cell disconnects the white cells, so undo it
            is_shaded[r, c] = False
        else:
            # This is a valid move
            shaded_count += 1

    # Ensure at least one cell is shaded for a non-trivial puzzle (if size > 1)
    if np.sum(is_shaded) == 0 and size > 1:
        # Failed to add any. Just force one random one.
        # Connectivity and adjacency are guaranteed for a single cell.
        r, c = np.random.randint(0, size, 2)
        is_shaded[r, c] = True

    return is_shaded


def generate_random_hitori_game(size: int, seed=None) -> np.ndarray:
    """
    Generates a random, valid Hitori puzzle of a given size.

    A valid puzzle is one that has at least one solution. This function
    constructs the puzzle from a known solution, guaranteeing its validity.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer")

    if size == 1:
        return np.array([[1]])  # Trivial case

    # 1. Create the "solved" grid (no duplicates)
    solution_grid = _generate_latin_square(size)

    # 2. Create the shading pattern
    #    Keep trying until we get a non-trivial pattern (at least one shaded cell)
    is_shaded = np.zeros((size, size), dtype=bool)
    while np.sum(is_shaded) == 0:
        is_shaded = _generate_shading_pattern(size)

    # 3. Create the puzzle grid from the solution and shading
    puzzle_grid = solution_grid.copy()

    # Get coordinates of all shaded and white cells
    shaded_indices = np.argwhere(is_shaded)
    white_indices = np.argwhere(~is_shaded)

    # 4. "Plant" duplicates for each shaded cell
    for r, c in shaded_indices:
        # Find all white cells in this cell's row or column
        possible_targets = []
        # Check row
        for j in range(size):
            if not is_shaded[r, j]:
                possible_targets.append((r, j))
        # Check column
        for i in range(size):
            if not is_shaded[i, c]:
                possible_targets.append((i, c))

        if not possible_targets:
            # This should not happen with a valid shading pattern
            # (a shaded cell must be next to at least one white cell)
            # Fallback: pick any white cell in the grid
            target_idx = np.random.randint(len(white_indices))
            target_r, target_c = white_indices[target_idx]
        else:
            # Pick a random white cell from the row/column
            target_idx = np.random.randint(len(possible_targets))
            target_r, target_c = possible_targets[target_idx]

        # Set the puzzle's shaded cell to match the target's solution value
        puzzle_grid[r, c] = solution_grid[target_r, target_c]

    return puzzle_grid


# --- Example Usage ---
if __name__ == "__main__":
    puzzle_size = 6
    hitori_puzzle = generate_random_hitori_game(puzzle_size)

    print(f"Generated {puzzle_size}x{puzzle_size} Hitori Puzzle:")
    print(hitori_puzzle)
