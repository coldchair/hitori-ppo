import numpy as np
from .hitori_events import HitoriEvent, get_events


def generate_solution(game_grid: np.ndarray) -> np.ndarray:
    """Generates the solution for a given Hitori game grid using backtracking.

    Args:
        game_grid (np.ndarray): The Hitori game grid (size NxN).

    Returns:
        np.ndarray: The solution shaded grid (size NxN) where each cell is
                    either 0 (unshaded) or 1 (shaded).
    """
    N = game_grid.shape[0]
    # The solution grid we are building in-place
    shaded = np.zeros_like(game_grid, dtype=int)

    def check_final_solution():
        """
        Checks if the *current* 'shaded' grid is a valid and complete solution.
        This is the main "victory condition" for the base case of the recursion.

        The 'get_events' function checks for adjacency and connectivity violations
        incrementally, so we only need to check for the "no duplicates" rule
        in the final unshaded grid.
        """
        for i in range(N):
            row_vals = set()
            col_vals = set()
            for j in range(N):
                # Check row for unshaded duplicates
                if shaded[i, j] == 0:  # If unshaded
                    val = game_grid[i, j]
                    if val in row_vals:
                        return False  # Found a row duplicate
                    row_vals.add(val)

                # Check column for unshaded duplicates
                if shaded[j, i] == 0:  # If unshaded
                    val = game_grid[j, i]
                    if val in col_vals:
                        return False  # Found a column duplicate
                    col_vals.add(val)

        # If we passed all checks, it's a valid solution
        return True

    def solve(r, c):
        """
        Recursive helper to find a solution by deciding the
        state of cell (r, c) and all subsequent cells.

        Returns True if a solution is found, False otherwise.
        """

        # --- Base Case ---
        if r == N:
            # We have made a decision (shade or unshade) for every cell.
            # Now we must check if this complete configuration is a valid solution.
            return check_final_solution()

        # --- Calculate next cell position ---
        # We iterate row by row, column by column
        next_r, next_c = r, c + 1
        if next_c == N:
            next_r, next_c = r + 1, 0

        # --- Decision 1: Try UNSHADING cell (r, c) ---
        # This is the "default" state. `shaded[r, c]` is already 0.
        # We just recurse to the next cell.
        if solve(next_r, next_c):
            return True  # A solution was found down this path

        # --- Decision 2: Try SHADING cell (r, c) ---
        # If "unshading" (Decision 1) didn't lead to a solution,
        # we *must* try shading this cell.

        # First, check if shading this cell is even a valid move.
        # `get_events` checks the current state (`shaded`) + the new move ((r,c)).
        events, is_terminating = get_events(game_grid, shaded, (r, c))

        # Check for *strict game-ending violations*
        # (is_terminating is True for AdjacentShading and NewDisconnect)
        if is_terminating and HitoriEvent.CompletesGame not in events:
            # This move is 100% illegal.
            # Since unshading (Decision 1) also failed, this
            # entire branch of the search is a dead end.
            return False

        # --- If we are here, shading is a *potentially* valid move ---

        # 1. Apply the move
        shaded[r, c] = 1

        # 2. Check for an "instant win" (optimization)
        if HitoriEvent.CompletesGame in events:
            return True  # This move solved the puzzle!

        # 3. Recurse to the next cell
        if solve(next_r, next_c):
            return True  # This path (starting with this shade) led to a solution

        # --- Backtrack ---
        # If we are here, it means shading cell (r, c) did *not*
        # lead to a solution. We must undo our move.
        shaded[r, c] = 0

        # Report failure for this path
        return False

    # --- Start the backtracking process from cell (0, 0) ---
    if solve(0, 0):
        # `solve` modifies `shaded` in-place.
        return shaded
    else:
        raise ValueError("No solution exists for the given Hitori game grid.")
