"""
This file contains the get_events function for a Hitori puzzle game.
It determines the consequences of a player's move to shade a new cell.
"""

import numpy as np
from enum import Enum, auto
from typing import List, Tuple, Set


class HitoriEvent(Enum):
    """
    Defines the possible events or rule violations that can occur
    when a user tries to shade a cell in a Hitori puzzle.
    """

    AlreadyShaded = auto()  # Trying to shade a cell which is already shaded
    RemovesDuplicate = auto()  # The number is duplicate either in row or column
    AdjacentShading = auto()  # Shading a cell adjacent to a previously shaded cell
    NewDisconnect = auto()  # New shading disconnects un-shaded (white) squares
    CompletesGame = auto()  # This move successfully solves the puzzle
    ShadesUnique = auto()  # Trying to shade a cell unique in its row/column


def _are_unshaded_connected(shaded: np.ndarray) -> bool:
    """
    Checks if all unshaded (0) cells in the grid are connected
    in a single component using Breadth-First Search (BFS).
    """
    N = shaded.shape[0]
    unshaded_coords = np.argwhere(shaded == 0)

    if unshaded_coords.shape[0] == 0:
        # If there are no unshaded cells, they are trivially "connected"
        # (though this is an invalid end state for Hitori).
        return True

    total_unshaded = unshaded_coords.shape[0]
    start_node = tuple(unshaded_coords[0])

    visited: Set[Tuple[int, int]] = set()
    queue: List[Tuple[int, int]] = []

    queue.append(start_node)
    visited.add(start_node)

    count = 0
    while queue:
        (r, c) = queue.pop(0)
        count += 1

        # Check all four neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            # Check if neighbor is in bounds, unshaded, and not visited
            if (
                0 <= nr < N
                and 0 <= nc < N
                and shaded[nr, nc] == 0
                and neighbor not in visited
            ):

                visited.add(neighbor)
                queue.append(neighbor)

    # If the number of visited nodes equals total unshaded, all are connected
    return count == total_unshaded


def _has_unshaded_duplicates(game_grid: np.ndarray, shaded: np.ndarray) -> bool:
    """
    Checks if the grid, given the current shading, has any duplicates
    among the unshaded (0) cells in any row or column.
    """
    N = game_grid.shape[0]

    # Check rows for unshaded duplicates
    for i in range(N):
        seen: Set[int] = set()
        for j in range(N):
            if shaded[i, j] == 0:  # If cell is unshaded
                num = game_grid[i, j]
                if num in seen:
                    return True  # Found a duplicate
                seen.add(num)

    # Check columns for unshaded duplicates
    for j in range(N):
        seen: Set[int] = set()
        for i in range(N):
            if shaded[i, j] == 0:  # If cell is unshaded
                num = game_grid[i, j]
                if num in seen:
                    return True  # Found a duplicate
                seen.add(num)

    return False  # No unshaded duplicates found


def get_events(
    game_grid: np.ndarray, shaded: np.ndarray, new_cell_shade: Tuple[int, int]
) -> Tuple[List[HitoriEvent], bool]:
    """
    Analyzes the effect of shading a new cell and returns a list of
    resulting events and a boolean indicating if the game should terminate.

    Args:
        game_grid: (N x N) grid with numbers.
        shaded: (N x N) grid (0=unshaded, 1=shaded) of the *current* state.
        new_cell_shade: (row, col) tuple of the cell to be shaded.

    Returns:
        A tuple containing:
        - A List of HitoriEvent enums describing the consequences.
        - A boolean 'is_terminating' which is True if a strict rule
          is violated (AdjacentShading, NewDisconnect) or if the
          game is completed.
    """

    events: List[HitoriEvent] = []
    is_terminating: bool = False

    (r, c) = new_cell_shade
    N = game_grid.shape[0]

    # --- 1. Check AlreadyShaded ---
    # If the cell is already shaded, this is a "no-op".
    # We return this event and stop processing.
    if shaded[r, c] == 1:
        return [HitoriEvent.AlreadyShaded], False

    # Create the hypothetical grid *after* the move
    hypothetical_shaded = shaded.copy()
    hypothetical_shaded[r, c] = 1

    # --- 2. Check AdjacentShading (Terminating Violation) ---
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        # Check if neighbor is in bounds and already shaded
        if 0 <= nr < N and 0 <= nc < N and shaded[nr, nc] == 1:
            events.append(HitoriEvent.AdjacentShading)
            is_terminating = True
            break  # Only need to find one adjacent shaded cell

    # --- 3. Check NewDisconnect (Terminating Violation) ---
    if not _are_unshaded_connected(hypothetical_shaded):
        events.append(HitoriEvent.NewDisconnect)
        is_terminating = True

    # --- 4. Check RemovesDuplicate / ShadesUnique ---
    num_to_shade = game_grid[r, c]
    # Count occurrences in the *original* full row and column
    row_count = np.count_nonzero(game_grid[r, :] == num_to_shade)
    col_count = np.count_nonzero(game_grid[:, c] == num_to_shade)

    if row_count > 1 or col_count > 1:
        # Shading this cell helps remove a duplicate
        events.append(HitoriEvent.RemovesDuplicate)
    else:
        # Shading this cell, which is already unique. This is usually a bad move.
        events.append(HitoriEvent.ShadesUnique)
        is_terminating = True

    # --- 5. Check CompletesGame (Terminating Event) ---
    # We only check for completion IF the move was not already a
    # terminating violation (i.e., no adjacent shading, no disconnect, no shades unique).
    if not is_terminating:
        # A game is complete if:
        # 1. No adjacent shaded cells (checked by AdjacentShading)
        # 2. All unshaded cells are connected (checked by NewDisconnect)
        # 3. No unshaded duplicates remain (checked by helper)
        if not _has_unshaded_duplicates(game_grid, hypothetical_shaded):
            events.append(HitoriEvent.CompletesGame)
            is_terminating = True

    return events, is_terminating


# --- Example Usage ---
if __name__ == "__main__":
    # Example Puzzle:
    # [[3 5 4 2 1 5]
    # [2 4 3 1 4 6]
    # [1 5 6 4 3 2]
    # [5 5 1 5 2 4]
    # [6 1 5 2 2 3]
    # [4 5 2 4 6 1]]

    puzzle = np.array(
        [
            [3, 5, 4, 2, 1, 5],
            [2, 4, 3, 1, 4, 6],
            [1, 5, 6, 4, 3, 2],
            [5, 5, 1, 5, 2, 4],
            [6, 1, 5, 2, 2, 3],
            [4, 5, 2, 4, 6, 1],
        ]
    )

    size = puzzle.shape[0]

    # Start with an empty board
    current_shaded = np.zeros((size, size), dtype=bool)

    while True:
        print(puzzle)
        print(current_shaded)
        move = input("Enter cell to shade (row,col) or 'q' to quit: ")
        if move.lower() == "q":
            break
        try:
            r_str, c_str = move.split(",")
            r, c = int(r_str), int(c_str)
            if not (0 <= r < size and 0 <= c < size):
                print("Invalid cell coordinates. Try again.")
                continue
        except ValueError:
            print("Invalid input format. Use 'row,col'. Try again.")
            continue

        events = get_events(puzzle, current_shaded, (r, c))
        print(f"Events for shading cell ({r}, {c}): {events}")
        # Apply the shading if it was not already shaded
        current_shaded[r, c] = True
