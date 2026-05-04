import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .hitori_generator import generate_random_hitori_game
from .hitori_solution import generate_solution
from typing import Any, Tuple, List, Set
from collections import Counter


class HitoriEnv(gym.Env):
    """
    A Gymnasium environment for the Hitori puzzle game.

    The environment follows the "mask-based" approach, where illegal
    actions are prevented by an action_mask rather than punished
    after the fact.

    Rewards:
    - +1.0: Completing the puzzle (terminal).
    - -1.0: Getting "stuck" with no valid moves left (terminal).
    - -0.01: For every valid step taken (step penalty).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "game_grid": spaces.Box(
                    low=1, high=self.size, shape=(size, size), dtype=np.uint32
                ),
                "shaded": spaces.MultiBinary((size, size)),
            }
        )

        self.action_space = spaces.Discrete(size * size)

        # These will be set in reset()
        self._game_grid = np.ones((size, size), dtype=np.uint32)
        self._shaded = np.zeros((self.size, self.size), dtype=bool)
        self._action_mask = np.ones(self.size * self.size, dtype=np.int8)

        # --- Optimization Caches ---
        # These will be populated in reset()
        self._row_counts: List[Counter] = []
        self._col_counts: List[Counter] = []
        # This will be populated in _compute_next_action_mask()
        self._articulation_points: Set[Tuple[int, int]] = set()
        # --- End Optimization Caches ---

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # --- Core Gym Methods ---

    def reset(self, seed=None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._game_grid = generate_random_hitori_game(self.size, seed=seed).astype(
            np.uint32
        )

        if (
            options is not None
            and "log_solution" in options
            and options["log_solution"]
        ):
            print("Solution:")
            print(generate_solution(self._game_grid))

        self._shaded = np.zeros((self.size, self.size), dtype=bool)

        # --- Populate Optimization Caches ---
        self._row_counts = [Counter(self._game_grid[r, :]) for r in range(self.size)]
        self._col_counts = [Counter(self._game_grid[:, c]) for c in range(self.size)]
        # --- End Populate ---

        # Compute the initial action mask based on the starting state
        self._action_mask = self._compute_next_action_mask()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int):
        # The action is guaranteed by the mask to be valid (not shaded,
        # not adjacent, not disconnecting, not shading a unique cell)

        row, col = self._action_to_coords(action)
        self._shaded[row, col] = True

        # --- Check for Win Condition ---
        is_complete = self._check_completes_game()

        if is_complete:
            reward = 1.0  # Big reward for winning
            terminated = True
            self._action_mask.fill(0)  # No more actions possible
        else:
            # --- Not a win, so compute next mask and check for "Stuck" ---
            self._action_mask = self._compute_next_action_mask()

            is_stuck = not np.any(self._action_mask)

            if is_stuck:
                reward = -1.0  # Big penalty for getting stuck
                terminated = True
            else:
                reward = -0.01  # Small step penalty to encourage efficiency
                terminated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def action_masks(self):
        """Used by MaskablePPO to get the valid actions."""
        return self._action_mask

    def _get_obs(self):
        return {
            "game_grid": self._game_grid,
            "shaded": self._shaded,
        }

    def _get_info(self):
        return {}

    # --- Action Mask Computation ---

    def _compute_next_action_mask(self) -> np.ndarray:
        """
        Calculates the action mask for the *current* state.
        A '0' means the action is illegal.
        """
        new_mask = np.ones(self.size * self.size, dtype=np.int8)

        # 1. Mask already-shaded cells
        shaded_flat_indices = np.where(self._shaded.flatten())[0]
        new_mask[shaded_flat_indices] = 0

        self._articulation_points = self._find_unshaded_articulation_points()

        for action_idx in range(self.size * self.size):
            if new_mask[action_idx] == 0:
                continue  # Already masked (is shaded)

            row, col = self._action_to_coords(action_idx)

            # 2. Mask for AdjacentShading
            if self._check_adjacent_shading(row, col):
                new_mask[action_idx] = 0
                continue

            # 3. Mask for NewDisconnect
            if (row, col) in self._articulation_points:
                new_mask[action_idx] = 0
                continue

            # 4. Mask for ShadesUnique
            if self._check_shades_unique(row, col):
                new_mask[action_idx] = 0
                continue

        return new_mask

    # --- Hitori Rule Checkers (used for mask and win condition) ---

    def _check_adjacent_shading(self, row: int, col: int) -> bool:
        """Checks if shading (row, col) would be adjacent to an *existing* shaded cell."""
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self._shaded[nr, nc]:
                    return True  # Found adjacent shaded cell
        return False

    def _check_shades_unique(self, row: int, col: int) -> bool:
        """
        Checks if the cell (row, col) is unique in its row AND column.
        This is an O(1) check using the precomputed caches.
        """
        val = self._game_grid[row, col]

        # Get counts from the precomputed caches
        row_count = self._row_counts[row][val]
        col_count = self._col_counts[col][val]

        # If it's unique in its row AND unique in its col, shading it is illegal.
        return row_count == 1 and col_count == 1

    def _check_completes_game(self) -> bool:
        """
        Checks if the *current* state is a valid, complete solution.
        Assumes adjacent/connectivity rules are already met (by the mask).
        We only need to check for remaining unshaded duplicates.
        """
        return not self._has_unshaded_duplicates(self._shaded)

    # --- Helper Functions ---

    def _find_unshaded_articulation_points(self) -> Set[Tuple[int, int]]:
        """
        Finds all articulation points (cut vertices) in the current graph
        of unshaded cells using a single DFS pass.
        Runs in O(N^2) time, where N = self.size.
        """
        unshaded_coords = np.argwhere(self._shaded == 0)
        if unshaded_coords.shape[0] == 0:
            return set()

        # -1 = unvisited
        disc = np.ones((self.size, self.size), dtype=int) * -1
        low = np.ones((self.size, self.size), dtype=int) * -1

        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        articulation_points: Set[Tuple[int, int]] = set()
        time = 0

        # Find the first unshaded cell to start the DFS
        start_node = tuple(unshaded_coords[0])

        # We use a stack for an iterative DFS to avoid recursion depth limits
        stack: List[Tuple[Tuple[int, int], Tuple[int, int] | None]] = [
            (start_node, None)
        ]  # (node, parent)

        # This dict helps manage the recursive-like state in an iterative loop
        # It stores (children_count, neighbor_iterator)
        dfs_state: dict[tuple[int, int], Any] = {}

        while stack:
            current_node, parent_node = stack[-1]
            r, c = current_node

            if current_node not in dfs_state:
                # First time visiting this node: initialize
                disc[r, c] = low[r, c] = time
                time += 1
                parent[current_node] = parent_node

                # Prepare neighbor iterator and child count
                neighbors = []
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self.size
                        and 0 <= nc < self.size
                        and not self._shaded[nr, nc]
                    ):
                        neighbors.append((nr, nc))

                dfs_state[current_node] = [
                    0,
                    iter(neighbors),
                ]  # [child_count, neighbor_iter]

            # --- Process next neighbor ---
            state = dfs_state[current_node]
            try:
                neighbor = next(state[1])
                nr, nc = neighbor

                if neighbor == parent_node:
                    continue  # Skip parent

                if disc[nr, nc] != -1:
                    # Back-edge to an already visited node (not parent)
                    low[r, c] = min(low[r, c], disc[nr, nc])
                else:
                    # New node (tree-edge) - push to stack
                    state[0] += 1  # Increment child count
                    stack.append((neighbor, current_node))

            except StopIteration:
                # --- All neighbors processed, popping node from stack ---
                stack.pop()
                if parent_node is None:
                    # Root node check
                    if state[0] > 1:  # Root is AP if it has > 1 DFS children
                        articulation_points.add(current_node)
                else:
                    # Non-root node: update parent's low-link
                    pr, pc = parent_node
                    low[pr, pc] = min(low[pr, pc], low[r, c])

                    # AP check: non-root is AP if a child's low-link is >= self's discovery time
                    if low[r, c] >= disc[pr, pc]:
                        articulation_points.add(parent_node)

        return articulation_points

    def _has_unshaded_duplicates(self, shaded: np.ndarray) -> bool:
        """Checks for duplicates among unshaded cells in any row or column."""
        # Check rows
        for i in range(self.size):
            seen: Set[int] = set()
            for j in range(self.size):
                if not shaded[i, j]:  # If cell is unshaded
                    num = self._game_grid[i, j]
                    if num in seen:
                        return True  # Found a duplicate
                    seen.add(num)

        # Check columns
        for j in range(self.size):
            seen: Set[int] = set()
            for i in range(self.size):
                if not shaded[i, j]:  # If cell is unshaded
                    num = self._game_grid[i, j]
                    if num in seen:
                        return True  # Found a duplicate
                    seen.add(num)

        return False  # No unshaded duplicates found

    # --- Utility and Rendering ---

    def _action_to_coords(self, action: int) -> tuple[int, int]:
        """Convert 1D action index to 2D grid coordinates"""
        row = action // self.size
        col = action % self.size
        return (row, col)

    def _coords_to_action(self, row: int, col: int) -> int:
        """Convert 2D grid coordinates to 1D action index"""
        return row * self.size + col

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        # Create canvas and fill background
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Size of each grid square (integer pixels)
        pix_square_size = max(1, self.window_size // self.size)

        # Prepare font for rendering numbers
        try:
            font = pygame.font.SysFont(None, max(12, int(pix_square_size * 0.5)))
        except Exception:
            # Ensure font module is initialized
            pygame.font.init()
            font = pygame.font.SysFont(None, max(12, int(pix_square_size * 0.5)))

        # Draw each cell: shaded cells as dark rectangles, unshaded cells show the number
        for r in range(self.size):
            for c in range(self.size):
                x = c * pix_square_size
                y = r * pix_square_size
                rect = pygame.Rect(x, y, pix_square_size, pix_square_size)

                # Determine cell and text colors
                if self._shaded[r, c]:
                    bg_color = (50, 50, 50)
                    text_color = (200, 200, 200)  # Light gray for visibility
                else:
                    action_index = self._coords_to_action(r, c)
                    if self.action_masks()[action_index]:
                        bg_color = (220, 230, 255)  # Light blue for playable
                    else:
                        bg_color = (255, 255, 255)  # White for unshaded
                    text_color = (0, 0, 0)

                # Draw cell background
                pygame.draw.rect(canvas, bg_color, rect)

                # Draw the number
                try:
                    num = int(self._game_grid[r, c])
                except Exception:
                    num = self._game_grid[r, c]
                text_surf = font.render(str(num), True, text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                canvas.blit(text_surf, text_rect)

        # Draw grid lines
        for i in range(self.size + 1):
            # horizontal
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, i * pix_square_size),
                (self.window_size, i * pix_square_size),
                width=2,
            )
            # vertical
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (i * pix_square_size, 0),
                (i * pix_square_size, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # Blit to the visible window and update
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Keep framerate stable
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            # Return the RGB array (height, width, 3)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
