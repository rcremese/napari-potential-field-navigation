import numpy as np
import heapq
from itertools import product
from typing import Tuple, Union, List


def heuristic(a: Tuple[int], b: Tuple[int]) -> float:
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(b - a, ord=1)


def astar(
    binary_map: np.ndarray, start: Tuple[int], goal: Tuple[int]
) -> Union[bool, List[Tuple[int]]]:
    """A* algorithm for path planning in a binary map.

    Args:
        binary_map (np.ndarray): Binary map of the environment where 0 is free space and 1 is occupied space.
        start (Tuple[int]): Starting position as index in the map.
        goal (Tuple[int]): Goal position as index in the map.

    Returns:
        Union[bool, List[Tuple[int]]]: List of indices from start to goal if a path is found, False otherwise.
    """
    assert (
        binary_map.ndim == len(start) == len(goal)
    ), "Dimensions of the map, start and goal must be the same."
    ## Generate all possible l2 directions
    # directions = [
    #     np.array(i)
    #     for i in product([-1, 0, 1], repeat=binary_map.ndim)
    #     if i != tuple([0] * binary_map.ndim)
    # ]

    ## Generate all possible manhatan directions
    directions = np.array(
        [
            np.eye(binary_map.ndim, dtype=int)[i] * j
            for i in range(binary_map.ndim)
            for j in [-1, 1]
        ]
    )
    ## Initialize the open set, close set, came from, gscore, fscore and the heap
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for direction in directions:
            neighbor = tuple(np.array(current) + direction)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if np.any(np.array(neighbor) < 0) or np.any(
                np.array(neighbor) >= np.array(binary_map.shape)
            ):
                continue
            if binary_map[neighbor] == 1:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(
                neighbor, 0
            ):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in oheap
            ]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(
                    neighbor, goal
                )
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False


def wavefront_generation(
    binary_map: np.ndarray, goal: Tuple[int]
) -> np.ma.masked_array:
    # Generate all possible directions, including diagonals
    directions = [
        np.array(i)
        for i in product([-1, 0, 1], repeat=binary_map.ndim)
        if i != tuple([0] * binary_map.ndim)
    ]

    # Initialize the cost map with infinity
    cost_map = np.ma.masked_array(
        np.full(binary_map.shape, np.inf, dtype=np.float32),
        mask=binary_map,
    )
    cost_map[goal] = 0

    # Initialize the frontier with the goal
    frontier = [goal]

    while frontier:
        current = frontier.pop(0)
        for direction in directions:
            neighbor = tuple(np.array(current) + direction)
            if np.any(np.array(neighbor) < 0) or np.any(
                np.array(neighbor) >= np.array(binary_map.shape)
            ):
                continue
            if cost_map.mask[neighbor]:
                continue
            if cost_map[neighbor] > cost_map[current] + 1:
                cost_map[neighbor] = cost_map[current] + 1
                frontier.append(neighbor)
    return cost_map


def wavefront_planner(
    cost_map: np.ndarray, start: Tuple[int], goal: Tuple[int]
):
    # Generate all possible directions, including diagonals
    directions = [
        np.array(i)
        for i in product([-1, 0, 1], repeat=binary_map.ndim)
        if i != tuple([0] * binary_map.ndim)
    ]
    # Backtrack from the start to the goal

    path = []
    current = start
    while current != goal:
        path.append(current)
        neighbors = [
            tuple(np.array(current) + direction) for direction in directions
        ]
        current = min(
            neighbors,
            key=lambda x: (
                cost_map[x]
                if 0 <= x[0] < binary_map.shape[0]
                and 0 <= x[1] < binary_map.shape[1]
                and 0 <= x[2] < binary_map.shape[2]
                else np.inf
            ),
        )
    path.append(goal)

    return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define the 3D binary map
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    # binary_map = np.zeros_like(binary_map)

    # Define the start and end points
    start = (0, 0, 0)
    end = (2, 2, 2)

    # Find the path
    cost_map = wavefront_generation(binary_map, end)

    path = wavefront_planner(cost_map, start, end)
    print(path)

    path.append(start)

    # Plot the path
    path = np.array(path) + 0.5
    print(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(binary_map, facecolors="gray", edgecolor="k", alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "r")
    plt.show()
