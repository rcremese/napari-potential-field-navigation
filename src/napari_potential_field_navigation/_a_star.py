import numpy as np
import scipy.sparse.linalg as splinalg
import heapq
from itertools import product
from typing import Tuple, Union, List
import logging

from napari_potential_field_navigation._finite_difference import (
    create_poisson_system,
)


def heuristic(a: Tuple[int], b: Tuple[int]) -> float:
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(b - a, ord=1)


def astar(
    binary_map: np.ndarray,
    start: Tuple[int],
    goal: Tuple[int],
    distance: str = "l1",
) -> Union[bool, List[Tuple[int]]]:
    """A* algorithm for path planning in a binary map.

    Args:
        binary_map (np.ndarray): Binary map of the environment where 1 is free space and 0 is occupied space.
        start (Tuple[int]): Starting position as index in the map.
        goal (Tuple[int]): Goal position as index in the map.
        distance (str): Distance metric to use for the heuristic. Defaults to "l1".

    Returns:
        Union[bool, List[Tuple[int]]]: List of indices from start to goal if a path is found, False otherwise.
    """
    assert (
        binary_map.ndim == len(start) == len(goal)
    ), "Dimensions of the map, start and goal must be the same."
    assert distance in ["l1", "l2"], "Distance metric must be l1 or l2."
    ## Generate all possible manhatan directions
    if distance == "l1":
        directions = np.array(
            [
                np.eye(binary_map.ndim, dtype=int)[i] * j
                for i in range(binary_map.ndim)
                for j in [-1, 1]
            ]
        )
    ## Generate all possible l2 directions
    elif distance == "l2":
        directions = [
            np.array(i)
            for i in product([-1, 0, 1], repeat=binary_map.ndim)
            if i != tuple([0] * binary_map.ndim)
        ]

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
            if binary_map[neighbor] == 0:
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


def a_starfield(
    binary_map: np.ndarray,
    start: Tuple[int],
    goal: Tuple[int],
    distance: str = "l1",
) -> np.ma.masked_array:
    """Use the A* algorithm to find the shortest path in a binary map and create a distance map based on the solution.

    Args:
        binary_map (np.ndarray): Binary map of the environment where 1 is free space and 0 is occupied space.
        start (Tuple[int]): Starting position as index in the map.
        goal (Tuple[int]): Goal position as index in the map.
        distance (str): Distance metric to use for the heuristic. Defaults to "l1".
    Returns:
        np.ma.masked_array: Distance map to the goal as a masked array where masked values are obstacles.
    """
    logging.info("Start A* algorithm")
    path = astar(~binary_map, start, goal, distance)
    if path is False:
        logging.error("No path found")
        return False
    logging.info("Initial path found. Start cost map generation.")
    cost_map = np.ma.masked_array(
        np.full(binary_map.shape, 0, dtype=np.float32),
        mask=~binary_map,
    )
    ## Set the values of the path as the distance to the goal
    path.append(start)
    for i, p in enumerate(path):
        cost_map[p] = len(path) - i
    ## Create laplace matrix and the bc vector to solve the poisson equation
    laplace_mat, rhs = create_poisson_system(cost_map, spacing=(1, 1, 1))
    ## Solve the system on a subset of the map
    logging.info("Start solving the poisson equation")
    valid_indices = binary_map.flat != 0
    A = laplace_mat[valid_indices, :][:, valid_indices]
    b = rhs[valid_indices]
    x, info = splinalg.cg(A, b)
    logging.info(f"CG convergence info: {info}")
    if info != 0:
        logging.error("CG did not converge")
        return False
    ## Set the values of the solution to the cost map
    cost_map.flat[valid_indices] = x
    return cost_map


def wavefront_generation(
    binary_map: np.ndarray, goal: Tuple[int]
) -> np.ma.masked_array:
    """Generate a distance map to the goal.

    Args:
        binary_map (np.ndarray): Binary map of the environment where 1 is free space and 0 is occupied space.
        goal (Tuple[int]): _description_

    Returns:
        np.ma.masked_array: _description_
    """
    # Generate all possible directions, including diagonals
    directions = [
        np.array(i)
        for i in product([-1, 0, 1], repeat=binary_map.ndim)
        if i != tuple([0] * binary_map.ndim)
    ]

    # Initialize the cost map with infinity
    cost_map = np.ma.masked_array(
        np.full(binary_map.shape, np.inf, dtype=np.float32),
        mask=~binary_map,
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


def uneven_wavefront_generation(
    binary_map: np.ndarray,
    goals: List[Tuple[int]],
    spacing: Tuple[float] = None,
) -> np.ma.masked_array:
    """Generate a distance map to the goals with uneven spacing map.

    Args:
        binary_map (np.ndarray): binary map of the environment where 0 is free space and 1 is occupied space.
        goals (List[Tuple[int]]): List of goals as indices in the map.
        spacing (Tuple[float]): Soacing between the cells in the map. Defaults to None.

    Returns:
        np.ma.masked_array: Cost map to the goals as a masked array where masked values are obstacles.
    """
    if spacing is None:
        spacing = np.ones(binary_map.ndim)
    else:
        assert (
            len(spacing) == binary_map.ndim
        ), "Spacing must have the same dimension as the map."
        spacing = np.array(spacing)
    # Generate all possible directions, including diagonals
    directions = [
        np.array(i)
        for i in product([-1, 0, 1], repeat=binary_map.ndim)
        if i != tuple([0] * binary_map.ndim)
    ]

    # Initialize the cost map with infinity
    cost_map = np.ma.masked_array(
        np.full(binary_map.shape, np.inf), mask=binary_map
    )

    # Initialize the frontier with the goals
    frontier = goals.copy()

    for goal in goals:
        cost_map[goal] = 0

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
            cost = np.sum(np.abs(np.array(direction) * np.array(spacing)))
            if cost_map[neighbor] > cost_map[current] + cost:
                cost_map[neighbor] = cost_map[current] + cost
                frontier.append(neighbor)
    return cost_map


def wavefront_planner(
    cost_map: np.ndarray, start: Tuple[int], goal: Tuple[int]
):
    # Generate all possible directions, including diagonals
    directions = [
        np.array(i)
        for i in product([-1, 0, 1], repeat=cost_map.ndim)
        if i != tuple([0] * cost_map.ndim)
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
                if 0 <= x[0] < cost_map.shape[0]
                and 0 <= x[1] < cost_map.shape[1]
                and 0 <= x[2] < cost_map.shape[2]
                else np.inf
            ),
        )
    path.append(goal)

    return path


def test_astar():
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    start = (0, 0, 0)
    end = (2, 2, 2)
    path = astar(binary_map, start, end)
    assert path == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


def test_wavefront_generation():
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    end = (2, 2, 2)
    cost_map = wavefront_generation(binary_map, end)
    assert cost_map[end] == 0
    assert cost_map[0, 0, 0] == 4
    assert cost_map[1, 1, 1] == 3


def test_wavefront_planner():
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    start = (0, 0, 0)
    end = (2, 2, 2)
    cost_map = wavefront_generation(binary_map, end)
    path = wavefront_planner(cost_map, start, end)
    assert path == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


def test_uneven_wavefront_generation():
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    goals = [(0, 0, 0), (2, 2, 2)]
    cost_map = uneven_wavefront_generation(
        binary_map, goals, spacing=(1, 1, 1)
    )
    assert cost_map[0, 0, 0] == 0
    assert cost_map[2, 2, 2] == 4
    assert cost_map[1, 1, 1] == 3


def test_uneven_wavefront_planner():
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    goals = [(0, 0, 0), (2, 2, 2)]
    cost_map = uneven_wavefront_generation(
        binary_map, goals, spacing=(1, 1, 1)
    )
    path = wavefront_planner(cost_map, goals[0], goals[1])
    assert path == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test_astar()
    # test_wavefront_generation()
    # test_wavefront_planner()
    # test_uneven_wavefront_generation()
    # test_uneven_wavefront_planner()
    binary_map = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )

    # Convert the binary map to a masked array
    masked_map = np.ma.masked_equal(binary_map, 1)

    # Define the start and end points
    start = (0, 0, 0)
    goals = [(2, 2, 2), (1, 1, 1)]

    # Define the spacing between dimensions
    spacing = [1, 2, 3]

    # Find the path
    print("Start computation")
    cost_map = uneven_wavefront_generation(masked_map, goals, spacing)
    print("Cost map generated")
    # path = wavefront_planner(masked_map, start, goals[0])
    path = astar(~binary_map, start, goals[0])
    print("Path found")
    print(path)
    ## Plot the path
    path = np.array(path) + 0.5
    print(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(binary_map, facecolors="gray", edgecolor="k", alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "r")
    plt.show()
