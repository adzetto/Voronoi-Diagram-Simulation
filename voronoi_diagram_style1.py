"""Author: Suleyman Celleoglu
Date: 2024-06-13.

This code generates random points, finds the closest points to a given point,
and plots the Voronoi diagram based on these points.
"""

import random

import matplotlib.pyplot as plt
import numpy as np


def generate_points(
    n: int,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
) -> list:
    """Generate random points in polar coordinates."""
    points = []
    i = 0
    while i < n:
        r = random.uniform(r_min, r_max)
        theta = random.uniform(theta_min, theta_max)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        points.append((x, y))
        i += 1
    return points


def find_closest_pt(points: list, p_0: tuple, exclude_pts: list) -> tuple:
    """Find the closest point to p_0 not in exclude_pts."""
    min_distance = float("inf")
    closest_point = None
    closest_idx = -1

    for i, pt in enumerate(points):
        if pt in exclude_pts:
            continue
        distance = ((pt[0] - p_0[0]) ** 2 + (pt[1] - p_0[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_point = pt
            closest_idx = i

    return closest_idx, closest_point


def unit_vec(p_from: tuple, p_to: tuple) -> list:
    """Calculate the unit vector from p_from to p_to."""
    vec = [p_to[0] - p_from[0], p_to[1] - p_from[1]]
    norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
    return [vec[0] / norm, vec[1] / norm] if norm != 0 else [0, 0]


def dot_product(vec1: list, vec2: list) -> float:
    """Calculate the dot product of two vectors."""
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def filter_pts(points: list, base_pt: tuple, ref_vec: list) -> list:
    """Filter points based on the reference vector from the base point."""
    remaining_pts = []
    for pt in points:
        if pt == base_pt:
            continue
        unit_v = unit_vec(base_pt, pt)
        if dot_product(ref_vec, unit_v) >= 0:
            remaining_pts.append(pt)
    return remaining_pts


def intersection(mid1: tuple, norm1: list, mid2: tuple, norm2: list) -> tuple:
    """Find the intersection point of two lines."""
    x1, y1 = mid1
    nx1, ny1 = norm1
    x2, y2 = mid2
    nx2, ny2 = norm2

    denominator = nx1 * ny2 - ny1 * nx2
    if denominator == 0:
        return None, None

    t = ((x2 - x1) * ny2 - (y2 - y1) * nx2) / denominator
    intersect_x = x1 + t * nx1
    intersect_y = y1 + t * ny1

    return intersect_x, intersect_y


def cyclic_sort(points: np.ndarray) -> np.ndarray:
    """Sort points in a cyclic order."""
    sorted_indices = np.argsort(np.arctan2(points[:, 1], points[:, 0]))
    return points[sorted_indices]


def plot_voronoi(
    points_copy: list,
    selected_points: list,
    midpoints: np.ndarray,
    intersections: np.ndarray,
) -> None:
    """Plot the Voronoi diagram."""
    p_0 = (0, 0)

    plt.figure(figsize=(10, 8))
    generated_x = [point[0] for point in points_copy]
    generated_y = [point[1] for point in points_copy]
    plt.scatter(generated_x, generated_y, color="gray", label="Generated Points")

    selected_x = [point[0] for point in selected_points]
    selected_y = [point[1] for point in selected_points]
    selected_x.append(p_0[0])
    selected_y.append(p_0[1])
    plt.scatter(selected_x, selected_y, color="green", label="Selected Points")

    plt.scatter(p_0[0], p_0[1], color="red", marker="v", label="P_0", s=100)

    midpoints_x = [point[0] for point in midpoints]
    midpoints_y = [point[1] for point in midpoints]
    plt.scatter(midpoints_x, midpoints_y, color="blue", label="Midpoints")

    if intersections.size > 0:
        plt.scatter(
            intersections[:, 0],
            intersections[:, 1],
            color="purple",
            label="Intersections",
        )
        for i in range(len(intersections)):
            next_i = (i + 1) % len(intersections)
            plt.plot(
                [intersections[i, 0], intersections[next_i, 0]],
                [intersections[i, 1], intersections[next_i, 1]],
                color="purple",
            )

    for i, midpoint in enumerate(midpoints):
        start_point = midpoint
        perpendicular_direction = np.array([-midpoints[i][1], midpoints[i][0]])
        end_point1 = midpoint + 2 * perpendicular_direction
        end_point2 = midpoint - 2 * perpendicular_direction
        plt.plot(
            [start_point[0], end_point1[0]],
            [start_point[1], end_point1[1]],
            "blue",
            linestyle="--",
        )
        plt.plot(
            [start_point[0], end_point2[0]],
            [start_point[1], end_point2[1]],
            "blue",
            linestyle="--",
        )
    zero_line = 0
    plt.axhline(zero_line, color="black", linewidth=2, linestyle=":")
    plt.axvline(zero_line, color="black", linewidth=2, linestyle=":")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("Voronoi Cell and Points")
    plt.legend()
    plt.show()


def main() -> None:
    """Generate points and plot the Voronoi diagram."""
    p_0 = (0, 0)

    points = []
    quadrants = [
        (5, 2.5, 15, 5, 85),
        (5, 2.5, 15, 95, 175),
        (5, 2.5, 15, 185, 265),
        (5, 2.5, 15, 275, 355),
    ]
    for n, r_min, r_max, theta_min, theta_max in quadrants:
        points.extend(generate_points(n, r_min, r_max, theta_min, theta_max))

    points_copy = points.copy()
    selected_points = []

    while points:
        closest_idx, closest_point = find_closest_pt(points, p_0, selected_points)
        selected_points.append(closest_point)
        ref_vec = unit_vec(closest_point, p_0)
        points = filter_pts(points, closest_point, ref_vec)

    midpoints = []
    normals = []

    for pt in selected_points:
        midpoint = (np.array(pt) + np.array(p_0)) / 2
        normal = unit_vec(p_0, pt)
        midpoints.append(midpoint)
        normals.append(normal)

    midpoints = np.array(midpoints)
    normals = np.array(normals)

    sorted_indices = np.argsort(np.arctan2(midpoints[:, 1], midpoints[:, 0]))
    midpoints, normals = midpoints[sorted_indices], normals[sorted_indices]

    intersections = []
    perpendicular_directions = []

    for i, normal in enumerate(normals):
        perpendicular_direction = np.array([-normal[1], normal[0]])
        perpendicular_directions.append(perpendicular_direction)

    for i in range(len(midpoints)):
        next_i = (i + 1) % len(midpoints)
        intersect = intersection(
            tuple(midpoints[i]),
            list(perpendicular_directions[i]),
            tuple(midpoints[next_i]),
            list(perpendicular_directions[next_i]),
        )
        if intersect is not None:
            intersections.append(intersect)

    intersections = np.array(intersections)
    if intersections.size > 0:
        intersections = cyclic_sort(intersections)

    plot_voronoi(points_copy, selected_points, midpoints, intersections)


main()