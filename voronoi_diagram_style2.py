import random

import matplotlib.pyplot as plt
import numpy as np


def gen_pts(n, r_min, r_max, t_min, t_max):
    pts = []
    for _ in range(n):
        r = random.uniform(r_min, r_max)
        t = random.uniform(t_min, t_max)
        x = r * np.cos(np.radians(t))
        y = r * np.sin(np.radians(t))
        dist = np.sqrt(x**2 + y**2)
        pts.append((x, y, dist))
    return pts


def nrm(v):
    return np.sqrt(np.sum(np.square(v)))


def cust_sort(arr):
    return sorted(range(len(arr)), key=lambda x: arr[x])


def find_u_vec(P_from, P_to):
    diff = np.array(P_to[:2]) - np.array(P_from[:2])
    return diff / nrm(diff) if nrm(diff) != 0 else np.array([0, 0])


def norm_vec(v):
    return [
        x / (sum(x**2 for x in v)) ** 0.5 if (sum(x**2 for x in v)) ** 0.5 else 0
        for x in v
    ]


def dot_prod(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def filter_pts_by_dp(pts, base, ref_v):
    return [
        p
        for p in pts
        if p[:2] != base[:2]
        and dot_prod(norm_vec(ref_v), norm_vec((p[0] - base[0], p[1] - base[1]))) >= 0
    ]


def find_closest(pts, P0, closest):
    pts_arr = np.array([p[:2] for p in pts])
    dists = np.linalg.norm(pts_arr - P0, axis=1)
    dists[[pts.index(p) for p in closest if p in pts]] = float("inf")
    cl_idx = np.argmin(dists)
    return cl_idx, pts[cl_idx]


def calc_inter(mid1, norm1, mid2, norm2):
    return (
        (
            mid1[0]
            + ((mid2[0] - mid1[0]) * norm2[1] - (mid2[1] - mid1[1]) * norm2[0])
            / (norm1[0] * norm2[1] - norm1[1] * norm2[0])
            * norm1[0],
            mid1[1]
            + ((mid2[0] - mid1[0]) * norm2[1] - (mid2[1] - mid1[1]) * norm2[0])
            / (norm1[0] * norm2[1] - norm1[1] * norm2[0])
            * norm1[1],
        )
        if norm1[0] * norm2[1] - norm1[1] * norm2[0] != 0
        else None
    )


def sort_pts_cyclic(pts):
    return pts[np.argsort(np.arctan2(*(pts - np.mean(pts, axis=0)).T))]


def cust_roll(arr, shift):
    return arr[-shift:].tolist() + arr[:-shift].tolist()


def cust_hstack(arr1, arr2):
    return [item for sublist in zip(arr1, arr2) for item in sublist]


def cust_reshape(arr, shape):
    return [arr[i : i + shape[1]] for i in range(0, len(arr), shape[1])]


def main() -> None:
    P0 = (0, 0)
    pts_first = (
        gen_pts(n=5, r_min=2.5, r_max=15, t_min=5, t_max=85)
        + gen_pts(n=5, r_min=2.5, r_max=15, t_min=95, t_max=175)
        + gen_pts(n=5, r_min=2.5, r_max=15, t_min=185, t_max=265)
        + gen_pts(n=5, r_min=2.5, r_max=15, t_min=275, t_max=355)
    )

    pts = pts_first.copy()
    closest = []

    while pts:
        cl_idx, cl_pt = find_closest(pts, P0, closest)
        closest.append(cl_pt)
        ref_v = find_u_vec(P_from=cl_pt, P_to=P0)
        pts = filter_pts_by_dp(pts, cl_pt, ref_v)

    mids = np.array([(np.array(P[:2]) + P0) / 2 for P in closest])
    vecs = np.array([np.array(P[:2]) - np.array(P0) for P in closest])
    norms = np.array([-vecs[:, 1], vecs[:, 0]]).T
    unit_norms = np.array([norm_vec(n) for n in norms])
    angles = np.arctan2(mids[:, 1], mids[:, 0])
    sorted_idx = np.argsort(angles)
    mids = mids[sorted_idx]
    unit_norms = unit_norms[sorted_idx]

    inter_pts = [
        calc_inter(
            mids[i],
            unit_norms[i],
            mids[(i + 1) % len(mids)],
            unit_norms[(i + 1) % len(mids)],
        )
        for i in range(len(mids))
    ]

    inter_pts = [pt for pt in inter_pts if pt is not None]

    plt.figure(figsize=(10, 8))

    plt.scatter(
        [p[0] for p in pts_first],
        [p[1] for p in pts_first],
        color="gray",
        label="First Generated",
    )

    all_pts = [*closest, P0]
    x_coords = [p[0] for p in all_pts]
    y_coords = [p[1] for p in all_pts]
    plt.scatter(x_coords, y_coords, color="green", label="Selected Points")

    for p in closest:
        plt.plot([P0[0], p[0]], [P0[1], p[1]], "gray", linestyle="dotted")
    plt.scatter(
        [p[0] for p in mids],
        [p[1] for p in mids],
        color="blue",
        label="Midpoints",
    )
    inter_pts = np.array(inter_pts)

    if inter_pts.size > 0:
        inter_pts = np.array(inter_pts)
        plt.scatter(
            inter_pts[:, 0],
            inter_pts[:, 1],
            color="purple",
            label="Intersections",
        )

        rolled_pts = cust_roll(inter_pts, -1)
        stacked_pts = cust_hstack(inter_pts, rolled_pts)
        reshaped_pts = cust_reshape(stacked_pts, (-1, 2, 2))

        lines = np.array(reshaped_pts)
        plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, color="purple")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Voronoi")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
