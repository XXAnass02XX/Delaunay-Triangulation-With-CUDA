#!/usr/bin/env python3
import sys

def generate_grid_points(N=50):
    """Generate N x N uniformly spaced points in [0,1]x[0,1]."""
    points = []
    for i in range(N):
        for j in range(N):
            # Evenly space points from 0 to 1.
            x = i / (N - 1)
            y = j / (N - 1)
            points.append((x, y))
    return points

def main():
    N = 50  # Change this value for a different grid size.
    points = generate_grid_points(N)
    with open("points.txt", "w") as f:
        for x, y in points:
            f.write(f"{x} {y}\n")
    print(f"Generated {N*N} grid points and saved them to points.txt")

if __name__ == "__main__":
    main()
