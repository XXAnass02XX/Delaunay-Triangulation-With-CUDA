#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sys

def load_triangles(filename):
    triangles = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    curr = []
    for line in lines:
        line = line.strip()
        if not line:
            if len(curr) == 3:
                triangles.append(curr)
            curr = []
        else:
            parts = line.split()
            if len(parts) == 2:
                x, y = map(float, parts)
                curr.append((x, y))
    if len(curr) == 3:
        triangles.append(curr)
    return triangles

def plot_triangles(triangles, out_image="triangulation.png"):
    plt.figure(figsize=(8,8))
    for tri in triangles:
        xs = [p[0] for p in tri] + [tri[0][0]]
        ys = [p[1] for p in tri] + [tri[0][1]]
        plt.plot(xs, ys, 'b-')
    plt.title("Delaunay Triangulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal')
    plt.savefig(out_image)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 visualizer.py output_triangles.txt")
        sys.exit(1)
    triangles = load_triangles(sys.argv[1])
    plot_triangles(triangles)
