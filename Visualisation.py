import matplotlib.pyplot as plt
import numpy as np
import sys
def load_points_and_triangles(filename):
    points = []
    triangles = []
    try:
        with open(filename, 'r') as file:
            current_triangle = []
            for line in file:
                line = line.strip()
                if line == "":
                    if len(current_triangle) == 3:
                        indices = []
                        for point in current_triangle:
                            if point not in points:
                                points.append(point)
                            indices.append(points.index(point))
                        triangles.append(indices)
                    current_triangle = []
                else:
                    try:
                        x, y = map(float, line.split())
                        current_triangle.append((x, y))
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
            # Handle last triangle if no blank line at the end
            if len(current_triangle) == 3:
                indices = []
                for point in current_triangle:
                    if point not in points:
                        points.append(point)
                    indices.append(points.index(point))
                triangles.append(indices)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"Error: {e}")
    return np.array(points), np.array(triangles)

def plot_triangles(points, triangles,image_name):
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], triangles, 'bo-', markersize=5, linewidth=1)
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Delaunay Triangles')
    plt.savefig(image_name)
    plt.show()
    plt.close()
    

# File path to your data
# Replace with your actual file name
n = len(sys.argv)
if n > 1:
    points, triangles = load_points_and_triangles("seq_triangles_5.txt")
    plot_triangles(points, triangles,"seq_triangles.png")  
else :
    for i in range(2,5):
        points, triangles = load_points_and_triangles(f"triangles_{i}.txt")
        plot_triangles(points, triangles,f"triangles_{i}.png")  