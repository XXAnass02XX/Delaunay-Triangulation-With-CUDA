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

# Function to read edges from a file
def read_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_edge = []
        for line in lines:
            line = line.strip()
            if line:  # If line is not empty
                x, y = map(float, line.split())
                current_edge.append((x, y))
                if len(current_edge) == 2:  # Once two points are read, it's a complete edge
                    edges.append(current_edge)
                    current_edge = []
    return edges

def plot_triangles(points, triangles,image_name, file_path):
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], triangles, 'bo-', markersize=5, linewidth=1)
    plt.gca().set_aspect('equal')
    # Read edges from the file
    edges = read_edges(file_path)

    # Plot the edges
    # plt.figure(figsize=(6, 6))
    for edge in edges:
        x_coords = [edge[0][0], edge[1][0]]
        y_coords = [edge[0][1], edge[1][1]]
        plt.plot(x_coords, y_coords,color='red', marker='o')
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
    plot_triangles(points, triangles,"seq_triangles.png", "poly_added4.txt")  
else :
    for i in range(5):
        points, triangles = load_points_and_triangles(f"triangles_{i}.txt")
        plot_triangles(points, triangles,f"triangles_{i}.png", f"poly_added{i}.txt")  