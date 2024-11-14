import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def load_triangles(filename):
    triangles = []
    try:
        with open(filename, 'r') as file:
            points = []
            for line in file:
                if line.strip() == "":  # Blank line indicates end of a triangle
                    if len(points) == 3:
                        triangles.append(points)
                    points = []
                else:
                    try:
                        x, y = map(float, line.split())
                        points.append((x, y))
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")  # Handle invalid lines
            # Handle case if the last triangle is not added
            if len(points) == 3:
                triangles.append(points)
        if len(triangles) == 0:
            print("No triangles found in the file.")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"Error: {e}")
    return triangles

def plot_triangles(triangles):
    if not triangles:
        print("No triangles to plot.")
        return
    
    fig, ax = plt.subplots()
    for tri in triangles:
        polygon = Polygon(tri, edgecolor='blue', fill=False)
        ax.add_patch(polygon)
        for point in tri:
            ax.plot(point[0], point[1], 'ro')  # Plot the vertices

    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Delaunay Triangulation')
    plt.show()

# Load and plot the triangles
triangles = load_triangles("triangles.txt")
plot_triangles(triangles)
