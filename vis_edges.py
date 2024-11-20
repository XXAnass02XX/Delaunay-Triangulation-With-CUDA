import matplotlib.pyplot as plt

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

# Read edges from the file
file_path = 'poly_added1.txt'
edges = read_edges(file_path)

# Plot the edges
plt.figure(figsize=(6, 6))
for edge in edges:
    x_coords = [edge[0][0], edge[1][0]]
    y_coords = [edge[0][1], edge[1][1]]
    plt.plot(x_coords, y_coords, marker='o')  # Plot each edge with markers at the endpoints

# Set plot properties
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Edges Plot')
plt.grid(True)
plt.axis('equal')  # Equal scaling for x and y axes
plt.show()
