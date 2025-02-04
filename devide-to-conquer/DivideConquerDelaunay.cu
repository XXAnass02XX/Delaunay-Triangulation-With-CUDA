// GridTriangulation.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

struct Point {
    double x, y;
};

struct Triangle {
    Point a, b, c;
};

// Each cell in the grid (of size (N-1) x (N-1)) is processed by one thread.
__global__ void gridTriangulationKernel(const Point* points, int N, Triangle* triangles) {
    // i: row index (0 .. N-2), j: column index (0 .. N-2)
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row index in cell grid
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col index in cell grid

    if (i < (N - 1) && j < (N - 1)) {
        int cellIndex = i * (N - 1) + j;
        // For the cell at grid (i,j), the four corner indices in the point array (row-major):
        // top-left:   i * N + j
        // top-right:  i * N + (j+1)
        // bottom-left: (i+1) * N + j
        // bottom-right: (i+1) * N + (j+1)
        int idx_top_left    = i * N + j;
        int idx_top_right   = i * N + (j + 1);
        int idx_bottom_left = (i + 1) * N + j;
        int idx_bottom_right= (i + 1) * N + (j + 1);
        
        // Retrieve the four corner points.
        Point top_left    = points[idx_top_left];
        Point top_right   = points[idx_top_right];
        Point bottom_left = points[idx_bottom_left];
        Point bottom_right= points[idx_bottom_right];
        
        // Choose a consistent diagonal.
        // For example, split along the diagonal from top-left to bottom-right.
        // First triangle: (top_left, bottom_left, bottom_right)
        // Second triangle: (top_left, bottom_right, top_right)
        int triIndex = cellIndex * 2;
        triangles[triIndex].a     = top_left;
        triangles[triIndex].b     = bottom_left;
        triangles[triIndex].c     = bottom_right;
        
        triangles[triIndex + 1].a = top_left;
        triangles[triIndex + 1].b = bottom_right;
        triangles[triIndex + 1].c = top_right;
    }
}

int main() {
    // Read points from file "points.txt"
    std::ifstream infile("points.txt");
    if (!infile) {
        std::cerr << "Error: Could not open points.txt" << std::endl;
        return 1;
    }
    std::vector<Point> h_points;
    double x, y;
    while (infile >> x >> y) {
        h_points.push_back({x, y});
    }
    infile.close();

    int numPoints = h_points.size();
    // Deduce grid dimension: assume a square grid.
    int N = static_cast<int>(std::round(std::sqrt(numPoints)));
    if (N * N != numPoints) {
        std::cerr << "Error: Number of points (" << numPoints << ") is not a perfect square." << std::endl;
        return 1;
    }
    std::cout << "Read " << numPoints << " points, grid dimension " << N << "x" << N << std::endl;

    // Number of cells and triangles.
    int numCells = (N - 1) * (N - 1);
    int numTriangles = numCells * 2;

    // Allocate device memory.
    Point* d_points;
    Triangle* d_triangles;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));

    // Copy points to device.
    cudaMemcpy(d_points, h_points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    // Launch the kernel over the cell grid.
    dim3 blockDim(16, 16);
    dim3 gridDim((N - 1 + blockDim.x - 1) / blockDim.x, (N - 1 + blockDim.y - 1) / blockDim.y);
    gridTriangulationKernel<<<gridDim, blockDim>>>(d_points, N, d_triangles);
    cudaDeviceSynchronize();

    // Copy the generated triangles back to host.
    std::vector<Triangle> h_triangles(numTriangles);
    cudaMemcpy(h_triangles.data(), d_triangles, numTriangles * sizeof(Triangle), cudaMemcpyDeviceToHost);

    // Write triangles to file "triangles.txt"
    std::ofstream outfile("triangles.txt");
    if (!outfile) {
        std::cerr << "Error: Could not open triangles.txt for writing." << std::endl;
        return 1;
    }
    for (int i = 0; i < numTriangles; i++) {
        Triangle &tri = h_triangles[i];
        outfile << tri.a.x << " " << tri.a.y << "\n";
        outfile << tri.b.x << " " << tri.b.y << "\n";
        outfile << tri.c.x << " " << tri.c.y << "\n\n";
    }
    outfile.close();

    std::cout << "Triangulation written to triangles.txt with " << numTriangles << " triangles." << std::endl;

    // Clean up.
    cudaFree(d_points);
    cudaFree(d_triangles);

    return 0;
}