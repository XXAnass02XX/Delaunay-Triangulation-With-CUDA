#ifndef DELAUNAYTRIANGULATION_CUH
#define DELAUNAYTRIANGULATION_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include "Point.cuh"
#include "Triangle.cuh"

// --------------------------------------------------------------------------
// Kernel that loads triangles (from unified memory) into shared memory and
// tests if point p is inside each triangle's circumcircle.
// The result is written into a unified boolean array.
__global__ void checkCircumcircleUnified(Triangle* triangles, bool* isBadTriangle, const Point p, int triangleCount) {
    extern __shared__ Triangle s_triangles[];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x;
    
    if (globalIdx < triangleCount) {
        s_triangles[localIdx] = triangles[globalIdx];
    }
    __syncthreads();
    
    if (globalIdx < triangleCount) {
        isBadTriangle[globalIdx] = s_triangles[localIdx].isInCircumcircle(p);
    }
}

// --------------------------------------------------------------------------
// The DelaunayTriangulation class now holds its triangles in unified memory.
// We pre-allocate a fixed buffer for triangles and flags. No host/device
// copies are performed during point insertion.
class DelaunayTriangulation {
public:
    Triangle* d_triangles; // Unified memory array for triangles.
    bool* d_isBadTriangle; // Unified memory array for flags.
    int triangleCount;     // Current number of triangles.
    int maxTriangles;      // Maximum triangles allocated.

    // The constructor pre-allocates a large unified memory buffer.
    DelaunayTriangulation(int maxTri = 10000) : triangleCount(0), maxTriangles(maxTri) {
        cudaMallocManaged(&d_triangles, maxTriangles * sizeof(Triangle));
        cudaMallocManaged(&d_isBadTriangle, maxTriangles * sizeof(bool));
    }
    
    ~DelaunayTriangulation() {
        cudaFree(d_triangles);
        cudaFree(d_isBadTriangle);
    }
    
    // Create a super-triangle that encloses all points.
    void initializeWithSuperTriangle() {
        Point p1(-1.5, -1.5);
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);
        d_triangles[0] = Triangle(p1, p2, p3);
        triangleCount = 1;
    }
    
    // Remove triangles that use any super-triangle vertex.
    void removeTrianglesWithSuperVertices() {
        Point p1(-1.5, -1.5);
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);
        int newCount = 0;
        for (int i = 0; i < triangleCount; i++) {
            Triangle& tri = d_triangles[i];
            if ((tri.a == p1 || tri.a == p2 || tri.a == p3) ||
                (tri.b == p1 || tri.b == p2 || tri.b == p3) ||
                (tri.c == p1 || tri.c == p2 || tri.c == p3)) {
                continue;
            }
            d_triangles[newCount++] = tri;
        }
        triangleCount = newCount;
    }
    
    // Incrementally add a new point to the triangulation.
    // A kernel (using shared memory) marks triangles whose circumcircles
    // contain the new point. Then the CPU (accessing unified memory)
    // compacts the triangle array and adds new triangles.
    void addPoint(const Point& p) {
        // Launch kernel to mark triangles whose circumcircle contains p.
        int threadsPerBlock = 256;
        int blocks = (triangleCount + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedMemSize = threadsPerBlock * sizeof(Triangle);
        checkCircumcircleUnified<<<blocks, threadsPerBlock, sharedMemSize>>>(d_triangles, d_isBadTriangle, p, triangleCount);
        cudaDeviceSynchronize();
        
        // Build a list of bad triangles and compute the polygon boundary.
        std::vector<Triangle> badTriangles;
        std::vector<std::pair<Point, Point>> polygonEdges;
        for (int i = 0; i < triangleCount; i++) {
            if (d_isBadTriangle[i]) {
                badTriangles.push_back(d_triangles[i]);
                addEdgeIfUnique(polygonEdges, { d_triangles[i].a, d_triangles[i].b });
                addEdgeIfUnique(polygonEdges, { d_triangles[i].b, d_triangles[i].c });
                addEdgeIfUnique(polygonEdges, { d_triangles[i].c, d_triangles[i].a });
            }
        }
        
        // Compact the triangle array by removing bad triangles.
        int newCount = 0;
        for (int i = 0; i < triangleCount; i++) {
            if (!d_isBadTriangle[i]) {
                d_triangles[newCount++] = d_triangles[i];
            }
        }
        triangleCount = newCount;
        
        // Create new triangles from each polygon edge and the new point.
        for (const auto& edge : polygonEdges) {
            if (triangleCount < maxTriangles) {
                d_triangles[triangleCount++] = Triangle(edge.first, edge.second, p);
            } else {
                std::cerr << "Error: Exceeded maximum triangle capacity!" << std::endl;
            }
        }
    }
    
    // Add an edge to the polygon if it is unique.
    // (If an edge appears twice in opposite order, it is removed.)
    void addEdgeIfUnique(std::vector<std::pair<Point, Point>>& edges, const std::pair<Point, Point>& edge) {
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            if ((it->first == edge.second && it->second == edge.first) ||
                (it->first == edge.first && it->second == edge.second)) {
                edges.erase(it);
                return;
            }
        }
        edges.push_back(edge);
    }
    
    // Export the final triangulation to a file.
    void exportTriangles(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing!" << std::endl;
            return;
        }
        for (int i = 0; i < triangleCount; i++) {
            Triangle& tri = d_triangles[i];
            file << tri.a.x << " " << tri.a.y << "\n";
            file << tri.b.x << " " << tri.b.y << "\n";
            file << tri.c.x << " " << tri.c.y << "\n";
            file << "\n";
        }
        file.close();
    }
};

#endif
