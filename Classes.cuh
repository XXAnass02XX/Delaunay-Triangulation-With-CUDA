#ifndef CLASSES_CUH
#define CLASSES_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 
#include <fstream>  
struct Point {
    double x, y;

    __host__ __device__ Point(double x = 0, double y = 0) : x(x), y(y) {}

    __host__ __device__ bool operator==(const Point& other) const {
        return (x == other.x && y == other.y);
    }
};

struct Triangle {
    Point a, b, c;

    __host__ __device__ Triangle(const Point& a, const Point& b, const Point& c) : a(a), b(b), c(c) {}

    __host__ __device__ bool isInCircumcircle(const Point& p) const {
        double ax = a.x - p.x;
        double ay = a.y - p.y;
        double bx = b.x - p.x;
        double by = b.y - p.y;
        double cx = c.x - p.x;
        double cy = c.y - p.y;

        double det = (ax * ax + ay * ay) * (bx * cy - by * cx) -
                     (bx * bx + by * by) * (ax * cy - ay * cx) +
                     (cx * cx + cy * cy) * (ax * by - ay * bx);

        return det > 0;
    }

    __host__ __device__ bool operator==(const Triangle& other) const {
        return (a == other.a && b == other.b && c == other.c);
    }
};

__global__ void checkCircumcircle(Triangle* triangles, bool* isBadTriangle, const Point p, int numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTriangles) {
        isBadTriangle[idx] = triangles[idx].isInCircumcircle(p);// TODO wrap divergence
    }
}

class DelaunayTriangulation {
public:
    std::vector<Triangle> triangles;

    DelaunayTriangulation() {}
    DelaunayTriangulation(const DelaunayTriangulation& other) {
        triangles = other.triangles;
    }
    void initializeWithSuperTriangle() {
        Point p1(-1.5, -1.5);// TODO 1000
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);
        triangles.push_back(Triangle(p1, p2, p3));
    }
    void removeTrianglesWithSuperVertices() {
        Point p1(-1.5, -1.5); // TODO make an attribute fot these points in this class
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);

        auto isSuperVertexTriangle = [&](const Triangle& tri) {
            return (tri.a == p1 || tri.a == p2 || tri.a == p3 ||
                    tri.b == p1 || tri.b == p2 || tri.b == p3 ||
                    tri.c == p1 || tri.c == p2 || tri.c == p3);
        };

        triangles.erase(std::remove_if(triangles.begin(), triangles.end(), isSuperVertexTriangle), triangles.end());
    }

    void addPoint(const Point& p, int i) {
        std::vector<Triangle> badTriangles;
        std::vector<std::pair<Point, Point>> polygonEdges;
        int numTriangles = triangles.size();
        Triangle* d_triangles;
        bool* d_isBadTriangle;
        cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
        cudaMalloc(&d_isBadTriangle, numTriangles * sizeof(bool));
        cudaMemcpy(d_triangles, triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        checkCircumcircle<<<numTriangles, 1>>>(d_triangles, d_isBadTriangle, p, numTriangles);
        cudaDeviceSynchronize();
        bool* h_isBadTriangle = new bool[numTriangles];
        cudaMemcpy(h_isBadTriangle, d_isBadTriangle, numTriangles * sizeof(bool), cudaMemcpyDeviceToHost);
        for (int i = 0; i < numTriangles; ++i) {
            if (h_isBadTriangle[i]) {
                badTriangles.push_back(triangles[i]);
                addEdgeIfUnique(polygonEdges, {triangles[i].a, triangles[i].b});
                addEdgeIfUnique(polygonEdges, {triangles[i].b, triangles[i].c});
                addEdgeIfUnique(polygonEdges, {triangles[i].c, triangles[i].a});
            }
        }

        delete[] h_isBadTriangle;
        for (const auto& tri : badTriangles) {
            removeTriangle(tri);
        }
        for (const auto& edge : polygonEdges) {
            triangles.emplace_back(edge.first, edge.second, p);
        }
    }
    void addEdgeIfUnique(std::vector<std::pair<Point, Point>>& edges, const std::pair<Point, Point>& edge) {
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            if ((it->first.x == edge.second.x && it->first.y == edge.second.y &&
                 it->second.x == edge.first.x && it->second.y == edge.first.y) ||
                (it->first.x == edge.first.x && it->first.y == edge.first.y &&
                 it->second.x == edge.second.x && it->second.y == edge.second.y)) {
                edges.erase(it);
                return;
            }
        }
        edges.push_back(edge);
    }
    void removeTriangle(const Triangle& tri) {
        auto it = std::remove(triangles.begin(), triangles.end(), tri);
        triangles.erase(it, triangles.end());
    }    
    void exportTriangles(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing!" << std::endl;
            return;
        }
        for (const auto& tri : triangles) {
            file << tri.a.x << " " << tri.a.y << "\n";
            file << tri.b.x << " " << tri.b.y << "\n";
            file << tri.c.x << " " << tri.c.y << "\n";
            file << "\n";
        }

        file.close();
    }
};

#endif // CLASSES_CUH
