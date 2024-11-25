#ifndef DELAUNAYTRIANGULATION_CUH
#define DELAUNAYTRIANGULATION_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 
#include <fstream>  
#include "Point.cuh"
#include "Triangle.cuh"

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
        for (int i = 0; i < numTriangles; ++i) { //TODO we can make this loop in par
            if (h_isBadTriangle[i]) {
                //adding edges from bad triangles to make new ones after
                badTriangles.push_back(triangles[i]);
                //TODO we can make this three function call in par with three threads
                addEdgeIfUnique(polygonEdges, {triangles[i].a, triangles[i].b});
                addEdgeIfUnique(polygonEdges, {triangles[i].b, triangles[i].c});
                addEdgeIfUnique(polygonEdges, {triangles[i].c, triangles[i].a});
            }
        }

        delete[] h_isBadTriangle;
        for (const auto& tri : badTriangles) {
            //TODO to make this in par we need to think of something like having a big vector and setting to true bad vectores (hash table maybe)
            removeTriangle(tri);
        }
        for (const auto& edge : polygonEdges) {
            triangles.emplace_back(edge.first, edge.second, p);
        }
    }
    void addEdgeIfUnique(std::vector<std::pair<Point, Point>>& edges, const std::pair<Point, Point>& edge) {
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            //TODO we can make this in par with three threads
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

#endif