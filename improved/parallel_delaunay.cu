#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "DelaunayTriangulation.cuh"
#include "Point.cuh"
#include "Triangle.cuh"
#include <ctime>

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));
    std::vector<Point> points;
    
    // Create a 50x50 grid of points.
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 50; j++) {
            points.emplace_back(((float)i) / 1000, ((float)j) / 1000);
        }
    }
    
    // Allocate unified memory for the triangulation.
    DelaunayTriangulation delaunay(10000);  // Adjust maxTriangles as needed.
    delaunay.initializeWithSuperTriangle();
    
    // Incrementally add each point.
    for (const auto& point : points) {
        delaunay.addPoint(point);
    }
    
    delaunay.removeTrianglesWithSuperVertices();
    
    std::string filename = "par_triangles_txt/par_triangles_" + std::to_string(points.size()) + ".txt";
    delaunay.exportTriangles(filename);
    
    return 0;
}
