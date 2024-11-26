#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
//#include "Classes.cuh"
#include "DelaunayTriangulation.cuh"
#include "Point.cuh"
#include "Triangle.cuh"
#include <ctime> 

#define NUM_POINTS 100*100


int main() {
    std::srand(static_cast<unsigned>(std::time(0)));
    std::vector<Point> points;
    for (int i = 0;i < 50;i++){
        for (int j = 0;j < 50;j++){
            points.emplace_back(((float)i)/1000,((float)j)/1000);
        }
    }
    DelaunayTriangulation delaunay;
    delaunay.initializeWithSuperTriangle();
    int i = 0;
    std::string filename;
    for (const auto& point : points) {
        delaunay.addPoint(point, i);
    }

    delaunay.removeTrianglesWithSuperVertices();
    filename = "par_triangles_txt/par_triangles_" + std::to_string(points.size()) + ".txt";
    delaunay.exportTriangles(filename);
    return 0;
}
