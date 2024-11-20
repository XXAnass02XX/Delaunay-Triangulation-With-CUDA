#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "Classes.cuh"
#include <ctime> 

#define NUM_POINTS 5


int main() {
    std::srand(static_cast<unsigned>(std::time(0)));
    // Number of points for the Delaunay triangulation

    // Create a vector of points (random points for this example)
    std::vector<Point> points = {Point(0.27249, 0.484608),Point(0.353358, 0.493073),Point(0.548086, 0.270041),Point(0.526795, 0.516974),Point(0.22938, 0.331764)};
    /*std::vector<Point> points;
    for (int i = 0; i < NUM_POINTS; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        points.emplace_back(x, y);
    }*/

    // Create an instance of DelaunayTriangulation
    DelaunayTriangulation delaunay;

    delaunay.initializeWithSuperTriangle();

    // Add points one by one and update the triangulation
    std::string filename;
    for (int i = 0; i < NUM_POINTS; ++i) {
        delaunay.addPoint(points[i]);
        DelaunayTriangulation delau_debug = DelaunayTriangulation(delaunay);
        delau_debug.removeTrianglesWithSuperVertices();
        filename = "triangles_"  + std::to_string(i)+ ".txt";
        delau_debug.exportTriangles(filename);

        
        std::cout << "----------------" << i  << std::endl;
    }

    delaunay.removeTrianglesWithSuperVertices();
    // Optionally, print the resulting triangles
    std::cout << "Final number of triangles: " << delaunay.triangles.size() << std::endl;
    for (const auto& triangle : delaunay.triangles) {
        std::cout << "Triangle: (" << triangle.a.x << ", " << triangle.a.y << "), "
                  << "(" << triangle.b.x << ", " << triangle.b.y << "), "
                  << "(" << triangle.c.x << ", " << triangle.c.y << ")" << std::endl;
    }

    // Construct the filename with the number of points
    /*std::string filename = "triangles_" + std::to_string(points.size()) + ".txt";

    // Export the triangles to a text file
    delaunay.exportTriangles(filename);

    std::cout << "Triangulation completed and exported to triangles.txt" << std::endl;*/

    return 0;
}
