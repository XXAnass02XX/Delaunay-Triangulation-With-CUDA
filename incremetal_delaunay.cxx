#include <iostream>
#include <vector>
#include <fstream>
#include "Classes.h"
#include <cstdlib> 
#include <ctime> 


int main() {
    // Create a DelaunayTriangulation instance
    DelaunayTriangulation triangulation;

    // Seed for random number generation
    std::srand(static_cast<unsigned>(std::time(0)));

    
    std::vector<Point> points;
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        points.emplace_back(x, y);
    }
    
    triangulation.initializeWithSuperTriangle();

    // Insert each point into the triangulation
    for (const auto& point : points) {
        triangulation.addPoint(point);
    }    

    // Filter out triangles that include any super triangle vertices
    triangulation.removeTrianglesWithSuperVertices();


    // Construct the filename with the number of points
    std::string filename = "triangles_" + std::to_string(points.size()) + ".txt";

    // Export the triangles to a text file
    triangulation.exportTriangles(filename);

    std::cout << "Triangulation completed and exported to triangles.txt" << std::endl;
    return 0;
}
