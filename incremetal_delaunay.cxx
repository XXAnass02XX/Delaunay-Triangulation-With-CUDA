#include <iostream>
#include <vector>
#include <fstream>
//#include "Classes.h"
#include "DelaunayTrianglulation.h"
#include "Triangle.h"
#include "Point.h"
#include <cstdlib> 
#include <ctime> 


int main() {
    // Create a DelaunayTriangulation instance
    DelaunayTriangulation triangulation;

    // Seed for random number generation
    std::srand(static_cast<unsigned>(std::time(0)));

    
    std::vector<Point> points = {Point(0.27249, 0.484608),Point(0.353358, 0.493073),Point(0.548086, 0.270041),Point(0.526795, 0.516974),Point(0.22938, 0.331764)};
    /*for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX; // we divide by rand_max to normalize in [0,1] 
        double y = static_cast<double>(std::rand()) / RAND_MAX; // TODO just float
        points.emplace_back(x, y); //todo why emplace_back and not just push_back
    }*/
    
    triangulation.initializeWithSuperTriangle();

    // Insert each point into the triangulation
    int i = 0;
    std::string filename;
    for (const auto& point : points) {
        triangulation.addPoint(point,i);
        
        DelaunayTriangulation delau_debug = DelaunayTriangulation(triangulation);
        // delau_debug.removeTrianglesWithSuperVertices();
        filename = "triangles_"  + std::to_string(i)+ ".txt";
        delau_debug.exportTriangles(filename);
        i++;
    }    

    // Filter out triangles that include any super triangle vertices
    triangulation.removeTrianglesWithSuperVertices();


    // Construct the filename with the number of points
    filename = "seq_triangles_" + std::to_string(points.size()) + ".txt";

    // Export the triangles to a text file
    triangulation.exportTriangles(filename);

    std::cout << "Triangulation completed and exported to triangles.txt" << std::endl;
    return 0;
}
