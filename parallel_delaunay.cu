#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "Classes.cuh"
#include <ctime> 

#define NUM_POINTS 100*100


int main() {
    std::srand(static_cast<unsigned>(std::time(0)));
    // Number of points for the Delaunay triangulation

    // Create a vector of points (random points for this example)
    //std::vector<Point> points = {Point(0.27249, 0.484608),Point(0.353358, 0.493073),Point(0.548086, 0.270041),Point(0.526795, 0.556974),Point(0.22938, 0.331764)};
    /*std::vector<Point> points;
    for (int i = 0; i < NUM_POINTS; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        points.emplace_back(x, y);
    }*/
      std::vector<Point> points;
   for (int i = 0;i<200;i++){
    for (int j = 0;j < 100;j++){
        points.emplace_back(((float)i)/1000,((float)j)/1000);
    }
   }

    // Create an instance of DelaunayTriangulation
    DelaunayTriangulation delaunay;

    delaunay.initializeWithSuperTriangle();

    // Add points one by one and update the triangulation
    int i = 0;
    std::string filename;
    for (const auto& point : points) {
        delaunay.addPoint(point, i);
        /*DelaunayTriangulation delau_debug = DelaunayTriangulation(delaunay);
        //delau_debug.removeTrianglesWithSuperVertices();
        filename = "par_triangles_txt/par_triangles_"  + std::to_string(i)+ ".txt";
        delau_debug.exportTriangles(filename);*/
    }

    delaunay.removeTrianglesWithSuperVertices();

    // Construct the filename with the number of points
    filename = "par_triangles_txt/par_triangles_" + std::to_string(points.size()) + ".txt";

    // Export the triangles to a text file
    delaunay.exportTriangles(filename);

    return 0;
}
