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
    DelaunayTriangulation triangulation;
    std::srand(static_cast<unsigned>(std::time(0)));
   std::vector<Point> points;
   for (int i = 0;i < 50;i++){
    for (int j = 0;j < 50;j++){
        points.emplace_back(((float)i)/1000,((float)j)/1000);
    }
   }
    
    triangulation.initializeWithSuperTriangle();
    int i = 0;
    std::string filename;
    for (const auto& point : points) {
        triangulation.addPoint(point,i);
        i++;
    }    
    triangulation.removeTrianglesWithSuperVertices();
    filename = "seq_triangles_txt/seq_triangles_" + std::to_string(points.size()) + ".txt";
    triangulation.exportTriangles(filename);

    std::cout << "Triangulation completed and exported to triangles.txt" << std::endl;
    return 0;
}
