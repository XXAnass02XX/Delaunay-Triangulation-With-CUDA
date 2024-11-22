#ifndef DELAUNAYTRIANGULATION_H
#define DELAUNAYTRIANGULATION_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Point.h"
#include "Triangle.h"
#include <fstream>
class DelaunayTriangulation {
private:
    std::vector<Triangle> triangles;

public:
    DelaunayTriangulation() {}

    void addPoint(const Point& p,int i) {
        std::vector<Triangle> badTriangles; // TODO free these
        std::vector<std::pair<Point, Point>> polygonEdges;
        for (const auto& tri : triangles) {
            if (tri.isInCircumcircle(p)) {
                badTriangles.push_back(tri);
            }
        }

        for (const auto& tri : badTriangles) {
            addEdgeIfUnique(polygonEdges, {tri.a, tri.b});
            addEdgeIfUnique(polygonEdges, {tri.b, tri.c});
            addEdgeIfUnique(polygonEdges, {tri.c, tri.a});
        }

        for (const auto& tri : badTriangles) {
            removeTriangle(tri);
        }

        for (const auto& edge : polygonEdges) {
            triangles.emplace_back(edge.first, edge.second, p);
        }
    }

    void initializeWithSuperTriangle() {
        Point p1(-1.5, -1.5);
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);
        triangles.emplace_back(p1, p2, p3);
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

    void removeTrianglesWithSuperVertices() {
        Point p1(-1.5, -1.5);
        Point p2(1.5, -1.5);
        Point p3(0, 1.5);
        triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
            [&](const Triangle& tri) {
                return (tri.a == p1 || tri.a == p2 || tri.a == p3 ||
                        tri.b == p1 || tri.b == p2 || tri.b == p3 ||
                        tri.c == p1 || tri.c == p2 || tri.c == p3);
            }), triangles.end());
    }

private:
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
        triangles.erase(
            std::remove(triangles.begin(), triangles.end(), tri),
            triangles.end()
        );
    }

};

#endif 