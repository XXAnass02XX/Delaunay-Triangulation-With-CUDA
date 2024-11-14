#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Classe représentant un point dans le plan 2D
class Point {
public:
    double x, y;
    
    Point(double x = 0, double y = 0) : x(x), y(y) {}

    bool operator==(const Point& other) const {
        return (x == other.x && y == other.y);
    }
};

// Classe représentant un triangle
class Triangle {
public:
    Point a, b, c;

    Triangle(const Point& a, const Point& b, const Point& c) : a(a), b(b), c(c) {}

    // Méthode pour vérifier si un point p est à l'intérieur du cercle circonscrit du triangle
    bool isInCircumcircle(const Point& p) const {
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

    bool operator==(const Triangle& other) const {
        return (a.x == other.a.x && a.y == other.a.y &&
                b.x == other.b.x && b.y == other.b.y &&
                c.x == other.c.x && c.y == other.c.y);
    }

};

// Classe représentant la triangulation de Delaunay
class DelaunayTriangulation {
private:
    std::vector<Triangle> triangles;

public:
    DelaunayTriangulation() {}

    void addPoint(const Point& p) {
        std::vector<Triangle> badTriangles;
        std::vector<std::pair<Point, Point>> polygonEdges;

        // Trouver tous les triangles dont le cercle circonscrit contient le point p
        for (const auto& tri : triangles) {
            if (tri.isInCircumcircle(p)) {
                badTriangles.push_back(tri);
            }
        }

        // std::cout << "Number of bad triangles found: " << badTriangles.size() << std::endl;

        // Trouver les bords de l'enveloppe du polygone formé par les triangles à supprimer
        for (const auto& tri : badTriangles) {
            // Ajouter les arêtes du triangle aux arêtes du polygone si elles sont partagées par un seul triangle
            addEdgeIfUnique(polygonEdges, {tri.a, tri.b});
            addEdgeIfUnique(polygonEdges, {tri.b, tri.c});
            addEdgeIfUnique(polygonEdges, {tri.c, tri.a});
        }

        // std::cout << "Number of unique polygon edges: " << polygonEdges.size() << std::endl;

        // Supprimer les triangles "mauvais" de la triangulation
        for (const auto& tri : badTriangles) {
            removeTriangle(tri);
        }

        // std::cout << "Remaining triangles after removal: " << triangles.size() << std::endl;

        // Créer de nouveaux triangles reliant le point p aux arêtes de l'enveloppe
        
        for (const auto& edge : polygonEdges) {
            triangles.emplace_back(edge.first, edge.second, p);
            // std::cout << "Triangle created with vertices: (" << edge.first.x << ", " << edge.first.y << "), ("
            //         << edge.second.x << ", " << edge.second.y << "), ("
            //         << p.x << ", " << p.y << ")\n";
        }

        // std::cout << "Total triangles after adding new ones: " << triangles.size() << std::endl;
    }

    void initializeWithSuperTriangle() {
        // Define a super triangle large enough to contain all points
        Point p1(-1000, -1000);
        Point p2(1000, -1000);
        Point p3(0, 1000);
        triangles.emplace_back(p1, p2, p3);
    }


    void exportTriangles(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing!" << std::endl;
            return;
        }

        // Export points and triangles
        for (const auto& tri : triangles) {
            file << tri.a.x << " " << tri.a.y << "\n";
            file << tri.b.x << " " << tri.b.y << "\n";
            file << tri.c.x << " " << tri.c.y << "\n";
            file << "\n";  // Separate each triangle with a blank line
        }

        file.close();
    }

    void removeTrianglesWithSuperVertices() {
        Point p1(-1000, -1000);
        Point p2(1000, -1000);
        Point p3(0, 1000);
        triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
            [&](const Triangle& tri) {
                return (tri.a == p1 || tri.a == p2 || tri.a == p3 ||
                        tri.b == p1 || tri.b == p2 || tri.b == p3 ||
                        tri.c == p1 || tri.c == p2 || tri.c == p3);
            }), triangles.end());
    }

private:
    // Méthode pour ajouter une arête si elle est unique
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