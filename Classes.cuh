#ifndef CLASSES_CUH
#define CLASSES_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// Point structure
struct Point {
    double x, y;

    __host__ __device__ Point(double x = 0, double y = 0) : x(x), y(y) {}

    __host__ __device__ bool operator==(const Point& other) const {
        return (x == other.x && y == other.y);
    }
};

// Triangle structure
struct Triangle {
    Point a, b, c;

    __host__ __device__ Triangle(const Point& a, const Point& b, const Point& c) : a(a), b(b), c(c) {}

    __host__ __device__ bool isInCircumcircle(const Point& p) const {
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

    __host__ __device__ bool operator==(const Triangle& other) const {
        return (a == other.a && b == other.b && c == other.c);
    }
};

// Delaunay triangulation class
class DelaunayTriangulation {
public:
    std::vector<Triangle> triangles;

    DelaunayTriangulation() {}

    // Initializes the triangulation with a large "super triangle" containing all points
    void initializeWithSuperTriangle() {
        Point p1(-1000, -1000);
        Point p2(1000, -1000);
        Point p3(0, 1000);
        triangles.push_back(Triangle(p1, p2, p3));
    }

    // Removes triangles that contain any super triangle vertices
    void removeTrianglesWithSuperVertices() {
        Point p1(-1000, -1000);
        Point p2(1000, -1000);
        Point p3(0, 1000);

        auto isSuperVertexTriangle = [&](const Triangle& tri) {
            return (tri.a == p1 || tri.a == p2 || tri.a == p3 ||
                    tri.b == p1 || tri.b == p2 || tri.b == p3 ||
                    tri.c == p1 || tri.c == p2 || tri.c == p3);
        };

        triangles.erase(std::remove_if(triangles.begin(), triangles.end(), isSuperVertexTriangle), triangles.end());
    }
};

#endif // CLASSES_CUH
