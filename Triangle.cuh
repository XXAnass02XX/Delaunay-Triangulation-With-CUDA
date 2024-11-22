#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 
#include <fstream> 
#include "Point.cuh"

class Triangle {
    public:
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

#endif