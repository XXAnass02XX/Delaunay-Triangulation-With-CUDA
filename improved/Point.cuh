#ifndef POINT_CUH
#define POINT_CUH

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 
#include <fstream>  

class Point {
    public:
    double x, y;

    __host__ __device__ Point(double x = 0, double y = 0) : x(x), y(y) {}

    __host__ __device__ bool operator==(const Point& other) const {
        return (x == other.x && y == other.y);
    }
};
#endif