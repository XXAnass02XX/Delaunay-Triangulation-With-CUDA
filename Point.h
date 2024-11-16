#ifndef POINT_H
#define POINT_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class Point {
public:
    double x, y;
    
    Point(double x = 0, double y = 0) : x(x), y(y) {}

    bool operator==(const Point& other) const {
        return (x == other.x && y == other.y);
    }
};

#endif 