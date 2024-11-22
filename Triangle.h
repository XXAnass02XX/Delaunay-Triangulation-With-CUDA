#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Point.h"

class Triangle {
public:
    Point a, b, c;

    Triangle(const Point& a, const Point& b, const Point& c) : a(a), b(b), c(c) {}
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
#endif 