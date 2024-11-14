#include "Classes.cuh"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <vector>

#define NUM_POINTS 10
#define BLOCK_SIZE 256

__global__ void generateRandomPoints(Point* points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_POINTS) {
        curandState state;
        curand_init((unsigned long long)clock() + idx, 0, 0, &state);
        points[idx].x = curand_uniform(&state) * 2000 - 1000; // Random x between -1000 and 1000
        points[idx].y = curand_uniform(&state) * 2000 - 1000; // Random y between -1000 and 1000
    }
}

__global__ void findBadTriangles(const Triangle* triangles, int numTriangles, const Point p, bool* badTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTriangles) {
        bool isBad = triangles[idx].isInCircumcircle(p);
        badTriangles[idx] = isBad;
        // Debug print (only on the first thread in the block to avoid excessive output)
        if (threadIdx.x == 0) {
            printf("Triangle %d: %s\n", idx, isBad ? "Bad" : "Good");
        }
    }
}


void exportTriangles(const std::vector<Triangle>& triangles, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }
    for (const auto& tri : triangles) {
        file << tri.a.x << " " << tri.a.y << "\n";
        file << tri.b.x << " " << tri.b.y << "\n";
        file << tri.c.x << " " << tri.c.y << "\n\n";
    }
    file.close();
}

int main() {
    DelaunayTriangulation triangulation;
    triangulation.initializeWithSuperTriangle();

    std::vector<Point> h_points(NUM_POINTS);
    Point* d_points;
    cudaMalloc(&d_points, NUM_POINTS * sizeof(Point));

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (NUM_POINTS + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Generating random points on the device..." << std::endl;
    generateRandomPoints<<<blocksPerGrid, threadsPerBlock>>>(d_points);
    cudaDeviceSynchronize();  // Ensure the kernel finishes before moving on
    std::cout << "Random points generated." << std::endl;

    cudaMemcpy(h_points.data(), d_points, NUM_POINTS * sizeof(Point), cudaMemcpyDeviceToHost);
    std::cout << "Points copied from device to host." << std::endl;

    // Copy triangles to device
    Triangle* d_triangles;
    cudaMalloc(&d_triangles, triangulation.triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, triangulation.triangles.data(), triangulation.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    std::vector<int> h_badTriangles(triangulation.triangles.size(), 0);
    bool* d_badTriangles;
    cudaMalloc(&d_badTriangles, h_badTriangles.size() * sizeof(bool));

    for (int i = 0; i < NUM_POINTS; ++i) {
        std::cout << "Processing point " << i + 1 << "/" << NUM_POINTS << "..." << std::endl;

        Point p = h_points[i];
        blocksPerGrid = (triangulation.triangles.size() + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "Launching kernel to find bad triangles for point (" << p.x << ", " << p.y << ")..." << std::endl;
        findBadTriangles<<<blocksPerGrid, threadsPerBlock>>>(
            d_triangles, triangulation.triangles.size(), p, d_badTriangles
        );
        cudaDeviceSynchronize();  // Ensure the kernel finishes before moving on
        std::cout << "Kernel completed for point (" << p.x << ", " << p.y << ")." << std::endl;

        cudaMemcpy(h_badTriangles.data(), d_badTriangles, h_badTriangles.size() * sizeof(bool), cudaMemcpyDeviceToHost);

        // Debug print to check bad triangles
        std::cout << "Bad triangles detected for point (" << p.x << ", " << p.y << "): ";
        for (int j = 0; j < h_badTriangles.size(); ++j) {
            if (h_badTriangles[j]) {
                std::cout << j << " ";  // Index of bad triangle
            }
        }
        std::cout << std::endl;

        // After finding bad triangles, remove them and add new ones
        std::vector<Triangle> newTriangles;
        std::vector<Triangle> remainingTriangles;
        for (int j = 0; j < triangulation.triangles.size(); ++j) {
            if (h_badTriangles[j]) {
                std::cout << "Creating new triangles for bad triangle " << j << std::endl;
                // Add new triangles based on the point p
                newTriangles.push_back(Triangle(p, triangulation.triangles[j].a, triangulation.triangles[j].b));
                newTriangles.push_back(Triangle(p, triangulation.triangles[j].b, triangulation.triangles[j].c));
                newTriangles.push_back(Triangle(p, triangulation.triangles[j].c, triangulation.triangles[j].a));
            } else {
                remainingTriangles.push_back(triangulation.triangles[j]);
            }
        }

        // Combine remaining and new triangles
        triangulation.triangles = remainingTriangles;
        triangulation.triangles.insert(triangulation.triangles.end(), newTriangles.begin(), newTriangles.end());
        std::cout << "Point " << i + 1 << " processed. Total triangles: " << triangulation.triangles.size() << std::endl;
    }

    triangulation.removeTrianglesWithSuperVertices();
    std::cout << "Removed super triangle vertices." << std::endl;

    std::string filename = "triangles_" + std::to_string(NUM_POINTS) + ".txt";
    exportTriangles(triangulation.triangles, filename);

    std::cout << "Triangulation completed and exported to " << filename << std::endl;

    cudaFree(d_points);
    cudaFree(d_triangles);
    cudaFree(d_badTriangles);

    return 0;
}
