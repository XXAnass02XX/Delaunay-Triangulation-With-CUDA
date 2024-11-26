
// parallizing the apppend to a list but 2 second slower for 100*100 points since we use atomic add 
__global__ void checkCircumcircle_01(Triangle* triangles, int* isBadTriangle, const Point p, int numTriangles, int* last_triangle_idx) {
    int threadIdx1D = threadIdx.x + threadIdx.y * blockDim.x;
    int globalIdx = threadIdx1D + blockIdx.x * blockDim.x*blockDim.y;
    if (globalIdx < numTriangles) {
        if (triangles[globalIdx].isInCircumcircle(p)){
            int idx = atomicAdd(last_triangle_idx, 1);
            isBadTriangle[idx] = globalIdx;
        }
    }
}
//+++++
std::vector<std::pair<Point, Point>> polygonEdges;
        int numTriangles = triangles.size();
        Triangle* d_triangles;
        int* d_isBadTriangle;
        int* d_last_idx;

        cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
        cudaMalloc(&d_isBadTriangle, numTriangles * sizeof(int));
        cudaMalloc(&d_last_idx, sizeof(int));
        int zero = 0;
        cudaMemcpy(d_last_idx, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_triangles, triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

        dim3 dimBlock(32, 32);
        dim3 dimGrid((numTriangles + 1024 - 1) / 1024);

        checkCircumcircle_01<<<dimGrid, dimBlock>>>(d_triangles, d_isBadTriangle, p, numTriangles, d_last_idx);
        cudaDeviceSynchronize();

        int last_idx;
        cudaMemcpy(&last_idx, d_last_idx, sizeof(int), cudaMemcpyDeviceToHost);

        int* h_isBadTriangle = new int[last_idx];
        cudaMemcpy(h_isBadTriangle, d_isBadTriangle, last_idx * sizeof(int), cudaMemcpyDeviceToHost);

        int bad_triangle_idx;
        std::vector<Triangle> badTriangles;
        for (int j = 0; j < last_idx; j++) {
            bad_triangle_idx = h_isBadTriangle[j];
            badTriangles.push_back(triangles[bad_triangle_idx]);
            addEdgeIfUnique(polygonEdges, {triangles[bad_triangle_idx].a, triangles[bad_triangle_idx].b});
            addEdgeIfUnique(polygonEdges, {triangles[bad_triangle_idx].b, triangles[bad_triangle_idx].c});
            addEdgeIfUnique(polygonEdges, {triangles[bad_triangle_idx].c, triangles[bad_triangle_idx].a});
        }
//---------------------------------------------------------------------------