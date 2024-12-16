# TODO

## Memory :
- [ ] Utilize Shared Memory for Bad Triangle Detection .
```cpp
__global__ void checkCircumcircle(Triangle* triangles, bool* isBadTriangle, const Point p, int numTriangles) {
    extern __shared__ Triangle sharedTriangles[];
    
    int threadIdx1D = threadIdx.x + threadIdx.y * blockDim.x;
    int globalIdx = threadIdx1D + blockIdx.x * blockDim.x*blockDim.y;
    
    // Load triangles into shared memory
    if (threadIdx1D < blockDim.x*blockDim.y && globalIdx < numTriangles) {
        sharedTriangles[threadIdx1D] = triangles[globalIdx];
    }
    __syncthreads();
    
    if (globalIdx < numTriangles) {
        isBadTriangle[globalIdx] = sharedTriangles[threadIdx1D].isInCircumcircle(p);
    }
}
```
## Parallelism
- [ ] Use Parallel Reduction Pattern for Edge Processing
```cpp
__global__ void findUniqueEdges(Triangle* badTriangles, Edge* edges, int* edgeCount, int numBadTriangles) {
    __shared__ Edge sharedEdges[1024]; // Adjust size as needed
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load edges into shared memory
    if (gid < numBadTriangles) {
        Triangle tri = badTriangles[gid];
        sharedEdges[tid*3] = Edge(tri.a, tri.b);
        sharedEdges[tid*3 + 1] = Edge(tri.b, tri.c);
        sharedEdges[tid*3 + 2] = Edge(tri.c, tri.a);
    }
    __syncthreads();
    
    // Parallel reduction to find unique edges
    // ... implement reduction logic here
}
```
- [ ] Implement Block-Level Parallelism for Triangle Removal
```cpp
__global__ void markTrianglesToRemove(Triangle* triangles, Triangle* badTriangles, bool* toRemove, int n) {
    extern __shared__ Triangle sharedBadTriangles[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load bad triangles into shared memory
    if (tid < blockDim.x) {
        sharedBadTriangles[tid] = badTriangles[tid];
    }
    __syncthreads();
    
    // Mark triangles for removal
    for (int i = bid * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        // Compare with shared bad triangles
    }
}
```
## Other

- [ ] Use Grid-Stride Loops for Large Triangle Sets
```cpp
__global__ void processBadTriangles(Triangle* triangles, bool* isBadTriangles, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride) {
        // Process triangles
    }
}
```
- [ ] Use Coalesced Memory Access
```cpp
struct TriangleData {
    Point* points_a;
    Point* points_b;
    Point* points_c;
};
```
