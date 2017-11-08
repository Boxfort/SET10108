__kernel void vecadd(__global int *A, __global int *B, __global int *C)
{
	// Get the work items unique ID
	int idx = get_global_id(0);
	// Add corresponding locations of A and B and store in C
	C[idx] = A[idx] + B[idx];
}