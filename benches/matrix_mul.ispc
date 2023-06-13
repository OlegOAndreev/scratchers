export void simplest_ispc_mul_impl(
    uniform int size,
    uniform float src_mat_1[],
    uniform float src_mat_2[],
    uniform float dst_mat[]
) {
    // Help the optimizer: we always pass arrays with length aligned to 8.
    size = size & ~7;
    for (uniform int i = 0; i < size; i++) {
        for (uniform int j = 0; j < size; j++) {
            varying float accum = 0.0f;
            foreach (k = 0 ... size) {
                accum += src_mat_1[i * size + k] * src_mat_2[k * size + j];
            }
            dst_mat[i * size + j] = reduce_add(accum);
        }
    }
}
