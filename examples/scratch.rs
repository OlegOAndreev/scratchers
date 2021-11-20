use scratchers::aligned_vec::{AlignedMatrix, AlignedVec};

extern "C" {
    #[cfg(feature = "ispc")]
    pub fn ispc_matrix_vec_mul_v1_parallel(
        src_mat: *const f32,
        src_mat_stride: i32,
        src_vec: *const f32,
        dst_vec: *mut f32,
        K: i32,
        L: i32,
    );
}

#[cfg(feature = "ispc")]
pub fn ispc_matrix_vec_mul_v1_wrapper(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let src_mat_slice = src_mat.as_f32_whole();
    let src_mat_stride = src_mat_slice.len() / src_mat.height();
    unsafe {
        ispc_matrix_vec_mul_v1_parallel(
            src_mat_slice.as_ptr(),
            src_mat_stride as i32,
            src_vec.as_f32().as_ptr(),
            dst_vec.as_f32_mut().as_mut_ptr(),
            src_vec.len() as i32,
            dst_vec.len() as i32,
        );
    }
}

fn main() {
    let mut src_mat = AlignedMatrix::new(4, 2);
    src_mat[0][0] = 1.0;
    src_mat[1][0] = 2.0;
    src_mat[0][1] = 3.0;
    src_mat[1][1] = 4.0;
    src_mat[0][2] = 5.0;
    src_mat[1][2] = 6.0;
    src_mat[0][3] = 7.0;
    src_mat[1][3] = 8.0;
    let mut src_vec = AlignedVec::new(4);
    src_vec[0] = 10.0;
    src_vec[1] = 11.0;
    src_vec[2] = 12.0;
    src_vec[3] = 13.0;
    let mut dst_vec = AlignedVec::new(2);
    ispc_matrix_vec_mul_v1_wrapper(&src_mat, &src_vec, &mut dst_vec);
    eprintln!("{:?}", dst_vec.as_f32());
}
