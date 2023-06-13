extern "C" {
    #[cfg(feature = "ispc")]
    pub fn simplest_ispc_mul_impl(
        size: i32,
        src_mat_1: *const f32,
        src_mat_2: *const f32,
        dst_mat: *mut f32,
    );
}

fn main() {
    const SIZE: usize = 8;
    let mut src_mat_1 = vec![0.0f32; SIZE * SIZE];
    for i in 0..SIZE {
        for j in 0..SIZE {
            src_mat_1[i * SIZE + j] = (i + j) as f32;
        }
    }
    let mut src_mat_2 = vec![0.0f32; SIZE * SIZE];
    for i in 0..SIZE {
        for j in 0..SIZE {
            src_mat_2[i * SIZE + j] = (i + j * 2) as f32;
        }
    }
    let mut dst_mat = vec![0.0f32; SIZE * SIZE];
    #[cfg(feature = "ispc")]
    unsafe {
        simplest_ispc_mul_impl(SIZE as i32, src_mat_1.as_ptr(), src_mat_2.as_ptr(), dst_mat.as_mut_ptr());
    }
    for i in 0..SIZE {
        eprintln!("{:?}", &dst_mat[i * SIZE..(i + 1) * SIZE]);
    }
}
