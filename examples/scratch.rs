// extern "C" {
//     #[cfg(feature = "ispc")]
//     pub fn simplest_ispc_mul_impl(
//         size: i32,
//         src_mat_1: *const f32,
//         src_mat_2: *const f32,
//         dst_mat: *mut f32,
//     );
// }


use scratchers::tile_mul::mul_tile_simd_impl_8;

fn simplest_mul(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    n: usize,
) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                unsafe {
                    acc += *m1.get_unchecked(i * n + k) * *m2.get_unchecked(k * n + j);
                }
            }
            md[i * n + j] = acc;
        }
    }
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
    // #[cfg(feature = "ispc")]
    // unsafe {
    //     simplest_ispc_mul_impl(SIZE as i32, src_mat_1.as_ptr(), src_mat_2.as_ptr(), dst_mat.as_mut_ptr());
    // }
    mul_tile_simd_impl_8(&src_mat_1, &src_mat_2, &mut dst_mat, 0, 0, 0, 8);
    for i in 0..SIZE {
        eprintln!("{:?}", &dst_mat[i * SIZE..(i + 1) * SIZE]);
    }

    simplest_mul(&src_mat_1, &src_mat_2, &mut dst_mat, SIZE);
    for i in 0..SIZE {
        eprintln!("{:?}", &dst_mat[i * SIZE..(i + 1) * SIZE]);
    }
}
