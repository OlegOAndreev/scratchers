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
use std::arch::asm;

fn simplest_mul(m1: &[f32], m2: &[f32], md: &mut [f32], n: usize) {
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

fn f32x8_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        // Can't use _mm256_set1_ps due to vzeroupper.
        let one = &[1.0f32];
        if STORE == 8 {
            if stride == 8 {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "vmovups ymmword ptr [{data} + 64], {tmp0}",
                    "vmovups ymmword ptr [{data} + 96], {tmp0}",
                    "add {data}, 128",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + {stride}], {tmp0}",
                    "lea {data}, [{data} + {stride} * 2]",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + {stride}], {tmp0}",
                    "lea {data}, [{data} + {stride} * 2]",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            }
        } else if STORE == 16 {
            if stride == 16 {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "vmovups ymmword ptr [{data} + 64], {tmp0}",
                    "vmovups ymmword ptr [{data} + 96], {tmp0}",
                    "vmovups ymmword ptr [{data} + 128], {tmp0}",
                    "vmovups ymmword ptr [{data} + 160], {tmp0}",
                    "vmovups ymmword ptr [{data} + 192], {tmp0}",
                    "vmovups ymmword ptr [{data} + 224], {tmp0}",
                    "add {data}, 256",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            }
        } else {
            unimplemented!();
        }
    } else {
        unimplemented!();
    }
}

fn main() {
    let mut rows = [1.0f32; 800];
    f32x8_store_asm::<8>(rows.as_mut_ptr(), 8, 100);
    // const SIZE: usize = 8;
    // let mut src_mat_1 = vec![0.0f32; SIZE * SIZE];
    // for i in 0..SIZE {
    //     for j in 0..SIZE {
    //         src_mat_1[i * SIZE + j] = (i + j) as f32;
    //     }
    // }
    // let mut src_mat_2 = vec![0.0f32; SIZE * SIZE];
    // for i in 0..SIZE {
    //     for j in 0..SIZE {
    //         src_mat_2[i * SIZE + j] = (i + j * 2) as f32;
    //     }
    // }
    // let mut dst_mat = vec![0.0f32; SIZE * SIZE];
    // // #[cfg(feature = "ispc")]
    // // unsafe {
    // //     simplest_ispc_mul_impl(SIZE as i32, src_mat_1.as_ptr(), src_mat_2.as_ptr(), dst_mat.as_mut_ptr());
    // // }
    // mul_tile_simd_impl_8(&src_mat_1, &src_mat_2, &mut dst_mat, 0, 0, 0, 8);
    // for i in 0..SIZE {
    //     eprintln!("{:?}", &dst_mat[i * SIZE..(i + 1) * SIZE]);
    // }
    //
    // simplest_mul(&src_mat_1, &src_mat_2, &mut dst_mat, SIZE);
    // for i in 0..SIZE {
    //     eprintln!("{:?}", &dst_mat[i * SIZE..(i + 1) * SIZE]);
    // }
}
