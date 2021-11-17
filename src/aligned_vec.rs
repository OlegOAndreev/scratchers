use std::ops::{Bound, Index, IndexMut, RangeBounds};

use glam::Vec4;

// This module exists until https://github.com/rust-lang/portable-simd lands in stable.

// Row-major matrix with each row aligned to 16 bytes. NOTE: Interpreting rows as Vec4 allows
// reading the "margin" (0-3 elements past the end of the row).
#[derive(Clone, Debug)]
pub struct AlignedMatrix {
    storage: Vec<Vec4>,
    // Stride is in Vec4.
    stride: usize,
    // width is in f32.
    width: usize,
    height: usize,
}

#[allow(dead_code)]
impl AlignedMatrix {
    // Allocates width x height matrix.
    #[inline(always)]
    pub fn new(width: usize, height: usize) -> Self {
        let stride = (width + 3) / 4;
        Self {
            storage: vec![Vec4::ZERO; stride * height],
            stride,
            width,
            height,
        }
    }

    #[inline(always)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    pub fn height(&self) -> usize {
        self.height
    }

    // Fills the matrix with value. NOTE: Does not affect the margin.
    #[inline(always)]
    pub fn fill(&mut self, value: f32) {
        for row in 0..self.height {
            self.as_f32_mut(row).fill(value);
        }
    }

    // Returns the row as slice of f32 elements.
    #[inline(always)]
    pub fn as_f32(&self, row_index: usize) -> &[f32] {
        let row_start = row_index * self.stride * 4;
        let storage: &[f32] = bytemuck::cast_slice(&self.storage[..]);
        &storage[row_start..row_start + self.width]
    }

    // Returns the row as mutable slice of f32 elements.
    #[inline(always)]
    pub fn as_f32_mut(&mut self, row_index: usize) -> &mut [f32] {
        let row_start = row_index * self.stride * 4;
        let storage: &mut [f32] = bytemuck::cast_slice_mut(&mut self.storage[..]);
        &mut storage[row_start..row_start + self.width]
    }

    // Returns the row as slice of Vec4 elements.
    #[inline(always)]
    pub fn as_vec4(&self, row_index: usize) -> &[Vec4] {
        let row_start = row_index * self.stride;
        &self.storage[row_start..row_start + self.stride]
    }

    // Returns the row as mutable slice of Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_mut(&mut self, row_index: usize) -> &mut [Vec4] {
        let row_start = row_index * self.stride;
        &mut self.storage[row_start..row_start + self.stride]
    }

    // Returns slice of f32 elements from the given row.
    #[inline(always)]
    pub fn as_f32_slice<S>(&self, row_index: usize, bounds: S) -> &[f32]
        where S: RangeBounds<usize>
    {
        let row_start = row_index * self.stride * 4;
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.width,
        };
        assert!(end <= self.width, "Out of bounds slice end: {}, width {}", end, self.width);
        let storage: &[f32] = bytemuck::cast_slice(&self.storage[..]);
        &storage[row_start + start..row_start + end]
    }

    // Returns mutable slice of f32 elements from the given row.
    #[inline(always)]
    pub fn as_f32_slice_mut<S>(&mut self, row_index: usize, bounds: S) -> &mut [f32]
        where S: RangeBounds<usize>
    {
        let row_start = row_index * self.stride * 4;
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.width,
        };
        assert!(end <= self.width, "Out of bounds slice end: {}, width {}", end, self.width);
        let storage: &mut [f32] = bytemuck::cast_slice_mut(&mut self.storage[..]);
        &mut storage[row_start + start..row_start + end]
    }

    // Returns slice of Vec4 elements from the given row. range must be in Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_slice<S>(&self, row_index: usize, bounds: S) -> &[Vec4]
        where S: RangeBounds<usize>
    {
        let row_start = row_index * self.stride;
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.stride,
        };
        &self.storage[row_start + start..row_start + end]
    }

    // Returns mutable slice of Vec4 elements from the given row. range must be in Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_slice_mut<S>(&mut self, row_index: usize, bounds: S) -> &mut [Vec4]
        where S: RangeBounds<usize>
    {
        let row_start = row_index * self.stride;
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.stride,
        };
        &mut self.storage[row_start + start..row_start + end]
    }
}

impl Index<usize> for AlignedMatrix {
    type Output = [f32];

    #[inline(always)]
    fn index(&self, index: usize) -> &[f32] {
        self.as_f32(index)
    }
}

impl IndexMut<usize> for AlignedMatrix {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut [f32] {
        self.as_f32_mut(index)
    }
}

// Vector aligned to 16 bytes. NOTE: Interpreting the vector as Vec4 allows reading 0-3 elements
// past the end.
#[derive(Clone, Debug)]
pub struct AlignedVec {
    storage: Vec<Vec4>,
    len: usize,
}

#[allow(dead_code)]
impl AlignedVec {
    // Allocates vector.
    #[inline(always)]
    pub fn new(len: usize) -> Self {
        let l = (len + 3) / 4;
        Self {
            storage: vec![Vec4::ZERO; l],
            len,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    // Fills the vector with value. NOTE: Does not fill the margin.
    #[inline(always)]
    pub fn fill(&mut self, value: f32) {
        self.as_f32_slice_mut(..).fill(value);
    }

    // Returns iterator over f32 elements.
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<f32> {
        self.as_f32_slice(..).iter()
    }

    // Returns mutable iterator over f32 elements.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f32> {
        self.as_f32_slice_mut(..).iter_mut()
    }

    // Returns the vector as slice of f32 elements.
    #[inline(always)]
    pub fn as_f32(&self) -> &[f32] {
        self.as_f32_slice(..)
    }

    // Returns the vector as mutable slice of f32 elements.
    #[inline(always)]
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        self.as_f32_slice_mut(..)
    }

    // Returns the vector as slice of Vec4 elements.
    #[inline(always)]
    pub fn as_vec4(&self) -> &[Vec4] {
        &self.storage[..]
    }

    // Returns the vector as mutable slice of Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_mut(&mut self) -> &mut [Vec4] {
        &mut self.storage[..]
    }

    // Returns slice of f32 elements.
    #[inline(always)]
    pub fn as_f32_slice<S>(&self, bounds: S) -> &[f32]
        where S: RangeBounds<usize>
    {
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.len,
        };
        assert!(end <= self.len, "Out of bounds slice end: {}, width {}", end, self.len);
        let storage: &[f32] = bytemuck::cast_slice(&self.storage[..]);
        &storage[start..end]
    }

    // Returns mutable slice of f32 elements.
    #[inline(always)]
    pub fn as_f32_slice_mut<S>(&mut self, bounds: S) -> &mut [f32]
        where S: RangeBounds<usize>
    {
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => self.len,
        };
        assert!(end <= self.len, "Out of bounds slice end: {}, width {}", end, self.len);
        let storage: &mut [f32] = bytemuck::cast_slice_mut(&mut self.storage[..]);
        &mut storage[start..end]
    }

    // Returns slice of Vec4 elements. range must be in Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_slice<S>(&self, bounds: S) -> &[Vec4]
        where S: RangeBounds<usize>
    {
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => (self.len + 3) / 4,
        };
        &self.storage[start..end]
    }

    // Returns mutable slice of Vec4 elements. range must be in Vec4 elements.
    #[inline(always)]
    pub fn as_vec4_slice_mut<S>(&mut self, bounds: S) -> &mut [Vec4]
        where S: RangeBounds<usize>
    {
        let start = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&bound) => bound + 1,
            Bound::Excluded(&bound) => bound,
            Bound::Unbounded => (self.len + 3) / 4,
        };
        &mut self.storage[start..end]
    }
}

impl Index<usize> for AlignedVec {
    type Output = f32;

    #[inline(always)]
    fn index(&self, index: usize) -> &f32 {
        let storage: &[f32] = bytemuck::cast_slice(&self.storage[..]);
        &storage[index]
    }
}

impl IndexMut<usize> for AlignedVec {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        let storage: &mut [f32] = bytemuck::cast_slice_mut(&mut self.storage[..]);
        &mut storage[index]
    }
}

#[cfg(test)]
mod tests {
    use glam::{vec4, Vec4};

    use crate::aligned_vec::{AlignedMatrix, AlignedVec};

    #[test]
    fn test_aligned_matrix() {
        const BASE: f32 = 123.0;
        for w in 0..64 {
            for h in 0..9 {
                let mut m = AlignedMatrix::new(w, h);

                // Fill in matrix
                for r in 0..h {
                    for i in 0..w {
                        m[r][i] = (i + r * 5) as f32 + BASE;
                    }
                }

                // Read through f32 slice.
                for r in 0..h {
                    let s = m.as_f32(r);
                    assert_eq!(s.len(), w);
                    for i in 0..w {
                        assert_eq!(s[i], (i + r * 5) as f32 + BASE);
                    }
                }

                for r in 0..h {
                    let s = m.as_f32_slice(r, ..);
                    assert_eq!(s.len(), w);
                    for i in 0..w {
                        assert_eq!(s[i], (i + r * 5) as f32 + BASE);
                    }
                }

                for r in 0..h {
                    for i in 0..w + 1 {
                        let s = m.as_f32_slice(r, i..);
                        assert_eq!(s.len(), w - i);
                        if i < w {
                            assert_eq!(s[0], (i + r * 5) as f32 + BASE);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w + 1 {
                        let s = m.as_f32_slice(r, ..i);
                        assert_eq!(s.len(), i);
                        if i > 0 {
                            assert_eq!(s[i - 1], (i - 1 + r * 5) as f32 + BASE);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w {
                        let s = m.as_f32_slice(r, i..i + 1);
                        assert_eq!(s.len(), 1);
                        assert_eq!(s[0], (i + r * 5) as f32 + BASE);
                    }
                }

                if w > 0 {
                    for r in 0..h {
                        for i in 0..w - 1 {
                            let s = m.as_f32_slice(r, i..i + 2);
                            assert_eq!(s.len(), 2);
                            assert_eq!(s[0], (i + r * 5) as f32 + BASE);
                            assert_eq!(s[1], (i + 1 + r * 5) as f32 + BASE);
                        }
                    }
                }

                // Read through Vec4 slice.
                let w4 = (w + 3) / 4;
                for r in 0..h {
                    let s = m.as_vec4(r);
                    assert_eq!(s.len(), w4);
                    for i in 0..w4 {
                        assert_eq!(s[i], get_row_vec4(r, i, w, BASE));
                    }
                }

                for r in 0..h {
                    let s = m.as_vec4_slice(r, ..);
                    assert_eq!(s.len(), w4);
                    for i in 0..w4 {
                        assert_eq!(s[i], get_row_vec4(r, i, w, BASE));
                    }
                }

                for r in 0..h {
                    for i in 0..w4 + 1 {
                        let s = m.as_vec4_slice(r, i..);
                        assert_eq!(s.len(), w4 - i);
                        if i < w4 {
                            assert_eq!(s[0], get_row_vec4(r, i, w, BASE));
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w4 + 1 {
                        let s = m.as_vec4_slice(r, ..i);
                        assert_eq!(s.len(), i);
                        if i > 0 {
                            assert_eq!(s[i - 1], get_row_vec4(r, i - 1, w, BASE));
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w4 {
                        let s = m.as_vec4_slice(r, i..i + 1);
                        assert_eq!(s.len(), 1);
                        assert_eq!(s[0], get_row_vec4(r, i, w, BASE));
                    }
                }

                if w4 > 0 {
                    for r in 0..h {
                        for i in 0..w4 - 1 {
                            let s = m.as_vec4_slice(r, i..i + 2);
                            assert_eq!(s.len(), 2);
                            assert_eq!(s[0], get_row_vec4(r, i, w, BASE));
                            assert_eq!(s[1], get_row_vec4(r, i + 1, w, BASE));
                        }
                    }
                }

                // Write through f32 slice.
                for r in 0..h {
                    for i in 0..w {
                        let value = i as f32 + BASE * 10.0;
                        m.as_f32_mut(r)[i] = value;
                        assert_eq!(m.as_f32_slice(r, i..i + 1)[0], value)
                    }
                }

                for r in 0..h {
                    for i in 0..w {
                        let value = i as f32 + BASE * 2.0;
                        m.as_f32_slice_mut(r, ..)[i] = value;
                        assert_eq!(m.as_f32_slice(r, i..i + 1)[0], value)
                    }
                }

                for r in 0..h {
                    for i in 0..w + 1 {
                        let s = m.as_f32_slice_mut(r, i..);
                        assert_eq!(s.len(), w - i);
                        if i < w {
                            let value = i as f32 + BASE * 3.0;
                            s[0] = value;
                            assert_eq!(m.as_f32_slice(r, i..i + 1)[0], value);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w + 1 {
                        let s = m.as_f32_slice_mut(r, ..i);
                        assert_eq!(s.len(), i);
                        if i > 0 {
                            let value = i as f32 + BASE * 4.0;
                            s[i - 1] = value;
                            assert_eq!(m.as_f32_slice(r, i - 1..i)[0], value);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w {
                        let s = m.as_f32_slice_mut(r, i..i + 1);
                        assert_eq!(s.len(), 1);
                        let value = i as f32 + BASE * 5.0;
                        s[0] = value;
                        assert_eq!(m.as_f32_slice(r, i..i + 1)[0], value);
                    }
                }

                // Write through Vec4 slice.
                for r in 0..h {
                    for i in 0..w4 {
                        let value = Vec4::new(BASE * 10.0, 1.0 + BASE * 10.0,
                                              2.0 + BASE * 10.0, 3.0 + BASE * 10.0);
                        m.as_vec4_mut(r)[i] = value;
                        test_row_vec4(&m, r, i, value);
                    }
                }

                for r in 0..h {
                    for i in 0..w4 {
                        let value = Vec4::new(BASE * 3.0, 1.0 + BASE * 3.0,
                                              2.0 + BASE * 3.0, 3.0 + BASE * 3.0);
                        m.as_vec4_slice_mut(r, ..)[i] = value;
                        test_row_vec4(&m, r, i, value);
                    }
                }

                for r in 0..h {
                    for i in 0..w4 + 1 {
                        let value = Vec4::new(BASE * 6.0, 1.0 + BASE * 6.0,
                                              2.0 + BASE * 6.0, 3.0 + BASE * 6.0);
                        let s = m.as_vec4_slice_mut(r, i..);
                        assert_eq!(s.len(), w4 - i);
                        if i < w4 {
                            s[0] = value;
                            test_row_vec4(&m, r, i, value);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w4 + 1 {
                        let value = Vec4::new(BASE * 7.0, 1.0 + BASE * 7.0,
                                              2.0 + BASE * 7.0, 3.0 + BASE * 7.0);
                        let s = m.as_vec4_slice_mut(r, ..i);
                        assert_eq!(s.len(), i);
                        if i > 0 {
                            s[i - 1] = value;
                            test_row_vec4(&m, r, i - 1, value);
                        }
                    }
                }

                for r in 0..h {
                    for i in 0..w4 {
                        let value = Vec4::new(BASE * 8.0, 1.0 + BASE * 8.0,
                                              2.0 + BASE * 8.0, 3.0 + BASE * 8.0);
                        let s = m.as_vec4_slice_mut(r, i..i + 1);
                        assert_eq!(s.len(), 1);
                        s[0] = value;
                        test_row_vec4(&m, r, i, value);
                    }
                }
            }
        }
    }

    #[test]
    fn test_aligned_vec() {
        const BASE: f32 = 123.0;
        for l in 0..256 {
            let mut v = AlignedVec::new(l);
            // Fill in vector.
            let s = v.as_f32_slice_mut(..);
            for i in 0..l {
                s[i] = i as f32 + BASE;
            }

            // Read through f32 slice.
            let s = v.as_f32();
            assert_eq!(s.len(), l);
            for i in 0..l {
                assert_eq!(s[i], i as f32 + BASE);
            }

            let s = v.as_f32_slice(..);
            assert_eq!(s.len(), l);
            for i in 0..l {
                assert_eq!(s[i], i as f32 + BASE);
            }

            for i in 0..l + 1 {
                let s = v.as_f32_slice(i..);
                assert_eq!(s.len(), l - i);
                if i < l {
                    assert_eq!(s[0], i as f32 + BASE);
                }
            }

            for i in 0..l + 1 {
                let s = v.as_f32_slice(..i);
                assert_eq!(s.len(), i);
                if i > 0 {
                    assert_eq!(s[i - 1], (i - 1) as f32 + BASE);
                }
            }

            for i in 0..l {
                let s = v.as_f32_slice(i..i + 1);
                assert_eq!(s.len(), 1);
                assert_eq!(s[0], i as f32 + BASE);
            }

            if l > 0 {
                for i in 0..l - 1 {
                    let s = v.as_f32_slice(i..i + 2);
                    assert_eq!(s.len(), 2);
                    assert_eq!(s[0], i as f32 + BASE);
                    assert_eq!(s[1], (i + 1) as f32 + BASE);
                }
            }

            // Read through f32 index
            for i in 0..l {
                assert_eq!(v[i], i as f32 + BASE);
            }

            // Iterate
            for (i, &value) in v.iter().enumerate() {
                assert_eq!(value, i as f32 + BASE);
            }

            // Read through Vec4 slice.
            let l4 = (l + 3) / 4;
            let s = v.as_vec4();
            assert_eq!(s.len(), l4);
            for i in 0..l4 {
                assert_eq!(s[i], get_vec4(i, l, BASE));
            }

            let s = v.as_vec4_slice(..);
            assert_eq!(s.len(), l4);
            for i in 0..l4 {
                assert_eq!(s[i], get_vec4(i, l, BASE));
            }

            for i in 0..l4 + 1 {
                let s = v.as_vec4_slice(i..);
                assert_eq!(s.len(), l4 - i);
                if i < l4 {
                    assert_eq!(s[0], get_vec4(i, l, BASE));
                }
            }

            for i in 0..l4 + 1 {
                let s = v.as_vec4_slice(..i);
                assert_eq!(s.len(), i);
                if i > 0 {
                    assert_eq!(s[i - 1], get_vec4(i - 1, l, BASE));
                }
            }

            for i in 0..l4 {
                let s = v.as_vec4_slice(i..i + 1);
                assert_eq!(s.len(), 1);
                assert_eq!(s[0], get_vec4(i, l, BASE));
            }

            if l4 > 0 {
                for i in 0..l4 - 1 {
                    let s = v.as_vec4_slice(i..i + 2);
                    assert_eq!(s.len(), 2);
                    assert_eq!(s[0], get_vec4(i, l, BASE));
                    assert_eq!(s[1], get_vec4(i + 1, l, BASE));
                }
            }

            // Write through f32 slice.
            for i in 0..l {
                let value = i as f32 + BASE * 10.0;
                v.as_f32_mut()[i] = value;
                assert_eq!(v.as_f32_slice(i..i + 1)[0], value)
            }

            for i in 0..l {
                let value = i as f32 + BASE * 2.0;
                v.as_f32_slice_mut(..)[i] = value;
                assert_eq!(v.as_f32_slice(i..i + 1)[0], value)
            }

            for i in 0..l + 1 {
                let s = v.as_f32_slice_mut(i..);
                assert_eq!(s.len(), l - i);
                if i < l {
                    let value = i as f32 + BASE * 3.0;
                    s[0] = value;
                    assert_eq!(v.as_f32_slice(i..i + 1)[0], value);
                }
            }

            for i in 0..l + 1 {
                let s = v.as_f32_slice_mut(..i);
                assert_eq!(s.len(), i);
                if i > 0 {
                    let value = i as f32 + BASE * 4.0;
                    s[i - 1] = value;
                    assert_eq!(v.as_f32_slice(i - 1..i)[0], value);
                }
            }

            for i in 0..l {
                let s = v.as_f32_slice_mut(i..i + 1);
                assert_eq!(s.len(), 1);
                let value = i as f32 + BASE * 5.0;
                s[0] = value;
                assert_eq!(v.as_f32_slice(i..i + 1)[0], value);
            }

            // Write through iterator
            for (i, value) in v.iter_mut().enumerate() {
                *value = i as f32 + BASE * 6.0;
            }
            for i in 0..l {
                assert_eq!(v[i], i as f32 + BASE * 6.0);
            }

            // Write through Vec4 slice.
            for i in 0..l4 {
                let value = Vec4::new(BASE * 10.0, 1.0 + BASE * 10.0,
                                      2.0 + BASE * 10.0, 3.0 + BASE * 10.0);
                v.as_vec4_mut()[i] = value;
                test_vec4(&v, i, value);
            }

            for i in 0..l4 {
                let value = Vec4::new(BASE * 3.0, 1.0 + BASE * 3.0,
                                      2.0 + BASE * 3.0, 3.0 + BASE * 3.0);
                v.as_vec4_slice_mut(..)[i] = value;
                test_vec4(&v, i, value);
            }

            for i in 0..l4 + 1 {
                let value = Vec4::new(BASE * 4.0, 1.0 + BASE * 4.0,
                                      2.0 + BASE * 4.0, 3.0 + BASE * 4.0);
                let s = v.as_vec4_slice_mut(i..);
                assert_eq!(s.len(), l4 - i);
                if i < l4 {
                    s[0] = value;
                    test_vec4(&v, i, value);
                }
            }

            for i in 0..l4 + 1 {
                let value = Vec4::new(BASE * 5.0, 1.0 + BASE * 5.0,
                                      2.0 + BASE * 5.0, 3.0 + BASE * 5.0);
                let s = v.as_vec4_slice_mut(..i);
                assert_eq!(s.len(), i);
                if i > 0 {
                    s[i - 1] = value;
                    test_vec4(&v, i - 1, value);
                }
            }

            for i in 0..l4 {
                let value = Vec4::new(BASE * 6.0, 1.0 + BASE * 6.0,
                                      2.0 + BASE * 6.0, 3.0 + BASE * 6.0);
                let s = v.as_vec4_slice_mut(i..i + 1);
                assert_eq!(s.len(), 1);
                s[0] = value;
                test_vec4(&v, i, value);
            }
        }
    }

    fn get_vec4(i: usize, size: usize, base: f32) -> Vec4 {
        match size - i * 4 {
            1 => vec4((i * 4) as f32 + base, 0.0, 0.0, 0.0),
            2 => vec4((i * 4) as f32 + base, (i * 4 + 1) as f32 + base, 0.0, 0.0),
            3 => vec4((i * 4) as f32 + base, (i * 4 + 1) as f32 + base,
                      (i * 4 + 2) as f32 + base, 0.0),
            _ => vec4((i * 4) as f32 + base, (i * 4 + 1) as f32 + base,
                      (i * 4 + 2) as f32 + base, (i * 4 + 3) as f32 + base)
        }
    }

    fn get_row_vec4(r: usize, i: usize, size: usize, base: f32) -> Vec4 {
        match size - i * 4 {
            1 => vec4((i * 4 + r * 5) as f32 + base, 0.0, 0.0, 0.0),
            2 => vec4((i * 4 + r * 5) as f32 + base, (i * 4 + r * 5 + 1) as f32 + base, 0.0, 0.0),
            3 => vec4((i * 4 + r * 5) as f32 + base, (i * 4 + r * 5 + 1) as f32 + base,
                      (i * 4 + r * 5 + 2) as f32 + base, 0.0),
            _ => vec4((i * 4 + r * 5) as f32 + base, (i * 4 + r * 5 + 1) as f32 + base,
                      (i * 4 + r * 5 + 2) as f32 + base, (i * 4 + r * 5 + 3) as f32 + base)
        }
    }

    fn test_vec4(v: &AlignedVec, i: usize, value: Vec4) {
        assert_eq!(v.as_f32_slice(i * 4..i * 4 + 1)[0], value.x);
        if i * 4 + 4 <= v.len() {
            assert_eq!(v.as_f32_slice(i * 4 + 1..i * 4 + 2)[0], value.y);
            assert_eq!(v.as_f32_slice(i * 4 + 2..i * 4 + 3)[0], value.z);
            assert_eq!(v.as_f32_slice(i * 4 + 3..i * 4 + 4)[0], value.w);
        }
    }

    fn test_row_vec4(m: &AlignedMatrix, r: usize, i: usize, value: Vec4) {
        assert_eq!(m.as_f32_slice(r, i * 4..i * 4 + 1)[0], value.x);
        if i * 4 + 4 <= m.width() {
            assert_eq!(m.as_f32_slice(r, i * 4 + 1..i * 4 + 2)[0], value.y);
            assert_eq!(m.as_f32_slice(r, i * 4 + 2..i * 4 + 3)[0], value.z);
            assert_eq!(m.as_f32_slice(r, i * 4 + 3..i * 4 + 4)[0], value.w);
        }
    }
}
