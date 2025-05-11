package main

import (
	"unsafe"
)

func HorizontalSumNaive(buf []byte, from int, to int) uint64 {
	var ret uint64
	for i := from; i < to; i++ {
		ret += uint64(buf[i])
	}
	return ret
}

func HorizontalSumUnroll(buf []byte, from int, to int) uint64 {
	var r0, r1, r2, r3 uint64
	unrolledTo := (to-from) & ^7 + from
	for i := from; i < unrolledTo; i += 8 {
		r0 += uint64(buf[i])
		r1 += uint64(buf[i+1])
		r2 += uint64(buf[i+2])
		r3 += uint64(buf[i+3])
		r0 += uint64(buf[i+4])
		r1 += uint64(buf[i+5])
		r2 += uint64(buf[i+6])
		r3 += uint64(buf[i+7])
	}
	ret := r0 + r1 + r2 + r3
	for i := unrolledTo; i < to; i++ {
		ret += uint64(buf[i])
	}
	return ret
}

func HorizontalSumInt8ToInt16(buf []byte, from int, to int) uint64 {
	// Unaligned pointers are ok in Go: https://github.com/golang/go/issues/37298, don't bother aligning `from`.
	unrolledLen := (to - from) & ^7
	unrolledTo := from + unrolledLen
	intBuf := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(buf[from:unrolledTo]))), unrolledLen/8)

	var ret uint64
	i := 0
	for {
		// Accumulators for even and odd bytes in each uint64.
		var accumLo uint64
		var accumHi uint64

		if i >= len(intBuf)-16 {
			for j := i; j < len(intBuf); j++ {
				accumLo += intBuf[j] & 0x00FF00FF00FF00FF
				accumHi += (intBuf[j] >> 8) & 0x00FF00FF00FF00FF
			}
			ret += hsumInt16(accumLo)
			ret += hsumInt16(accumHi)
			break
		}

		// Don't blow out the I-cache by over-unrolling.
		for j := i; j < i+16; j += 4 {
			accumLo += intBuf[j] & 0x00FF00FF00FF00FF
			accumHi += (intBuf[j] >> 8) & 0x00FF00FF00FF00FF
			accumLo += intBuf[j+1] & 0x00FF00FF00FF00FF
			accumHi += (intBuf[j+1] >> 8) & 0x00FF00FF00FF00FF
			accumLo += intBuf[j+2] & 0x00FF00FF00FF00FF
			accumHi += (intBuf[j+2] >> 8) & 0x00FF00FF00FF00FF
			accumLo += intBuf[j+3] & 0x00FF00FF00FF00FF
			accumHi += (intBuf[j+3] >> 8) & 0x00FF00FF00FF00FF
		}
		i += 16
		ret += hsumInt16(accumLo)
		ret += hsumInt16(accumHi)
	}

	for i := unrolledTo; i < to; i++ {
		ret += uint64(buf[i])
	}
	return ret
}

func HorizontalSumInt8ToInt16Unsafe(buf []byte, from int, to int) uint64 {
	var ret uint64
	if to-from < 8 {
		for i := from; i < to; i++ {
			ret += uint64(buf[i])
		}
		return ret
	}

	// Unaligned pointers are ok in Go: https://github.com/golang/go/issues/37298, don't bother aligning `from`.
	intPtr := (*uint64)(unsafe.Pointer(unsafe.SliceData(buf[from:])))
	n := to - from
	for {
		// Accumulators for 4-bit nibbles in each byte.
		var accumLo uint64
		var accumHi uint64

		// We can add up to 16 nibbles (x8 bytes in uint64) before we overflow the byte.
		if n < 128 {
			for n >= 8 {
				accumLo += *intPtr & 0x00FF00FF00FF00FF
				accumHi += (*intPtr >> 8) & 0x00FF00FF00FF00FF
				intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
				n -= 8
			}
			// Add the remainder. The buffer is guaranteed to be 8 bytes or larger.
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), n-8))
			// Zero top bytes.
			maskBits := (8 - n) * 8
			remainder := (*intPtr >> maskBits) << maskBits
			accumLo += remainder & 0x00FF00FF00FF00FF
			accumHi += (remainder >> 8) & 0x00FF00FF00FF00FF

			ret += hsumInt16(accumLo)
			ret += hsumInt16(accumHi)
			break
		}

		// Don't blow out the I-cache by over-unrolling.
		for j := 0; j < 4; j++ {
			accumLo += *intPtr & 0x00FF00FF00FF00FF
			accumHi += (*intPtr >> 8) & 0x00FF00FF00FF00FF
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x00FF00FF00FF00FF
			accumHi += (*intPtr >> 8) & 0x00FF00FF00FF00FF
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x00FF00FF00FF00FF
			accumHi += (*intPtr >> 8) & 0x00FF00FF00FF00FF
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x00FF00FF00FF00FF
			accumHi += (*intPtr >> 8) & 0x00FF00FF00FF00FF
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
		}
		n -= 128
		ret += hsumInt16(accumLo)
		ret += hsumInt16(accumHi)
	}

	return ret
}

func HorizontalSumInt4ToInt8(buf []byte, from int, to int) uint64 {
	// Unaligned pointers are ok in Go: https://github.com/golang/go/issues/37298, don't bother aligning `from`.
	unrolledLen := (to - from) & ^7
	unrolledTo := from + unrolledLen
	intBuf := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(buf[from:unrolledTo]))), unrolledLen/8)

	var ret uint64
	i := 0
	for {
		// Accumulators for 4-bit nibbles in each byte.
		var accumLo uint64
		var accumHi uint64

		// We can add up to 16 nibbles before we overflow the byte.
		if i >= len(intBuf)-16 {
			for j := i; j < len(intBuf); j++ {
				accumLo += intBuf[j] & 0x0F0F0F0F0F0F0F0F
				accumHi += (intBuf[j] >> 4) & 0x0F0F0F0F0F0F0F0F
			}
			ret += hsumInt8(accumLo)
			ret += hsumInt8(accumHi) << 4
			break
		}

		// Don't blow out the I-cache by over-unrolling.
		for j := i; j < i+16; j += 4 {
			accumLo += intBuf[j] & 0x0F0F0F0F0F0F0F0F
			accumHi += (intBuf[j] >> 4) & 0x0F0F0F0F0F0F0F0F
			accumLo += intBuf[j+1] & 0x0F0F0F0F0F0F0F0F
			accumHi += (intBuf[j+1] >> 4) & 0x0F0F0F0F0F0F0F0F
			accumLo += intBuf[j+2] & 0x0F0F0F0F0F0F0F0F
			accumHi += (intBuf[j+2] >> 4) & 0x0F0F0F0F0F0F0F0F
			accumLo += intBuf[j+3] & 0x0F0F0F0F0F0F0F0F
			accumHi += (intBuf[j+3] >> 4) & 0x0F0F0F0F0F0F0F0F
		}
		i += 16
		ret += hsumInt8(accumLo)
		ret += hsumInt8(accumHi) << 4
	}

	for i := unrolledTo; i < to; i++ {
		ret += uint64(buf[i])
	}
	return ret
}

func HorizontalSumInt4ToInt8Unsafe(buf []byte, from int, to int) uint64 {
	var ret uint64
	if to-from < 8 {
		for i := from; i < to; i++ {
			ret += uint64(buf[i])
		}
		return ret
	}

	// Unaligned pointers are ok in Go: https://github.com/golang/go/issues/37298, don't bother aligning `from`.
	intPtr := (*uint64)(unsafe.Pointer(unsafe.SliceData(buf[from:])))
	n := to - from
	for {
		// Accumulators for 4-bit nibbles in each byte.
		var accumLo uint64
		var accumHi uint64

		// We can add up to 16 nibbles (x8 bytes in uint64) before we overflow the byte.
		if n < 128 {
			for n >= 8 {
				accumLo += *intPtr & 0x0F0F0F0F0F0F0F0F
				accumHi += (*intPtr >> 4) & 0x0F0F0F0F0F0F0F0F
				intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
				n -= 8
			}
			// Add the remainder. The buffer is guaranteed to be 8 bytes or larger.
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), n-8))
			// Zero top bytes.
			maskBits := (8 - n) * 8
			remainder := (*intPtr >> maskBits) << maskBits
			accumLo += remainder & 0x0F0F0F0F0F0F0F0F
			accumHi += (remainder >> 4) & 0x0F0F0F0F0F0F0F0F

			ret += hsumInt8(accumLo)
			ret += hsumInt8(accumHi) << 4
			break
		}

		// Don't blow out the I-cache by over-unrolling.
		for j := 0; j < 4; j++ {
			accumLo += *intPtr & 0x0F0F0F0F0F0F0F0F
			accumHi += (*intPtr >> 4) & 0x0F0F0F0F0F0F0F0F
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x0F0F0F0F0F0F0F0F
			accumHi += (*intPtr >> 4) & 0x0F0F0F0F0F0F0F0F
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x0F0F0F0F0F0F0F0F
			accumHi += (*intPtr >> 4) & 0x0F0F0F0F0F0F0F0F
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
			accumLo += *intPtr & 0x0F0F0F0F0F0F0F0F
			accumHi += (*intPtr >> 4) & 0x0F0F0F0F0F0F0F0F
			intPtr = (*uint64)(unsafe.Add(unsafe.Pointer(intPtr), 8))
		}
		n -= 128
		ret += hsumInt8(accumLo)
		ret += hsumInt8(accumHi) << 4
	}

	return ret
}

func hsumInt8(a uint64) uint64 {
	// Do pairwise sum of 8-bit, 16-bit and 32-bit pairs.
	a = a&0x00FF00FF00FF00FF + (a>>8)&0x00FF00FF00FF00FF
	a = a&0x0000FFFF0000FFFF + (a>>16)&0x0000FFFF0000FFFF
	return (a >> 32) + a&0xFFFFFFFF
}

func hsumInt16(a uint64) uint64 {
	// Do pairwise sum of 16-bit and 32-bit pairs.
	a = a&0x0000FFFF0000FFFF + (a>>16)&0x0000FFFF0000FFFF
	return (a >> 32) + a&0xFFFFFFFF
}
