//go:build amd64

package main

import "unsafe"

const avx2Enabled = true

func HorizontalSumAvx2(buf []byte, from int, to int) uint64 {
	if to == from {
		return 0
	}
	return horizontalSumAvx2Asm(unsafe.SliceData(buf[from:]), to-from)
}

func horizontalSumAvx2Asm(ptr *byte, size int) uint64
