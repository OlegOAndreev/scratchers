//go:build amd64

package main

import "unsafe"

const avx2Enabled = true
const neonEnabled = false

func HorizontalSumAvx2(buf []byte, from int, to int) uint64 {
	if to == from {
		return 0
	}
	return horizontalSumAvx2Asm(unsafe.SliceData(buf[from:]), to-from)
}

func HorizontalSumAvx2V2(buf []byte, from int, to int) uint64 {
	if to == from {
		return 0
	}
	return horizontalSumAvx2AsmV2(unsafe.SliceData(buf[from:]), to-from)
}

func horizontalSumAvx2Asm(ptr *byte, size int) uint64
func horizontalSumAvx2AsmV2(ptr *byte, size int) uint64

func HorizontalSumNeon(buf []byte, from int, to int) uint64 {
	panic("neon not enabled")
}

func HorizontalSumNeonV2(buf []byte, from int, to int) uint64 {
	panic("neon not enabled")
}
