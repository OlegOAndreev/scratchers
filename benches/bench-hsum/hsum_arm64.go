//go:build arm64

package main

import "unsafe"

const avx2Enabled = false
const neonEnabled = true

func HorizontalSumNeon(buf []byte, from int, to int) uint64 {
	if to == from {
		return 0
	}
	return horizontalSumNeonAsm(unsafe.SliceData(buf[from:]), to-from)
}

func HorizontalSumNeonV2(buf []byte, from int, to int) uint64 {
	if to == from {
		return 0
	}
	return horizontalSumNeonAsmV2(unsafe.SliceData(buf[from:]), to-from)
}

func horizontalSumNeonAsm(ptr *byte, size int) uint64
func horizontalSumNeonAsmV2(ptr *byte, size int) uint64

func HorizontalSumAvx2(buf []byte, from int, to int) uint64 {
	panic("avx2 not enabled")
}

func HorizontalSumAvx2V2(buf []byte, from int, to int) uint64 {
	panic("avx2 not enabled")
}
