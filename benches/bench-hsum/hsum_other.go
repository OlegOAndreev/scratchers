//go:build !amd64 && !arm64

package main

const avx2Enabled = false
const neonEnabled = false

func HorizontalSumAvx2(buf []byte, from int, to int) uint64 {
	panic("avx2 not enabled")
}

func HorizontalSumAvx2V2(buf []byte, from int, to int) uint64 {
	panic("avx2 not enabled")
}

func HorizontalSumNeon(buf []byte, from int, to int) uint64 {
	panic("neon not enabled")
}

func HorizontalSumNeonV2(buf []byte, from int, to int) uint64 {
	panic("neon not enabled")
}
