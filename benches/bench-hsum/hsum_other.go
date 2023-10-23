//go:build !amd64

package main

const avx2Enabled = false

func HorizontalSumAvx2(buf []byte, from int, to int) uint64 {
	panic("avx2 not enabled")
}
