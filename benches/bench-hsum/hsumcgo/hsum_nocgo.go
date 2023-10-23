//go:build !cgo

package hsumcgo

const Enabled = false

func HorizontalAddCNaive(buf []byte, from int, to int) uint64 {
	panic("CGo disabled")
}

func HorizontalSumAvx2C(buf []byte, from int, to int) uint64 {
	panic("CGo disabled")
}

func Avx2Enabled() bool {
	panic("CGo disabled")
}
