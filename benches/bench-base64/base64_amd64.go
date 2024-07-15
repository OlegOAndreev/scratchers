//go:build amd64

package main

const avx2Enabled = true
const neonEnabled = false

func Base64EncodeAvx2(dst []byte, src []byte) []byte {
	if len(src) == 0 {
		return dst
	}

	resultLen := (len(src) + 2) / 3 * 4
	if resultLen < 4 {
		panic("oh no")
	}
	result := appendBytes(dst, resultLen)
	// I really don't want to do the div -> mul conversion for assembly
	numTriples := len(src) / 3
	remainder := len(src) % 3
	base64EncodeAvx2Asm(&result[len(dst)], &src[0], &encodeMap[0], numTriples, remainder)
	return result
}

func base64EncodeAvx2Asm(dst *byte, src *byte, encodeMap *byte, numTriples int, remainder int)
