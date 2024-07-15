//go:build amd64 || arm64

package main

import (
	"errors"
	"fmt"
	"unsafe"
)

var encodeMap = [64]byte{
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
	'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
	'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
	'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
}

const padding = '='

// Base64Encode appends base64-encoded src to dst.
func Base64Encode(dst []byte, src []byte) []byte {
	if len(src) == 0 {
		return dst
	}

	resultLen := (len(src) + 2) / 3 * 4
	result := appendBytes(dst, resultLen)
	srcPtr := unsafe.Pointer(&src[0])
	dstPtr := unsafe.Pointer(&result[len(dst)])
	encodePtr := unsafe.Pointer(&encodeMap[0])
	numTriplets := len(src) / 3
	for i := 0; i < numTriplets; i++ {
		word := uint32(*(*byte)(srcPtr))<<16 | uint32(*(*byte)(unsafe.Add(srcPtr, 1)))<<8 |
			uint32(*(*byte)(unsafe.Add(srcPtr, 2)))
		srcPtr = unsafe.Add(srcPtr, 3)
		// Assume little endian
		*(*uint32)(dstPtr) = uint32(*(*byte)(unsafe.Add(encodePtr, (word>>18)&0x3F))) |
			uint32(*(*byte)(unsafe.Add(encodePtr, (word>>12)&0x3F)))<<8 |
			uint32(*(*byte)(unsafe.Add(encodePtr, (word>>6)&0x3F)))<<16 |
			uint32(*(*byte)(unsafe.Add(encodePtr, word&0x3F)))<<24
		dstPtr = unsafe.Add(dstPtr, 4)
	}
	switch len(src) % 3 {
	case 1:
		word := uint32(*(*byte)(srcPtr)) << 16
		*(*uint32)(dstPtr) = uint32(*(*byte)(unsafe.Add(encodePtr, (word>>18)&0x3F))) |
			uint32(*(*byte)(unsafe.Add(encodePtr, (word>>12)&0x3F)))<<8 |
			uint32(padding)<<16 |
			uint32(padding)<<24
	case 2:
		word := uint32(*(*byte)(srcPtr))<<16 | uint32(*(*byte)(unsafe.Add(srcPtr, 1)))<<8
		*(*uint32)(dstPtr) = uint32(*(*byte)(unsafe.Add(encodePtr, (word>>18)&0x3F))) |
			uint32(*(*byte)(unsafe.Add(encodePtr, (word>>12)&0x3F)))<<8 |
			uint32(*(*byte)(unsafe.Add(encodePtr, (word>>6)&0x3F)))<<16 |
			uint32(padding)<<24
	}

	return result
}

var decodeMap = [256]byte{
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,
	0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
	0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
}

// Base64Decode appends base64-decoded src to dst. Minimal validation is performed: malformed input may result
// in data appended to dst (malformed input never leads to undefined behavior).
func Base64Decode(dst []byte, src []byte) ([]byte, error) {
	if len(src) == 0 {
		return dst, nil
	}

	if len(src)%4 != 0 {
		return nil, fmt.Errorf("wrong base64 length: %d", len(src))
	}

	numPadding := 0
	if src[len(src)-1] == padding {
		numPadding++
		if src[len(src)-2] == padding {
			numPadding++
		}
	}
	resultLen := len(src)/4*3 - numPadding
	result := appendBytes(dst, resultLen)
	srcPtr := unsafe.Pointer(&src[0])
	dstPtr := unsafe.Pointer(&result[len(dst)])
	decodePtr := unsafe.Pointer(&decodeMap[0])
	numQuads := len(src) / 4
	if numPadding > 0 {
		numQuads--
	}
	// We sign-extend elements from decodeMap when we convert int8 -> int32 and use the upper byte in word to check
	// if we got 0xFF for any of the elements.
	var allWords int32
	for i := 0; i < numQuads; i++ {
		sym := *(*uint32)(srcPtr)
		srcPtr = unsafe.Add(srcPtr, 4)
		word := int32(*(*int8)(unsafe.Add(decodePtr, sym&0xFF)))<<18 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>8)&0xFF)))<<12 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>16)&0xFF)))<<6 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>24)&0xFF)))
		allWords |= word
		*(*byte)(dstPtr) = byte((word >> 16) & 0xFF)
		*(*byte)(unsafe.Add(dstPtr, 1)) = byte((word >> 8) & 0xFF)
		*(*byte)(unsafe.Add(dstPtr, 2)) = byte(word & 0xFF)
		dstPtr = unsafe.Add(dstPtr, 3)
	}
	switch numPadding {
	case 1:
		sym := *(*uint32)(srcPtr)
		word := int32(*(*int8)(unsafe.Add(decodePtr, sym&0xFF)))<<18 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>8)&0xFF)))<<12 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>16)&0xFF)))<<6
		allWords |= word
		*(*int8)(dstPtr) = int8((word >> 16) & 0xFF)
		*(*int8)(unsafe.Add(dstPtr, 1)) = int8((word >> 8) & 0xFF)
	case 2:
		sym := *(*uint32)(srcPtr)
		word := int32(*(*int8)(unsafe.Add(decodePtr, sym&0xFF)))<<18 |
			int32(*(*int8)(unsafe.Add(decodePtr, (sym>>8)&0xFF)))<<12
		allWords |= word
		*(*int8)(dstPtr) = int8((word >> 16) & 0xFF)
	}

	// Check if the upper byte of allWords is 0xFF
	if allWords>>24 == -1 {
		return nil, errors.New("malformed base64")
	}

	return result, nil
}

func appendBytes(dst []byte, n int) []byte {
	newLen := len(dst) + n
	if newLen <= cap(dst) {
		return dst[:newLen]
	}
	// If only we could call bytealg.MakeNoZero here
	newDst := make([]byte, newLen)
	copy(newDst, dst)
	return newDst
}
