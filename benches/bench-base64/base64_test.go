package main

import (
	"bytes"
	"encoding/base64"
	"math/rand/v2"
	"testing"
)

func TestBase64Encode(t *testing.T) {
	existingSlice := []byte{1, 2, 3}
	t.Run("base", func(t *testing.T) {
		testBase64(t, func(src []byte) []byte {
			return Base64Encode(nil, src)
		}, func(src []byte) []byte {
			result, err := Base64Decode(nil, src)
			if err != nil {
				t.Errorf("failed to decode %s: %v", string(src), err)
			}
			return result
		})
	})
	t.Run("with prefix", func(t *testing.T) {
		testBase64(t, func(src []byte) []byte {
			result := Base64Encode(existingSlice, src)
			return result[len(existingSlice):]
		}, func(src []byte) []byte {
			result, err := Base64Decode(existingSlice, src)
			if err != nil {
				t.Errorf("failed to decode %s: %v", string(src), err)
			}
			return result[len(existingSlice):]
		})
	})
}

func testBase64(t *testing.T, encode func(src []byte) []byte, decode func(src []byte) []byte) {
	const maxLen = 100
	for l := 0; l < maxLen; l++ {
		buf := make([]byte, l)
		// Single byte patterns
		for b := 0; b < 256; b++ {
			for i := 0; i < l; i++ {
				buf[i] = byte(b)
			}
			result := encode(buf)
			_ = result
			if !bytes.Equal(result, base64.StdEncoding.AppendEncode(nil, buf)) {
				t.Errorf("failed encode for input %v", buf)
			}
			decodeResult := decode(result)
			if !bytes.Equal(decodeResult, buf) {
				t.Errorf("failed decode for input %v", buf)
			}
		}

		// Generate random buffers
		rnd := rand.New(fixedRandSource{})
		for j := 0; j < 10000; j++ {
			for i := 0; i < l; i++ {
				buf[i] = byte(rnd.IntN(256))
			}
			result := encode(buf)
			_ = result
			if !bytes.Equal(result, base64.StdEncoding.AppendEncode(nil, buf)) {
				t.Errorf("failed encode for input %v", buf)
			}
			decodeResult := decode(result)
			if !bytes.Equal(decodeResult, buf) {
				t.Errorf("failed decode for input %v", buf)
			}
		}
	}
}

func TestBase64DecodeError(t *testing.T) {
	_, err := Base64Decode(nil, []byte("aBcd"))
	if err != nil {
		t.Errorf("got unexpected error: %v\n", err)
	}
	tests := []string{"*", "*B==", "B*==", "*cD=", "c*D=", "cD*=", "*123", "1*23", "12*3", "123*"}
	for _, test := range tests {
		_, err = Base64Decode(nil, []byte(test))
		if err == nil {
			t.Errorf("expected error for %s\n", test)
		}
	}
}

func TestBase64Avx2Encode(t *testing.T) {
	if !avx2Enabled {
		t.Skip("AVX2 not supported")
	}

	existingSlice := []byte{1, 2, 3}
	t.Run("base", func(t *testing.T) {
		testBase64(t, func(src []byte) []byte {
			return Base64EncodeAvx2(nil, src)
		}, func(src []byte) []byte {
			result, err := Base64Decode(nil, src)
			if err != nil {
				t.Errorf("failed to decode %s: %v", string(src), err)
			}
			return result
		})
	})
	t.Run("with prefix", func(t *testing.T) {
		testBase64(t, func(src []byte) []byte {
			result := Base64EncodeAvx2(existingSlice, src)
			return result[len(existingSlice):]
		}, func(src []byte) []byte {
			result, err := Base64Decode(existingSlice, src)
			if err != nil {
				t.Errorf("failed to decode %s: %v", string(src), err)
			}
			return result[len(existingSlice):]
		})
	})
}
