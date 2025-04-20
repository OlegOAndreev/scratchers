package benches

import (
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"fmt"
	"io"
	mathRand "math/rand"
	"os"
	"testing"
)

func BenchmarkGzip(b *testing.B) {
	const messageSize = 128 * 1024 * 1024
	lowEntropyMessage, err := makeNonRandomSlice(messageSize)
	if err != nil {
		b.Fatal(err)
	}
	limitedByteMessage := makeLimitedByteSlice(messageSize, 4)
	highEntropyMessage := make([]byte, messageSize)
	_, err = rand.Read(highEntropyMessage)
	if err != nil {
		b.Fatal(err)
	}
	_ = os.WriteFile("low.bin", lowEntropyMessage, 0644)
	_ = os.WriteFile("limited.bin", limitedByteMessage, 0644)
	_ = os.WriteFile("high.bin", highEntropyMessage, 0644)

	b.Run("gzip low entropy", func(b *testing.B) {
		b.SetBytes(int64(len(lowEntropyMessage)))
		for i := 0; i < b.N; i++ {
			if err := gzipDiscard(lowEntropyMessage); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("gzip limited bytes", func(b *testing.B) {
		b.SetBytes(int64(len(limitedByteMessage)))
		for i := 0; i < b.N; i++ {
			if err := gzipDiscard(limitedByteMessage); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("gzip high entropy", func(b *testing.B) {
		b.SetBytes(int64(len(highEntropyMessage)))
		for i := 0; i < b.N; i++ {
			if err := gzipDiscard(highEntropyMessage); err != nil {
				b.Fatal(err)
			}
		}
	})

	gzippedLowEntropy, err := gzipSlice(lowEntropyMessage)
	if err != nil {
		b.Fatal(err)
	}
	gzippedLimitedBytes, err := gzipSlice(limitedByteMessage)
	if err != nil {
		b.Fatal(err)
	}
	gzippedHighEntropy, err := gzipSlice(highEntropyMessage)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("gunzip low entropy", func(b *testing.B) {
		b.SetBytes(int64(len(lowEntropyMessage)))
		buf := make([]byte, 128*1024)
		for i := 0; i < b.N; i++ {
			if err := gunzipDiscard(gzippedLowEntropy, buf); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("gunzip limited bytes", func(b *testing.B) {
		b.SetBytes(int64(len(limitedByteMessage)))
		buf := make([]byte, 128*1024)
		for i := 0; i < b.N; i++ {
			if err := gunzipDiscard(gzippedLimitedBytes, buf); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("gunzip high entropy", func(b *testing.B) {
		b.SetBytes(int64(len(highEntropyMessage)))
		buf := make([]byte, 128*1024)
		for i := 0; i < b.N; i++ {
			if err := gunzipDiscard(gzippedHighEntropy, buf); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func gzipSlice(src []byte) ([]byte, error) {
	var buf bytes.Buffer
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(src); err != nil {
		return nil, fmt.Errorf("write failed: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("close failed: %w", err)
	}
	return buf.Bytes(), nil
}

func gzipDiscard(src []byte) error {
	w := gzip.NewWriter(io.Discard)
	if _, err := w.Write(src); err != nil {
		return fmt.Errorf("write failed: %w", err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("close failed: %w", err)
	}
	return nil
}

func gunzipDiscard(src []byte, buf []byte) error {
	r, err := gzip.NewReader(bytes.NewReader(src))
	if err != nil {
		return fmt.Errorf("new reader failed: %w", err)
	}
	if _, err := io.CopyBuffer(io.Discard, r, buf); err != nil {
		return fmt.Errorf("copy failed: %w", err)
	}
	if err := r.Close(); err != nil {
		return fmt.Errorf("close failed: %w", err)
	}
	return nil
}

func makeNonRandomSlice(n int) ([]byte, error) {
	const numWords = 100
	const wordLen = 10
	result := make([]byte, 0, n)
	words := make([][]byte, numWords)
	for i := 0; i < numWords; i++ {
		words[i] = make([]byte, wordLen)
		_, err := rand.Read(words[i])
		if err != nil {
			return nil, err
		}
	}
	for len(result) < n {
		result = append(result, words[mathRand.Intn(numWords)]...)
	}
	return result[:n], nil
}

func makeLimitedByteSlice(n int, limit int) []byte {
	result := make([]byte, n)
	for i := 0; i < n; i++ {
		result[i] = byte(mathRand.Intn(limit))
	}
	return result
}
