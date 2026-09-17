// Benchmarks three algorithms for locating a needle inside one column of a large TSV file.
package main

import (
	"bytes"
	"encoding/csv"
	"io"
	"math/rand/v2"
	"strings"
	"testing"
	"time"
)

const (
	numColumns   = 20
	maxFieldLen  = 20
	minNeedleLen = 10
)

func BenchmarkTSVSearch(b *testing.B) {
	const datasetSize = 1 << 31 // 2Gb
	startTime := time.Now()
	ds := genDataset(datasetSize)
	b.Logf("Generated dataset size %d, lines %d, needle %q, needle_col %d, needle_line %d in %v",
		len(ds.data), ds.lines, ds.needle, ds.needleColumn, len(ds.needleLine), time.Since(startTime))

	for _, sm := range searchMethods {
		b.Run(sm.name, func(b *testing.B) {
			benchSearch(b, ds, sm.fn)
		})
	}
}

func benchSearch(b *testing.B, ds *dataset, fn searchFunc) {
	b.SetBytes(int64(len(ds.data)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		line, err := fn(ds.data, ds.needle, ds.needleColumn)
		if err != nil {
			b.Fatal(err)
		}
		if line != ds.needleLine {
			b.Fatalf("needle not located correctly: line=%q, want line=%q", line, ds.needleLine)
		}
	}
}

type searchFunc func(data []byte, needle []byte, columnIdx int) (line string, err error)

var searchMethods = []struct {
	name string
	fn   searchFunc
}{
	{"searchCSVReader", searchCSVReader},
	{"searchCSVReaderPerLine", searchCSVReaderPerLine},
	{"searchBytes", searchBytes},
}

// Create one csv reader per file.
func searchCSVReader(data []byte, needle []byte, columnIdx int) (string, error) {
	csvR := csv.NewReader(bytes.NewReader(data))
	csvR.Comma = '\t'
	csvR.LazyQuotes = true
	csvR.ReuseRecord = true

	needleStr := string(needle)
	var result string
	line := 0
	for {
		rec, err := csvR.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return "", err
		}
		if columnIdx < len(rec) && strings.Contains(rec[columnIdx], needleStr) {
			result = strings.Join(rec, "\t")
		}
		line++
	}
	return result, nil
}

// Create one csv reader per line.
func searchCSVReaderPerLine(data []byte, needle []byte, columnIdx int) (string, error) {
	needleStr := string(needle)
	lineStart := 0
	var result string
	for {
		lineEnd := bytes.IndexByte(data[lineStart:], '\n')
		if lineEnd == -1 {
			lineEnd = len(data)
		} else {
			lineEnd += lineStart
		}
		if lineStart == lineEnd {
			break
		}

		line := data[lineStart:lineEnd]
		csvR := csv.NewReader(bytes.NewReader(line))
		csvR.Comma = '\t'
		csvR.LazyQuotes = true
		rec, err := csvR.Read()
		if err != nil {
			// This should never return io.EOF
			return "", err
		}
		if columnIdx < len(rec) && strings.Contains(rec[columnIdx], needleStr) {
			result = string(line)
		}

		if lineEnd == len(data) {
			break
		}
		lineStart = lineEnd + 1
	}
	return result, nil
}

// Search the needle using bytes.Index and then reconstruct the line around it.
func searchBytes(data []byte, needle []byte, columnIdx int) (string, error) {
	var separator = []byte{'\t'}

	pos := 0
	var result string
	for pos+len(needle) <= len(data) {
		idx := bytes.Index(data[pos:], needle)
		if idx == -1 {
			break
		}
		pos += idx

		lineStart := bytes.LastIndexByte(data[:pos], '\n')
		if lineStart == -1 {
			lineStart = 0
		} else {
			lineStart++
		}
		if bytes.Count(data[lineStart:pos], separator) == columnIdx {
			lineEnd := bytes.IndexByte(data[pos:], '\n')
			if lineEnd == -1 {
				lineEnd = len(data)
			} else {
				lineEnd += pos
			}
			result = string(data[lineStart:lineEnd])
		}
		pos++
	}
	return result, nil
}

// dataset is one generated TSV dataset plus the needle hidden in it.
type dataset struct {
	data  []byte
	lines int

	needle       []byte
	needleColumn int    // Column index where the needle is located
	needleLine   string // Line where the needle is located
}

func genDataset(size int) *dataset {
	rng := rand.New(rand.NewPCG(uint64(size), uint64(size)))

	result := &dataset{
		// Make sure we never reallocate data
		data: make([]byte, 0, size+size/8),
	}
	// Do reservoir sampling
	candidates := 0

	for len(result.data) < size {
		lineStart := len(result.data)
		thisLineHasNeedle := false
		for columnIdx := range numColumns {
			if columnIdx > 0 {
				result.data = append(result.data, '\t')
			}
			fieldStart := len(result.data)
			result.data = appendField(result.data, rng)
			fieldEnd := len(result.data)

			if fieldEnd-fieldStart > minNeedleLen {
				candidates++
				if rng.IntN(candidates) == 0 {
					result.needle = result.data[fieldStart:fieldEnd]
					result.needleColumn = columnIdx
					thisLineHasNeedle = true
				}
			}
		}
		if thisLineHasNeedle {
			result.needleLine = string(result.data[lineStart:])
		}
		result.data = append(result.data, '\n')
		result.lines++
	}

	return result
}

func appendField(dst []byte, rng *rand.Rand) []byte {
	n := 1 + rng.IntN(maxFieldLen)
	for range n {
		dst = append(dst, '0'+byte(rng.IntN('z'-'0')))
	}
	return dst
}
