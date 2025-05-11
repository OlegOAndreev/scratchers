package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/olegoandreev/scratchers/benches/bench-hsum/hsumcgo"
)

type XorShift128 struct {
	a, b, c, d uint32
}

func NewXorShift128() *XorShift128 {
	return &XorShift128{a: 11, b: 22, c: 33, d: 44}
}

func (x *XorShift128) Next() uint32 {
	t := x.d
	s := x.a
	x.d = x.c
	x.c = x.b
	x.b = s
	t ^= t << 11
	t ^= t >> 8
	x.a = t ^ s ^ (s >> 19)
	return x.a
}

func (x *XorShift128) NextN(max uint32) uint32 {
	// https://lemire.me/blog/2019/06/06/nearly-divisionless-random-integer-generation-on-various-systems/
	return uint32(uint64(x.Next()) * uint64(max) >> 32)
}

var filterFunc string

func main() {
	if len(os.Args) > 1 {
		filterFunc = os.Args[1]
	}

	const N = 1024 * 1024 * 1024
	buf := make([]byte, N)
	fillRandom(buf, 13)

	fmt.Println("One large buffer")
	bench(buf, []pair{{0, N}}, N)

	fmt.Println("Large chunks in large buffer")
	indices, total := prepareIndices(N, 1024*1024, 10000)
	bench(buf, indices, total)

	fmt.Println("Small chunks in large buffer")
	indices, total = prepareIndices(N, 1024, 10000000)
	bench(buf, indices, total)

	fmt.Println("Small chunks in small buffer")
	indices, total = prepareIndices(1024*1024, 1024, 10000000)
	bench(buf, indices, total)

	fmt.Println("Very small chunks in very small buffer")
	indices, total = prepareIndices(16*1024, 128, 100000000)
	bench(buf, indices, total)
}

func fillRandom(buf []byte, limit byte) {
	rng := NewXorShift128()
	for i := 0; i < len(buf); i++ {
		buf[i] = byte(rng.NextN(uint32(limit)))
	}
}

func bench(buf []byte, indices []pair, total uint64) {
	type benchFunc struct {
		name string
		impl func([]byte, int, int) uint64
	}

	funcs := []benchFunc{
		{"HorizontalSumNaive (warmup)", HorizontalSumNaive},
		{"HorizontalSumUnroll", HorizontalSumUnroll},
		{"HorizontalSumInt8ToInt16", HorizontalSumInt8ToInt16},
		{"HorizontalSumInt8ToInt16Unsafe", HorizontalSumInt8ToInt16Unsafe},
		{"HorizontalSumInt4ToInt8", HorizontalSumInt4ToInt8},
		{"HorizontalSumInt4ToInt8Unsafe", HorizontalSumInt4ToInt8Unsafe},
	}
	if avx2Enabled {
		funcs = append(funcs, benchFunc{"HorizontalSumAvx2", HorizontalSumAvx2})
		funcs = append(funcs, benchFunc{"HorizontalSumAvx2V2", HorizontalSumAvx2V2})
	}
	if neonEnabled {
		funcs = append(funcs, benchFunc{"HorizontalSumNeon", HorizontalSumNeon})
		funcs = append(funcs, benchFunc{"HorizontalSumNeonV2", HorizontalSumNeonV2})
	}
	if hsumcgo.Enabled {
		funcs = append(funcs, benchFunc{"HorizontalSumNaiveC", hsumcgo.HorizontalSumNaiveC})
		if hsumcgo.Avx2Enabled() {
			funcs = append(funcs, benchFunc{"HorizontalSumAvx2C", hsumcgo.HorizontalSumAvx2C})
		}
	}

	for _, f := range funcs {
		if filterFunc != "" && !strings.Contains(f.name, filterFunc) && !strings.Contains(f.name, "(warmup)") {
			continue
		}
		startTime := time.Now()
		var sum uint64
		for _, p := range indices {
			sum += f.impl(buf, p.idx1, p.idx2)
		}
		gbSec := float64(total) / (time.Now().Sub(startTime).Seconds() * 1024.0 * 1024.0 * 1024.0)
		fmt.Printf("%-40s got %d, %.1f Gb/sec\n", f.name+":", sum, gbSec)
	}
}

type pair struct {
	idx1, idx2 int
}

func prepareIndices(bufLen int, maxPair int, count int) ([]pair, uint64) {
	ret := make([]pair, count)
	var total uint64
	rng := NewXorShift128()
	for i := 0; i < count; i++ {
		ret[i].idx1 = int(rng.NextN(uint32(bufLen)))
		l := rng.NextN(uint32(bufLen - ret[i].idx1))
		if l > uint32(maxPair) {
			l = uint32(maxPair)
		}
		total += uint64(l)
		ret[i].idx2 = ret[i].idx1 + int(l)
	}
	return ret, total
}
