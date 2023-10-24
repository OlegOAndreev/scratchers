package main

import (
	"math/rand"
	"testing"

	"github.com/olegoandreev/scratchers/benches/bench-hsum/hsumcgo"
)

func doTest(t *testing.T, f func(buf []byte, from int, to int) uint64, buf []byte, from int, to int) {
	golden := HorizontalSumNaive(buf, from, to)
	got := f(buf, from, to)
	if golden != got {
		t.Fatalf("different values for [%d:%d): should be %d, got %d", from, to, golden, got)
	}
}

func testHorizontalAdd(t *testing.T, f func(buf []byte, from int, to int) uint64) {
	doTestBuf := func(t *testing.T, buf []byte) {
		doTest(t, f, buf, 0, len(buf))
		if len(buf) > 1 {
			doTest(t, f, buf, 1, len(buf))
			doTest(t, f, buf, 0, len(buf)-1)
			doTest(t, f, buf, 1, len(buf)-1)
		}
	}

	t.Run("All zeros", func(t *testing.T) {
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0)
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, 0)
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0)
		}
	})

	t.Run("All ones", func(t *testing.T) {
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 1)
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, 1)
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 1)
		}
	})

	t.Run("All 0xFF", func(t *testing.T) {
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0xFF)
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, 0xFF)
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0xFF)
		}
	})

	t.Run("Small bytes", func(t *testing.T) {
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, byte(i%10))
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, byte(i%10))
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, byte(i%10))
		}
	})

	t.Run("Large numbers", func(t *testing.T) {
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0xF6+byte(i%10))
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, 0xF6+byte(i%10))
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, 0xF6+byte(i%10))
		}
	})

	t.Run("Random numbers", func(t *testing.T) {
		r := rand.New(rand.NewSource(0))
		var buf []byte
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, byte(r.Intn(0x100)))
		}
		for i := 0; i < 2000; i++ {
			buf = append(buf, byte(r.Intn(0x100)))
		}
		for i := 0; i < 200; i++ {
			doTestBuf(t, buf)
			buf = append(buf, byte(r.Intn(0x100)))
		}
	})

	t.Run("Large buffer", func(t *testing.T) {
		// Test the i32 overflow
		buf := make([]byte, 20000000)
		for i := 0; i < len(buf); i++ {
			buf[i] = 0xFE
		}
		doTestBuf(t, buf)
	})
}

func TestHorizontalAdd(t *testing.T) {
	t.Run("HorizontalSumUnroll", func(t *testing.T) {
		testHorizontalAdd(t, HorizontalSumUnroll)
	})
	t.Run("HorizontalSumNibbles", func(t *testing.T) {
		testHorizontalAdd(t, HorizontalSumNibbles)
	})
	t.Run("HorizontalSumNibblesUnsafe", func(t *testing.T) {
		testHorizontalAdd(t, HorizontalSumNibblesUnsafe)
	})
	if avx2Enabled {
		t.Run("HorizontalSumAvx2", func(t *testing.T) {
			testHorizontalAdd(t, HorizontalSumAvx2)
		})
		t.Run("HorizontalSumAvx2V2", func(t *testing.T) {
			testHorizontalAdd(t, HorizontalSumAvx2V2)
		})
	}
	if neonEnabled {
		t.Run("HorizontalSumNeon", func(t *testing.T) {
			testHorizontalAdd(t, HorizontalSumNeon)
		})
		t.Run("HorizontalSumNeonV2", func(t *testing.T) {
			testHorizontalAdd(t, HorizontalSumNeonV2)
		})
	}
	if hsumcgo.Enabled {
		t.Run("HorizontalSumNaiveC", func(t *testing.T) {
			testHorizontalAdd(t, hsumcgo.HorizontalSumNaiveC)
		})
		if hsumcgo.Avx2Enabled() {
			t.Run("HorizontalSumAvx2C", func(t *testing.T) {
				testHorizontalAdd(t, hsumcgo.HorizontalSumAvx2C)
			})
		}
	}
}

func testHorizontalAddRandom(t *testing.T, f func(buf []byte, from int, to int) uint64, buf []byte) {
	const COUNT = 1024
	r := rand.New(rand.NewSource(1))
	for i := 0; i < COUNT; i++ {
		from := r.Intn(len(buf))
		to := from + r.Intn(len(buf)-from)
		doTest(t, f, buf, from, to)
	}
}

func TestHorizontalAddRandom(t *testing.T) {
	const N = 1024 * 1024
	buf := make([]byte, N)
	r := rand.New(rand.NewSource(0))
	for i := 0; i < N; i++ {
		buf[i] = byte(r.Intn(0x100))
	}

	t.Run("HorizontalSumUnroll", func(t *testing.T) {
		testHorizontalAddRandom(t, HorizontalSumUnroll, buf)
	})
	t.Run("HorizontalSumNibbles", func(t *testing.T) {
		testHorizontalAddRandom(t, HorizontalSumNibbles, buf)
	})
	t.Run("HorizontalSumNibblesUnsafe", func(t *testing.T) {
		testHorizontalAddRandom(t, HorizontalSumNibblesUnsafe, buf)
	})
	if avx2Enabled {
		t.Run("HorizontalSumAvx2", func(t *testing.T) {
			testHorizontalAddRandom(t, HorizontalSumAvx2, buf)
		})
		t.Run("HorizontalSumAvx2V2", func(t *testing.T) {
			testHorizontalAddRandom(t, HorizontalSumAvx2V2, buf)
		})
	}
	if neonEnabled {
		t.Run("HorizontalSumNeon", func(t *testing.T) {
			testHorizontalAddRandom(t, HorizontalSumNeon, buf)
		})
		t.Run("HorizontalSumNeonV2", func(t *testing.T) {
			testHorizontalAddRandom(t, HorizontalSumNeonV2, buf)
		})
	}
	if hsumcgo.Enabled {
		t.Run("HorizontalSumNaiveC", func(t *testing.T) {
			testHorizontalAddRandom(t, hsumcgo.HorizontalSumNaiveC, buf)
		})
		if hsumcgo.Avx2Enabled() {
			t.Run("HorizontalSumAvx2C", func(t *testing.T) {
				testHorizontalAddRandom(t, hsumcgo.HorizontalSumAvx2C, buf)
			})
		}
	}
}
