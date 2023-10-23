// Package hsumcgo is separate because cgo does not mix with golang asm.
package hsumcgo

import "unsafe"

/*
#cgo CFLAGS: -O3
#cgo amd64 CFLAGS: -mavx2

#include <stdint.h>

uint64_t HorizontalSumNaiveCImpl(uint8_t* buf, size_t size)
{
	uint64_t ret = 0;
	for (size_t i = 0; i < size; i++) {
		ret += (uint64_t)buf[i];
	}
	return ret;
}

#ifdef __AVX2__

#include <immintrin.h>

// This mostly is a copy-paste of hsum_amd64.s, check the comments there.
uint64_t HorizontalSumAvx2CImpl(uint8_t* buf, size_t size)
{
	uint64_t ret = 0;
	size_t vsize = size / 16;
	while (vsize > 0) {
		__m256i accum = _mm256_setzero_si256();
		if (vsize >= 64) {
			for (size_t j = 0; j < 16; j++) {
				__m256i v0 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps((const float*)(buf))));
				__m256i v1 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps((const float*)(buf + 16))));
				__m256i v2 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps((const float*)(buf + 32))));
				__m256i v3 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps((const float*)(buf + 48))));
				buf += 64;
				v0 = _mm256_add_epi16(v0, v1);
				v2 = _mm256_add_epi16(v2, v3);
				accum = _mm256_add_epi16(accum, v0);
				accum = _mm256_add_epi16(accum, v2);
			}
			vsize -= 64;
		} else {
			for (size_t j = 0; j < vsize; j++) {
				__m256i v0 = _mm256_cvtepu8_epi16(_mm_castps_si128(_mm_loadu_ps((const float*)(buf))));
				buf += 16;
				accum = _mm256_add_epi16(accum, v0);
			}
			vsize = 0;
		}

		__m128i upper = _mm256_extracti128_si256(accum, 1);
		__m128i lower = _mm256_castsi256_si128(accum);
		__m128i s1 = _mm_add_epi16(lower, upper);
		__m128i upper_s1 = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(s1)));
		__m128i s2 = _mm_add_epi16(upper_s1, s1);
		uint64_t s3 = _mm_cvtsi128_si64(s2);
		uint64_t s4 = (s3 & 0x0000FFFF0000FFFF) + ((s3 >> 16) & 0x0000FFFF0000FFFF);
		uint64_t s5 = (s4 >> 32) + (s4 & 0xFFFFFFFF);
		ret += s5;
	}
	size %= 16;
	for (size_t j = 0; j < size; j++) {
		ret += (uint64_t)buf[j];
	}
	return ret;
}

int Avx2Enabled()
{
	return 1;
}

#else

uint64_t HorizontalSumAvx2CImpl(uint8_t* buf, size_t size)
{
	return 0;
}

int Avx2Enabled()
{
	return 0;
}

#endif
*/
import "C"

const Enabled = true

func HorizontalSumNaiveC(buf []byte, from int, to int) uint64 {
	ptr := (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(buf[from:])))
	size := C.size_t(to - from)
	return uint64(C.HorizontalSumNaiveCImpl(ptr, size))
}

func HorizontalSumAvx2C(buf []byte, from int, to int) uint64 {
	ptr := (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(buf[from:])))
	size := C.size_t(to - from)
	return uint64(C.HorizontalSumAvx2CImpl(ptr, size))
}

func Avx2Enabled() bool {
	return int(C.Avx2Enabled()) != 0
}
