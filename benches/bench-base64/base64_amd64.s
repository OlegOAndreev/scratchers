#include "textflag.h"

DATA encodeLoadShuffle<>+0(SB)/8, $0xFF030405FF000102
DATA encodeLoadShuffle<>+8(SB)/8, $0xFF090A0BFF060708
GLOBL encodeLoadShuffle<>(SB), RODATA, $16
DATA encodeByteMask<>+0(SB)/4, $0x00003F00
GLOBL encodeByteMask<>(SB), RODATA, $4
DATA setTopBit<>(SB)/1, $0x70
GLOBL setTopBit<>(SB), RODATA, $1
DATA nextMap<>(SB)/1, $0x10
GLOBL nextMap<>(SB), RODATA, $1
DATA const0<>(SB)/8, $0
GLOBL const0<>(SB), RODATA, $8
DATA const1<>(SB)/8, $1
GLOBL const1<>(SB), RODATA, $8
DATA const24<>(SB)/8, $24
GLOBL const24<>(SB), RODATA, $8
DATA const32<>(SB)/8, $32
GLOBL const32<>(SB), RODATA, $8

// '='
#define padding 61

TEXT ·base64EncodeAvx2Asm(SB), $0-40
    MOVQ dst+0(FP), DI
    MOVQ src+8(FP), SI
    MOVQ encodeMap+16(FP), R8
    MOVQ numTriples+24(FP), CX
    MOVQ remainder+32(FP), DX

    CMPQ CX, $6
    JGE simd_start

plain_main_loop:
    CMPQ CX, $2
    JB plain_tail

    // R11 = dst quad, AX = src triple
    MOVL (SI), AX
    BSWAPL AX

    MOVQ AX, R10
    SHRQ $8, R10
    ANDQ $63, R10
    MOVBLZX (R8)(R10*1), R11
    SHLQ $16, R11

    MOVQ AX, R10
    SHRQ $14, R10
    ANDQ $63, R10
    MOVBLZX (R8)(R10*1), R12
    SHLQ $16, R12

    MOVQ AX, R10
    SHRQ $20, R10
    ANDQ $63, R10
    MOVB (R8)(R10*1), R11
    SHLQ $8, R11

    SHRQ $26, AX
    MOVB (R8)(AX*1), R12

    ORQ R12, R11
    MOVL R11, (DI)

    ADDQ $3, SI
    ADDQ $4, DI
    DECQ CX
    JMP plain_main_loop

plain_tail:
    // CX = 0..1 remaining triples, DX = 0..2 remainder
    TESTQ CX, CX
    JZ plain_remainder

    // R11 = dst quad, AX = src triple
    MOVBLZX (SI), AX
    SHLQ $8, AX
    MOVB 1(SI), AX
    SHLQ $8, AX
    MOVB 2(SI), AX

    MOVQ AX, R10
    ANDQ $63, R10
    MOVBLZX (R8)(R10*1), R11
    SHLQ $16, R11

    MOVQ AX, R10
    SHRQ $6, R10
    ANDQ $63, R10
    MOVBLZX (R8)(R10*1), R12
    SHLQ $16, R12

    MOVQ AX, R10
    SHRQ $12, R10
    ANDQ $63, R10
    MOVB (R8)(R10*1), R11
    SHLQ $8, R11

    SHRQ $18, AX
    MOVB (R8)(AX*1), R12

    ORQ R12, R11
    MOVL R11, (DI)

    ADDQ $3, SI
    ADDQ $4, DI

plain_remainder:
    CMPQ DX, $1
    JB finish
    JE plain_remainder_1

    // DX = 2
    MOVL $padding, R11
    SHLQ $8, R11

    MOVBLZX (SI), AX
    SHLQ $8, AX
    MOVB 1(SI), AX

    MOVQ AX, R10
    SHLQ $2, R10
    ANDQ $63, R10
    MOVB (R8)(R10*1), R11
    SHLQ $8, R11

    MOVQ AX, R10
    SHRQ $4, R10
    ANDQ $63, R10
    MOVB (R8)(R10*1), R11
    SHLQ $8, R11

    SHRQ $10, AX
    MOVB (R8)(AX*1), R11

    MOVL R11, (DI)
    ADDQ $4, DI

    JMP finish

plain_remainder_1:
    // DX = 1
    MOVL $padding, R11
    SHLQ $8, R11
    MOVB $padding, R11
    SHLQ $8, R11

    MOVBLZX (SI), AX

    MOVQ AX, R10
    SHLQ $4, R10
    ANDQ $63, R10
    MOVB (R8)(R10*1), R11
    SHLQ $8, R11

    SHRQ $2, AX
    MOVB (R8)(AX*1), R11

    MOVL R11, (DI)
    ADDQ $4, DI

finish:
    RET

simd_start:
    // Y6 = shuffle triples -> quads: 123456789ABC -> 321065409870CBA0
    VBROADCASTI128 encodeLoadShuffle<>(SB), Y6
    // Y7 = 6-bit mask for 2nd byte in quad
    VPBROADCASTD encodeByteMask<>(SB), Y7
    // Y8 = 6-bit mask for 3rd byte in quad
    VPSLLD $8, Y7, Y8
    // Y9 = 6-bit mask for 4th byte in quad
    VPSLLD $16, Y7, Y9
    // Y10 = delta for setting top bit in each byte
    VPBROADCASTB setTopBit<>(SB), Y10
    // Y11 = delta for iterating over encodeMap subslices
    VPBROADCASTB nextMap<>(SB), Y11
    // Y12-Y15 = encodeMap slices
    VBROADCASTI128 (R8), Y12
    VBROADCASTI128 16(R8), Y13
    VBROADCASTI128 32(R8), Y14
    VBROADCASTI128 48(R8), Y15

simd_main_loop:
    // Main loop does two 16-byte loads, 28 bytes in total,
    CMPQ CX, $10
    JB simd_tail

    // Load 8 triples by doing to overlapping loads, shuffle into 8 quads (and swap byte order for shifts). PDEP may
    // be more suitable for this, but it has horrible latency in Zen2, so let's stick to the AVX2.
    MOVOU (SI), X0
    VINSERTI128 $1, 12(SI), Y0, Y0
    VPSHUFB Y6, Y0, Y0
    VPSRLD $4, Y0, Y1
    VPAND Y7, Y1, Y1
    VPSLLD $10, Y0, Y2
    VPAND Y8, Y2, Y2
    VPSLLD $24, Y0, Y3
    VPAND Y9, Y3, Y3
    VPSRLD $18, Y0, Y4
    VPOR Y2, Y1, Y1
    VPOR Y4, Y3, Y3
    // Y0 = 32 6-bit values
    VPOR Y3, Y1, Y0

    // Each 6-bit value is mapped to a byte using 4 vpshufb ops. For each byte 00[b1 b2][b3 b4 b5 b6] we need to
    // set top bit to 1 if [b1 b2] is non-zero: this makes pshufb output zero byte. We do this by adding 01110000b
    // (0x70) with saturation.
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    // Y2 = [00....] mapped to encodeMap[0:15]
    VPSHUFB Y1, Y12, Y2
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    // Y3 = [01....] mapped to encodeMap[16:31]
    VPSHUFB Y1, Y13, Y3
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    // Y4 = [10....] mapped to encodeMap[32:47]
    VPSHUFB Y1, Y14, Y4
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    // Y5 = [11....] mapped to encodeMap[48:63]
    VPSHUFB Y1, Y15, Y5

    VPOR Y3, Y2, Y2
    VPOR Y5, Y4, Y4
    VPOR Y4, Y2, Y2
    VMOVDQU Y2, (DI)

    ADDQ $24, SI
    ADDQ $32, DI

    SUBQ $8, CX
    JMP simd_main_loop

simd_tail:
    // CX = 1..10 remaining triples. The main problem is not doing out-of-bounds loads. Depending on the CX:
    //  1..4: single XMM load aligned with the end (the buffer contains at least 6 triples)
    //  5: two XMM loads: one aligned with the end and one with one triple delta
    //  6..8: two XMM loads: one aligned with the start of remainder and one aligned with the end
    //  9: first 8 triples are processed and then the last triple is processed by looping to the remainder
    // All cases are processed by the same code, the only difference in the prolog: input must be in Y0,
    // the destinations for two processed halves in R11 and R12, SI and DI updated accordingly.

    // R9 = CX*3 (src remaining bytes), R10 = CX*4 (dst remaining bytes)
    LEAQ (CX)(CX*2), R9
    LEAQ (CX*4), R10

    CMPQ CX, $5
    JB simd_tail_1_4
    JE simd_tail_5

    // CX = 6..9
    // Special processing for 9 remaining triples.
    CMPQ CX, $9
    CMOVQNE const0<>(SB), CX
    CMOVQEQ const1<>(SB), CX
    CMOVQEQ const24<>(SB), R9
    CMOVQEQ const32<>(SB), R10

    MOVOU (SI), X0
    ADDQ R9, SI
    MOVOU -16(SI), X1
    VPSRLDQ $4, X1, X1
    VINSERTI128 $1, X1, Y0, Y0

    MOVQ DI, R11
    ADDQ R10, DI
    LEAQ -16(DI), R12

    JMP simd_tail_main

simd_tail_5:
    // CX = 5
    XORQ CX, CX

    MOVOU -4(SI), X0
    VINSERTI128 $1, -1(SI), Y0, Y0
    VPSRLDQ $4, Y0, Y0
    ADDQ R9, SI

    MOVQ DI, R11
    LEAQ 4(DI), R12
    ADDQ R10, DI

    JMP simd_tail_main

simd_tail_1_4:
    // CX = 1..4
    XORQ CX, CX

    ADDQ R9, SI
    VBROADCASTI128 -16(SI),Y0
    VPSRLDQ $4, Y0, Y0

    ADDQ R10, DI
    LEAQ -16(DI), R11
    LEAQ -16(DI), R12

simd_tail_main:
    // This is a carbon copy of the main loop.
    VPSHUFB Y6, Y0, Y0
    VPSRLD $4, Y0, Y1
    VPAND Y7, Y1, Y1
    VPSLLD $10, Y0, Y2
    VPAND Y8, Y2, Y2
    VPSLLD $24, Y0, Y3
    VPAND Y9, Y3, Y3
    VPSRLD $18, Y0, Y4
    VPOR Y2, Y1, Y1
    VPOR Y4, Y3, Y3
    VPOR Y3, Y1, Y0

    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    VPSHUFB Y1, Y12, Y2
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    VPSHUFB Y1, Y13, Y3
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    VPSHUFB Y1, Y14, Y4
    VPADDUSB Y10, Y0, Y1
    VPSUBB Y11, Y0, Y0
    VPSHUFB Y1, Y15, Y5

    VPOR Y3, Y2, Y2
    VPOR Y5, Y4, Y4
    VPOR Y4, Y2, Y2

    MOVOU X2, (R11)
    VEXTRACTI128 $1, Y2, (R12)

    // Special processing for 9 remaining triples
    TESTQ CX, CX
    JNZ simd_tail

    VZEROUPPER
    JMP plain_remainder
