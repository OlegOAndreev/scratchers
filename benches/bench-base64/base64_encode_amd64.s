#include "textflag.h"

DATA encodeByteShuffle<>+0(SB)/8, $0xFF030405FF000102
DATA encodeByteShuffle<>+8(SB)/8, $0xFF090A0BFF060708
GLOBL encodeByteShuffle<>(SB), RODATA, $16
DATA encodeByteMask<>+0(SB)/4, $0x00003F00
GLOBL encodeByteMask<>(SB), RODATA, $4
DATA setTopBit<>(SB)/1, $0x70
GLOBL setTopBit<>(SB), RODATA, $1
DATA nextMap<>(SB)/1, $0x10
GLOBL nextMap<>(SB), RODATA, $1
// This is a copy of encodeMap from base64.go. It is used both for scalar loop and simd loop.
DATA encodeMap<>+0(SB)/8, $0x4847464544434241
DATA encodeMap<>+8(SB)/8, $0x504f4e4d4c4b4a49
DATA encodeMap<>+16(SB)/8, $0x5857565554535251
DATA encodeMap<>+24(SB)/8, $0x6665646362615a59
DATA encodeMap<>+32(SB)/8, $0x6e6d6c6b6a696867
DATA encodeMap<>+40(SB)/8, $0x767574737271706f
DATA encodeMap<>+48(SB)/8, $0x333231307a797877
DATA encodeMap<>+56(SB)/8, $0x2f2b393837363534
GLOBL encodeMap<>(SB), RODATA, $64

// '='
#define padding 0x3d

TEXT ·base64EncodeAvx2Asm(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), DI
    MOVQ src+8(FP), SI
    MOVQ numTriples+16(FP), CX
    MOVQ remainder+24(FP), DX
    LEAQ encodeMap<>(SB), R8

    CMPQ CX, $6
    JGE simd_start

scalar_main_loop:
    CMPQ CX, $3
    JB scalar_tail

    // R11 = dst quads, AX = src triples
    MOVQ (SI), AX
    BSWAPQ AX

    MOVQ AX, R10
    SHRQ $58, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, (DI)

    MOVQ AX, R10
    SHRQ $52, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 1(DI)

    MOVQ AX, R10
    SHRQ $46, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 2(DI)

    MOVQ AX, R10
    SHRQ $40, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 3(DI)

    MOVQ AX, R10
    SHRQ $34, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 4(DI)

    MOVQ AX, R10
    SHRQ $28, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 5(DI)

    MOVQ AX, R10
    SHRQ $22, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 6(DI)

    SHRQ $16, AX
    ANDQ $63, AX
    MOVBQZX (R8)(AX*1), R11
    MOVB R11, 7(DI)

    ADDQ $6, SI
    ADDQ $8, DI
    SUBQ $2, CX
    JMP scalar_main_loop

scalar_tail:
    // CX = 0..2 remaining triples, DX = 0..2 remainder
    TESTQ CX, CX
    JZ scalar_remainder

scalar_tail_body:
    // R11 = dst quad, AX = src triple
    MOVBQZX (SI), AX
    SHLQ $16, AX
    MOVBQZX 1(SI), R10
    SHLQ $8, R10
    ORQ R10, AX
    MOVBQZX 2(SI), R10
    ORQ R10, AX

    MOVQ AX, R10
    SHRQ $18, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, (DI)
    MOVQ AX, R10
    SHRQ $12, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 1(DI)
    MOVQ AX, R10
    SHRQ $6, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 2(DI)
    ANDQ $63, AX
    MOVBQZX (R8)(AX*1), R11
    MOVB R11, 3(DI)

    ADDQ $3, SI
    ADDQ $4, DI
    DECQ CX
    JNZ scalar_tail_body

scalar_remainder:
    CMPQ DX, $1
    JB finish
    JE scalar_remainder_1

    // DX = 2
    MOVBQZX (SI), AX
    SHLQ $8, AX
    MOVBQZX 1(SI), R10
    ORQ R10, AX

    MOVQ AX, R10
    SHRQ $10, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, (DI)
    MOVQ AX, R10
    SHRQ $4, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 1(DI)
    MOVQ AX, R10
    SHLQ $2, R10
    ANDQ $63, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, 2(DI)
    MOVB $padding, 3(DI)

    ADDQ $4, DI
    JMP finish

scalar_remainder_1:
    // DX = 1
    MOVBQZX (SI), AX

    MOVQ AX, R10
    SHRQ $2, R10
    MOVBQZX (R8)(R10*1), R11
    MOVB R11, (DI)
    SHLQ $4, AX
    ANDQ $63, AX
    MOVBQZX (R8)(AX*1), R11
    MOVB R11, 1(DI)
    MOVB $padding, 2(DI)
    MOVB $padding, 3(DI)

    ADDQ $4, DI

finish:
    RET

simd_start:
    // Y15 = shuffle triples -> quads: 123456789ABC -> 321065409870CBA0
    VBROADCASTI128 encodeByteShuffle<>(SB), Y15
    // Y14 = 6-bit mask for 2nd byte in quad
    VPBROADCASTD encodeByteMask<>(SB), Y14
    // Y13 = 6-bit mask for 3rd byte in quad
    VPSLLD $8, Y14, Y13
    // Y12 = 6-bit mask for 4th byte in quad
    VPSLLD $16, Y14, Y12
    // Y11 = delta to set the top bit if any of the bits in the upper nibble are set
    VPBROADCASTB setTopBit<>(SB), Y11
    // Y10 = delta for iterating over encodeMap subslices
    VPBROADCASTB nextMap<>(SB), Y10
    // Y6-Y9 = encodeMap 16-byte slices (Y6 = encodeMap[0:15], Y7 = encodeMap[16:31], Y8 = encodeMap[32:47],
    // Y9 = encodeMap[48:63])
    VBROADCASTI128 (R8), Y6
    VBROADCASTI128 16(R8), Y7
    VBROADCASTI128 32(R8), Y8
    VBROADCASTI128 48(R8), Y9

simd_main_loop:
    // Main loop does 28-byte loads and 32-byte stores
    CMPQ CX, $10
    JB simd_tail

    // Load 8 triples by doing to overlapping loads, shuffle into 8 quads (and swap byte order for shifts). PDEP may
    // be more suitable for this, but it has horrible latency in Zen2, so let's stick to the AVX2. Another way
    // is loading one YMM register and then doing VPERMD, but it works slower both on Intel and AMD I tested it on.
    VMOVDQU (SI), X0
    VINSERTI128 $1, 12(SI), Y0, Y0

simd_main_loop_body:
    VPSHUFB Y15, Y0, Y0
    VPSRLD $4, Y0, Y1
    VPAND Y14, Y1, Y1
    VPSLLD $10, Y0, Y2
    VPAND Y13, Y2, Y2
    VPSLLD $24, Y0, Y3
    VPAND Y12, Y3, Y3
    VPSRLD $18, Y0, Y4
    VPOR Y2, Y1, Y1
    VPOR Y4, Y3, Y3
    // Y0 = 32 6-bit values
    VPOR Y3, Y1, Y0

    // Each 6-bit value is mapped to a byte using 4 vpshufb ops. For each byte 00[b1 b2][b3 b4 b5 b6] we need to
    // set top bit to 1 if [b1 b2] is non-zero: this makes pshufb output zero byte. We do this by adding 01110000b
    // (0x70) with saturation.
    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    // Y2 = [00....] mapped to encodeMap[0:15]
    VPSHUFB Y1, Y6, Y2
    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    // Y3 = [01....] mapped to encodeMap[16:31]
    VPSHUFB Y1, Y7, Y3
    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    // Y4 = [10....] mapped to encodeMap[32:47]
    VPSHUFB Y1, Y8, Y4
    VPADDUSB Y11, Y0, Y1
    // Y0 = [11....] mapped to encodeMap[48:63]
    VPSHUFB Y1, Y9, Y0
    VPOR Y3, Y2, Y2
    VPOR Y0, Y4, Y4
    VPOR Y4, Y2, Y2

    VMOVDQU Y2, (DI)
    ADDQ $24, SI
    ADDQ $32, DI

    SUBQ $8, CX
    // This can happend if simd_tail jumped back to simd_main_loop_body and CX was 8.
    JZ simd_finish
    JMP simd_main_loop

simd_tail:
    // CX = 1..10 remaining triples. The main problem is not doing out-of-bounds loads. For CX:
    //  1..4: single XMM load aligned with the end of tail (the buffer contains at least 6 triples)
    //  5: fix up the SI and DI and use 6..7 case
    //  6..7: two XMM loads: one aligned with the start of tail and one aligned with the end
    //  8..9: two XMM loads: one aligned with the start of tail and one aligned with the end, then
    //        main loop body is reused, then simd_tail is re-entered again for remaining 0..2 triples
    // All cases are processed by the same code, the only difference in the prolog: input must be in Y0,
    // the destinations for two processed halves in R11 and R12, SI and DI updated accordingly.

    CMPQ CX, $5
    JB simd_tail_1_4
    JE simd_tail_5
    CMPQ CX, $8
    JB simd_tail_6_7

    // CX = 8..9. Do two loads into Y0 and pass back to main loop.
    VMOVDQU (SI), X0
    VMOVDQU 8(SI), X1
    VPSRLDQ $4, X1, X1
    VINSERTI128 $1, X1, Y0, Y0
    JMP simd_main_loop_body

simd_tail_5:
    // CX = 5
    INCQ CX
    SUBQ $3, SI
    SUBQ $4, DI
    // Pass-through to 6..7 case

simd_tail_6_7:
    // CX = 6..7
    VMOVDQU (SI), X0
    LEAQ (CX)(CX*2), R9
    ADDQ R9, SI
    VMOVDQU -16(SI), X1
    VPSRLDQ $4, X1, X1
    VINSERTI128 $1, X1, Y0, Y0

    MOVQ DI, R11
    LEAQ (DI)(CX*4), DI
    LEAQ -16(DI), R12

    JMP simd_tail_main

simd_tail_1_4:
    // CX = 1..4, mirror both halves
    LEAQ (CX)(CX*2), R9
    ADDQ R9, SI
    VBROADCASTI128 -16(SI), Y0
    VPSRLDQ $4, Y0, Y0

    LEAQ (DI)(CX*4), DI
    LEAQ -16(DI), R11
    MOVQ R11, R12

simd_tail_main:
    XORQ CX, CX
    // This is a carbon copy of the main loop.
    VPSHUFB Y15, Y0, Y0
    VPSRLD $4, Y0, Y1
    VPAND Y14, Y1, Y1
    VPSLLD $10, Y0, Y2
    VPAND Y13, Y2, Y2
    VPSLLD $24, Y0, Y3
    VPAND Y12, Y3, Y3
    VPSRLD $18, Y0, Y4
    VPOR Y2, Y1, Y1
    VPOR Y4, Y3, Y3
    VPOR Y3, Y1, Y0

    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    VPSHUFB Y1, Y6, Y2
    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    VPSHUFB Y1, Y7, Y3
    VPADDUSB Y11, Y0, Y1
    VPSUBB Y10, Y0, Y0
    VPSHUFB Y1, Y8, Y4
    VPADDUSB Y11, Y0, Y1
    VPSHUFB Y1, Y9, Y5

    VPOR Y3, Y2, Y2
    VPOR Y5, Y4, Y4
    VPOR Y4, Y2, Y2

    VMOVDQU X2, (R11)
    VEXTRACTI128 $1, Y2, (R12)

simd_finish:
    VZEROUPPER
    JMP scalar_remainder
