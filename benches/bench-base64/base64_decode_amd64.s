#include "textflag.h"

// This is a copy of decodeMap from base64.go. It is used only for scalar loop.
DATA decodeMap<>+0(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+16(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+24(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+32(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+40(SB)/8, $0x3FFFFFFF3EFFFFFF
DATA decodeMap<>+48(SB)/8, $0x3B3A393837363534
DATA decodeMap<>+56(SB)/8, $0xFFFFFFFFFFFF3D3C
DATA decodeMap<>+64(SB)/8, $0x06050403020100FF
DATA decodeMap<>+72(SB)/8, $0x0E0D0C0B0A090807
DATA decodeMap<>+80(SB)/8, $0x161514131211100F
DATA decodeMap<>+88(SB)/8, $0xFFFFFFFFFF191817
DATA decodeMap<>+96(SB)/8, $0x201F1E1D1C1B1AFF
DATA decodeMap<>+104(SB)/8, $0x2827262524232221
DATA decodeMap<>+112(SB)/8, $0x302F2E2D2C2B2A29
DATA decodeMap<>+120(SB)/8, $0xFFFFFFFFFF333231
DATA decodeMap<>+128(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+136(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+144(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+152(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+160(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+168(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+176(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+184(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+192(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+200(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+208(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+216(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+224(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+232(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+240(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA decodeMap<>+248(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL decodeMap<>(SB), RODATA, $256

DATA decodeByteMask<>+0(SB)/4, $0x0000FF00
GLOBL decodeByteMask<>(SB), RODATA, $4
DATA decodeByteShuffle<>+0(SB)/8, $0x090A040506000102
DATA decodeByteShuffle<>+8(SB)/8, $0xFFFFFFFF0C0D0E08
GLOBL decodeByteShuffle<>(SB), RODATA, $16

DATA firstRangeStart<>+0(SB)/1, $0x2B
GLOBL firstRangeStart<>(SB), RODATA, $1
DATA setTopBit<>+0(SB)/1, $0x70
GLOBL setTopBit<>(SB), RODATA, $1
DATA firstRangeMap<>+0(SB)/8, $0x3736354AFFFFFF3F
DATA firstRangeMap<>+8(SB)/8, $0xFF3E3D3C3B3A3939
GLOBL firstRangeMap<>(SB), RODATA, $16
DATA secondRangeAdd<>+0(SB)/1, $0xA5
GLOBL secondRangeAdd<>(SB), RODATA, $1
DATA secondRangeSub<>+0(SB)/1, $0xE5
GLOBL secondRangeSub<>(SB), RODATA, $1
DATA thirdRangeGt<>+0(SB)/1, $0x60
GLOBL thirdRangeGt<>(SB), RODATA, $1
DATA thirdRangeAdd<>+0(SB)/1, $0x85
GLOBL thirdRangeAdd<>(SB), RODATA, $1
DATA thirdRangeSub<>+0(SB)/1, $0xCB
GLOBL thirdRangeSub<>(SB), RODATA, $1
DATA one<>(SB)/1, $0x01
GLOBL one<>(SB), RODATA, $1

// '='
#define padding 0x3d

TEXT ·base64DecodeAvx2Asm(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), DI
    MOVQ src+8(FP), SI
    MOVQ numQuads+16(FP), CX
    LEAQ decodeMap<>(SB), R8

    // If the top bit of R15 is set, the input is malformed. The error checking is deferred until the end
    // of the function.
    XORQ R15, R15

    CMPQ CX, $8
    JGE simd_start

scalar_main_loop:
    CMPQ CX, $3
    JB scalar_tail

    // R11 = dst triples, AX = src quads. Using MOVBQSX we set the top bit in R11 if any of the decodeMap entries is
    // 0xFF.
    MOVBQZX (SI), AX
    MOVBQSX (R8)(AX*1), R11
    SHLQ $42, R11

    MOVBQZX 1(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $36, R12
    ORQ R12, R11

    MOVBQZX 2(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $30, R12
    ORQ R12, R11

    MOVBQZX 3(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $24, R12
    ORQ R12, R11

    MOVBQZX 4(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $18, R12
    ORQ R12, R11

    MOVBQZX 5(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $12, R12
    ORQ R12, R11

    MOVBQZX 6(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $6, R12
    ORQ R12, R11

    MOVBQZX 7(SI), AX
    MOVBQSX (R8)(AX*1), R12
    ORQ R12, R11

    // Save the top bit for error checking and shift the bytes for bswap.
    ORQ R11, R15
    SHLQ $16, R11
    BSWAPQ R11
    MOVQ R11, (DI)
    ADDQ $8, SI
    ADDQ $6, DI
    SUBQ $2, CX
    JMP scalar_main_loop

scalar_tail:
    // CX = 1..2 remaining quads
    CMPQ CX, $2
    JB final

    // R11 = dst triples, AX = src quads.
    MOVBQZX (SI), AX
    MOVBQSX (R8)(AX*1), R11
    SHLQ $18, R11

    MOVBQZX 1(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $12, R12
    ORQ R12, R11

    MOVBQZX 2(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $6, R12
    ORQ R12, R11

    MOVBQZX 3(SI), AX
    MOVBQSX (R8)(AX*1), R12
    ORQ R12, R11

    ORQ R11, R15
    SHLQ $8, R11
    BSWAPL R11
    MOVL R11, (DI)
    ADDQ $4, SI
    ADDQ $3, DI

final:
    // R11 = dst triples, AX = src quads, R14 = num padding
    XORQ R14, R14

    MOVBQZX (SI), AX
    MOVBQSX (R8)(AX*1), R11
    SHLQ $18, R11

    MOVBQZX 1(SI), AX
    MOVBQSX (R8)(AX*1), R12
    SHLQ $12, R12
    ORQ R12, R11

    MOVBQZX 2(SI), AX
    CMPQ AX, $padding
    JE final_padding_2
    MOVBQSX (R8)(AX*1), R12
    SHLQ $6, R12
    ORQ R12, R11

    MOVBQZX 3(SI), AX
    CMPQ AX, $padding
    JE final_padding_1
    MOVBQSX (R8)(AX*1), R12
    ORQ R12, R11
    JMP final_write

final_padding_2:
    MOVQ $2, R14
    // Check that the final symbol is padding as well.
    MOVQ $-1, R13
    MOVBQZX 3(SI), AX
    CMPQ AX, $padding
    CMOVQNE R13, R14
    JMP final_write

final_padding_1:
    MOVQ $1, R14

final_write:
    ORQ R11, R15
    SHLQ $8, R11
    BSWAPL R11
    MOVL R11, (DI)

    // If top bit in R15 is zero, R15 becomes 0, otherwise it becomes -1.
    SARQ $63, R15
    ORQ R15, R14

return:
    MOVQ R14, numPadding+24(FP)
    RET

simd_start:
    // simd loop does not use decodeMap at all. Instead it relies on splitting the input bytes into 3 separate ranges:
    //  0x2B..0x3A (16 byte): '+', ..., '/', '0'-'9', ...
    //  0x41..0x5A: 'A'-'Z'
    //  0x61..0x7A: 'a'-'z'
    // We map each of the ranges to corresponding value+1, or them and sub 1 after that. The out of range values
    // become 0xFF, and we gather the top bits for error checking.

    // Y15 = mask for 2nd byte in quad
    VPBROADCASTD decodeByteMask<>(SB), Y15
    // Y14 = mask for 3rd byte in quad
    VPSLLD $8, Y15, Y14
    // Y13 = shuffle quads -> triples: 321065409870CBA0 -> 123456789ABC000
    VBROADCASTI128 decodeByteShuffle<>(SB), Y13

//    // Y6 = start of the first range
//    VPBROADCASTB firstRangeStart<>(SB), Y6
//    // Y7 = delta to set the top bit if any of the bits in the upper nibble are set
//    VPBROADCASTB setTopBit<>(SB), Y7
//    // Y8 = map from the first range to 6-bits+1 or 0xFF
//    VBROADCASTI128 firstRangeMap<>(SB), Y8
//    // Y9 = 0xFF - 0x5A
//    VPBROADCASTB secondRangeAdd<>(SB), Y9
//    // Y10 = 0x41 + 0xFF - 0x5A - 1
//    VPBROADCASTB secondRangeSub<>(SB), Y10
//    // Y11 = one below the third range
//    VPBROADCASTB thirdRangeGt<>(SB), Y11
//    // Y12 = 0xFF - 0x7A
//    VPBROADCASTB thirdRangeAdd<>(SB), Y12
//    // Y13 = 0x61 + 0xFF - 0x7A - 0x1A - 1
//    VPBROADCASTB thirdRangeSub<>(SB), Y13
//    // Y14 = 0x0101...
//    VPBROADCASTB one<>(SB), Y14

    // If any of the top bits in bytes in Y5 are set, the input is malformed.
    VPXOR Y5, Y5, Y5

simd_main_loop:
    // Main loop does 32-byte loads and 28-byte stores
    VMOVDQU (SI), Y0

    // Move the start of the first range (0x2B..0x3A) to zero.
    VPSUBB Y6, Y0, Y1
    // Set top bit for all values outside of the first range.
    VPADDUSB Y7, Y1, Y1
    // Y1 = the first range mapped to 6-bits+1
    VPSHUFB Y1, Y8, Y1
    // Move the end of the second range (0x41..0x5A) to 0xFF
    VPADDB Y9, Y0, Y2
    // Y2 = the second range mapped to 6-bits+1
    VPSUBUSB Y10, Y2, Y2
    // Y4 = mask for values in the third range
    VPCMPGTB Y11, Y0, Y4
    // Move the end of the third range (0x61..0x7A) to 0xFF
    VPADDB Y12, Y0, Y3
    // Y3 = the third range mapped to 6-bits+1 and non-zero for values below 0x61
    VPSUBUSB Y13, Y3, Y3
    // Y3 = the third range mapped to 6-bits+1
    VPAND Y4, Y3, Y3

    VPOR Y2, Y1, Y0
    VPOR Y3, Y0, Y0
    // Y1 = the ranges mapped to 6-bits, out of range = 0xFF
    VPSUBB Y14, Y0, Y1
    // Save the bits for error checking later.
    VPOR Y1, Y5, Y5

    VMOVDQU Y1, (DI)
    ADDQ $32, SI
    ADDQ $28, DI


    RET
