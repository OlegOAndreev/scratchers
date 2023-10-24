TEXT ·horizontalSumNeonAsm(SB), $0-24
    MOVD ptr+0(FP), R8
    MOVD size+8(FP), R9

    // R1 = accumulator, R10 = number of 16-byte lines
    MOVD ZR, R1
    MOVD R9, R10
    LSR $4, R10
    CBZ R10, pre_short_loop

main_loop:
    // The main loop works with 16 i16 values. We sum up to 32 lines and then do a 16->8->4 i16 reduction
    // before transferring from SIMD to general register. (V1, V2) are the accumulators.
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    CMP $32, R10
    BLO main_loop_remaining

    // Do a partially-unrolled addition. 4 16-element lines at each iteration.
    MOVD $8, R11
    PCALIGN $16
main_loop_32:
    // Read 64 i8 values and add zip them with zeros to make 16 i16 values in (V7, V8, V9, V10, V11, V12, V13, V14).
    VLD1.P 64(R8), [V3.B16, V4.B16, V5.B16, V6.B16]
    SUB $1, R11, R11
    VUADDW V3.B8, V1.H8, V1.H8
    VUADDW2 V3.B16, V2.H8, V2.H8
    VUADDW V4.B8, V1.H8, V1.H8
    VUADDW2 V4.B16, V2.H8, V2.H8
    VUADDW V5.B8, V1.H8, V1.H8
    VUADDW2 V5.B16, V2.H8, V2.H8
    VUADDW V6.B8, V1.H8, V1.H8
    VUADDW2 V6.B16, V2.H8, V2.H8
    CBNZ R11, main_loop_32
    SUB $32, R10, R10
    JMP main_loop_reduce

    PCALIGN $16
main_loop_remaining:
    // Read 16 i8 values and add zip them with zeros to make 16 i16 values in (V4, V5).
    VLD1.P 16(R8), [V3.B16]
    SUB $1, R10, R10
    VUADDW V3.B8, V1.H8, V1.H8
    VUADDW2 V3.B16, V2.H8, V2.H8
    CBNZ R10, main_loop_remaining

main_loop_reduce:
    // Reduce 16 i16 values in (V1, V2) to 8 i16 values in V1.
    VADD V2.H8, V1.H8, V1.H8
    // Reduce 8 i16 values in V1 to 4 values in lower half of V1.
    VMOV V1.D[1], V2.D[0]
    VADD V2.H4, V1.H4, V1.H4
    // Move to scalar regs and do i16 -> i32 reduction.
    VMOV V1.D[0], R2
    MOVD $0x0000FFFF0000FFFF, R4
    MOVD R2, R3
    LSR $16, R3
    AND R4, R2
    AND R4, R3
    ADD R3, R2, R2
    // Do a i32 -> i64 reduction. Zero out the top half.
    MOVD R2, R3
    LSR $32, R3
    ADDW R3, R2, R2
    ADD R2, R1, R1
    CBNZ R10, main_loop

pre_short_loop:
    // R9 = remainder after reading all 16-byte lines
    AND $15, R9
    CBZ R9, finish

short_loop:
    MOVBU.P 1(R8), R2
    SUB $1, R9, R9
    ADD R2, R1, R1
    CBNZ R9, short_loop

finish:
    MOVD R1, ret+16(FP)
    RET

TEXT ·horizontalSumNeonAsmV2(SB), $0-24
    MOVD ptr+0(FP), R8
    MOVD size+8(FP), R9

    // R1 = accumulator, R10 = number of 16-byte lines
    MOVD ZR, R1
    MOVD R9, R10
    LSR $4, R10
    CBZ R10, pre_short_loop

main_loop:
    // The main loop works with 16 i16 values. We sum up to 32 lines and then do a 16->8->4 i16 reduction
    // before transferring from SIMD to general register. (V1, V2) are the accumulators.
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    CMP $32, R10
    BLO main_loop_remaining

    // Do a partially-unrolled addition. 4 16-element lines at each iteration.
    MOVD $8, R11
    PCALIGN $16
main_loop_32:
    // Read 64 i8 values.
    VLD1.P 64(R8), [V3.B16, V4.B16, V5.B16, V6.B16]
    SUB $1, R11, R11
    VUADDW V3.B8, V1.H8, V1.H8
    VUADDW2 V3.B16, V2.H8, V2.H8
    VUADDW V4.B8, V1.H8, V1.H8
    VUADDW2 V4.B16, V2.H8, V2.H8
    VUADDW V5.B8, V1.H8, V1.H8
    VUADDW2 V5.B16, V2.H8, V2.H8
    VUADDW V6.B8, V1.H8, V1.H8
    VUADDW2 V6.B16, V2.H8, V2.H8
    CBNZ R11, main_loop_32
    SUB $32, R10, R10
    JMP main_loop_reduce

    PCALIGN $16
main_loop_remaining:
    // Read 16 i8 values.
    VLD1.P 16(R8), [V3.B16]
    SUB $1, R10, R10
    VUADDW V3.B8, V1.H8, V1.H8
    VUADDW2 V3.B16, V2.H8, V2.H8
    CBNZ R10, main_loop_remaining

main_loop_reduce:
    // Reduce 16 i16 values in (V1, V2) to 8 i16 values in V1.
    VADD V2.H8, V1.H8, V1.H8
    // Sum 8 i16 values in V1. Unlike horizontalSumNeonAsm, uses special horizontal add instruction.
    VUADDLV V1.H8, V2
    VMOV V2.D[0], R2
    ADD R2, R1, R1
    CBNZ R10, main_loop

pre_short_loop:
    // R9 = remainder after reading all 16-byte lines
    AND $15, R9
    CBZ R9, finish

short_loop:
    MOVBU.P 1(R8), R2
    SUB $1, R9, R9
    ADD R2, R1, R1
    CBNZ R9, short_loop

finish:
    MOVD R1, ret+16(FP)
    RET
