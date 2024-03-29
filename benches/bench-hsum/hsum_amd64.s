TEXT ·horizontalSumAvx2Asm(SB), $0-24
    MOVQ ptr+0(FP), SI
    MOVQ size+8(FP), CX

    // AX = accumulator, DX = number of 16-byte lines
    XORQ AX, AX
    MOVQ CX, DX
    SHRQ $4, DX
    JZ pre_short_loop

main_loop:
    // The main loop works with 16 i16 values. We sum up to 64 lines and then do a 16->8->4 i16 reduction
    // before transferring from SIMD to general register. Y0 is the accumulator.
    VPXOR Y0, Y0, Y0
    CMPQ DX, $64
    JB main_loop_remaining

    // Do a partially-unrolled addition, 4 16-element lines at each iteration.
    MOVQ $16, BX
    // Unsupported for now, retry with go 1.22: https://github.com/golang/go/issues/56474
    // PCALIGN $16
main_loop_64:
    VPMOVZXBW (SI), Y1
    VPMOVZXBW 16(SI), Y2
    VPMOVZXBW 32(SI), Y3
    VPMOVZXBW 48(SI), Y4
    ADDQ $64, SI
    VPADDW Y2, Y1, Y1
    VPADDW Y4, Y3, Y3
    VPADDW Y1, Y0, Y0
    VPADDW Y3, Y0, Y0
    DECQ BX
    JNZ main_loop_64
    SUBQ $64, DX
    JMP main_loop_reduce

main_loop_remaining:
    VPMOVZXBW (SI), Y1
    ADDQ $16, SI
    VPADDW Y1, Y0, Y0
    DECQ DX
    JNZ main_loop_remaining

main_loop_reduce:
    // Reduce Y0 and add to AX. First, reduce 16 -> 8 i16 values.
    VEXTRACTI128 $1, Y0, X1
    PADDW X1, X0
    // Reduce 8 -> 4 i16 values.
    MOVHLPS X0, X1
    PADDW X1, X0
    // Move to regular registers and do a i16 -> i32 reduction.
    MOVQ X0, R8
    MOVQ $0x0000FFFF0000FFFF, R10
    MOVQ R8, R9
    SHRQ $16, R9
    ANDQ R10, R8
    ANDQ R10, R9
    ADDQ R9, R8
    // Do a i32 -> i64 reduction.
    MOVQ R8, R9
    SHRQ $32, R9
    // The values in two halves are not enough to overflow the i32 range, simply zero out the top half.
    ADDL R9, R8
    ADDQ R8, AX

    TESTQ DX, DX
    JNZ main_loop

pre_short_loop:
    // CX = remainder after reading all 16-byte lines
    ANDQ $15, CX
    JZ finish

short_loop:
    MOVBLZX (SI), BX
    INCQ SI
    ADDQ BX, AX
    DECQ CX
    JNZ short_loop

finish:
    MOVQ AX, ret+16(FP)
    RET

TEXT ·horizontalSumAvx2AsmV2(SB), $0-24
    MOVQ ptr+0(FP), SI
    MOVQ size+8(FP), CX

    // AX = accumulator, DX = number of 16-byte lines
    XORQ AX, AX
    MOVQ CX, DX
    SHRQ $4, DX
    JZ pre_short_loop

main_loop:
    // The main loop works with 16 i16 values. We sum up to 64 lines and then do a 16->8->4 i16 reduction
    // before transferring from SIMD to general register. Y0 is the accumulator.
    VPXOR Y0, Y0, Y0
    CMPQ DX, $64
    JB main_loop_remaining

    // Do a partially-unrolled addition, 4 16-element lines at each iteration.
    MOVQ $16, BX
    // Unsupported for now, retry with go 1.22: https://github.com/golang/go/issues/56474
    // PCALIGN $16
main_loop_64:
    VPMOVZXBW (SI), Y1
    VPMOVZXBW 16(SI), Y2
    VPMOVZXBW 32(SI), Y3
    VPMOVZXBW 48(SI), Y4
    ADDQ $64, SI
    VPADDW Y2, Y1, Y1
    VPADDW Y4, Y3, Y3
    VPADDW Y1, Y0, Y0
    VPADDW Y3, Y0, Y0
    DECQ BX
    JNZ main_loop_64
    SUBQ $64, DX
    JMP main_loop_reduce

main_loop_remaining:
    VPMOVZXBW (SI), Y1
    ADDQ $16, SI
    VPADDW Y1, Y0, Y0
    DECQ DX
    JNZ main_loop_remaining

main_loop_reduce:
    // Unlike horizontalSumAvx2Asm, use horizontal add instructions.
    // Reduce Y0 and add to AX. First, reduce 16 -> 8 -> 4 i16 values.
    VEXTRACTI128 $1, Y0, X1
    PHADDW X1, X0
    PHADDW X0, X0
    // Now convert i16 -> i32 values and do a 4 -> 2 -> 1 reduction.
    VPMOVZXWD X0, X0
    PHADDD X0, X0
    PHADDD X0, X0
    MOVL X0, R8
    ADDQ R8, AX

    TESTQ DX, DX
    JNZ main_loop

pre_short_loop:
    // CX = remainder after reading all 16-byte lines
    ANDQ $15, CX
    JZ finish

short_loop:
    MOVBLZX (SI), BX
    INCQ SI
    ADDQ BX, AX
    DECQ CX
    JNZ short_loop

finish:
    MOVQ AX, ret+16(FP)
    RET
