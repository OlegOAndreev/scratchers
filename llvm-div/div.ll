define i32 @my_div(i32, i32) {
entry:
    %eq.0 = icmp eq i32 %1, 0
    br i1 %eq.0, label %zero, label %non_zero

non_zero:
    %smaller.0 = icmp ult i32 %0, %1
    br i1 %smaller.0, label %zero, label %check_top_bit

check_top_bit:
    %has_top_bit.0 = and i32 %0, u0x80000000
    %not.0 = icmp eq i32 %has_top_bit.0, 0
    br i1 %not.0, label %find_max_mult_header, label %find_top_bit_header

find_top_bit_header:
    %top_bit.0 = phi i32 [ 1, %check_top_bit ], [ %top_bit.1, %find_top_bit_next ]
    %top_mult.0 = phi i32 [ %1, %check_top_bit ], [ %top_mult.1, %find_top_bit_next ]
    %has_top_bit.1 = and i32 %top_mult.0, u0x80000000
    %not.1 = icmp eq i32 %has_top_bit.1, 0
    br i1 %not.1, label %find_top_bit_next, label %found_top_bit

find_top_bit_next:
    %top_bit.1 = shl i32 %top_bit.0, 1
    %top_mult.1 = shl i32 %top_mult.0, 1
    br label %find_top_bit_header

found_top_bit:
    %smaller.3 = icmp ule i32 %top_mult.0, %0
    br i1 %smaller.3, label %found_top_bit_final, label %top_bit_shr

top_bit_shr:
    %top_bit.2 = lshr i32 %top_bit.0, 1
    %top_mult.2 = lshr i32 %top_mult.0, 1
    br label %found_top_bit_final

found_top_bit_final:
    %top_bit.3 = phi i32 [ %top_bit.0, %found_top_bit ], [ %top_bit.2, %top_bit_shr ]
    %top_mult.3 = phi i32 [ %top_mult.0, %found_top_bit ], [ %top_mult.2, %top_bit_shr ]
    %top_remaining_start = sub i32 %0, %top_mult.3
    br label %loop_bit_header

find_max_mult_header:
    %bit.0 = phi i32 [ 1, %check_top_bit ], [ %bit.1, %find_max_mult_next ]
    %mult.0 = phi i32 [ %1, %check_top_bit ], [ %mult.1, %find_max_mult_next ]
    %bit.1 = shl i32 %bit.0, 1
    %mult.1 = shl i32 %mult.0, 1
    br label %find_max_mult_next

find_max_mult_next:
    %smaller.1 = icmp ule i32 %mult.1, %0
    br i1 %smaller.1, label %find_max_mult_header, label %found_max_mult

found_max_mult:
    %remaining_start = sub i32 %0, %mult.0
    br label %loop_bit_header

loop_bit_header:
    %remaining.0 = phi i32 [ %remaining_start, %found_max_mult ], [ %remaining.2, %loop_bit_next ], [ %top_remaining_start, %found_top_bit_final ]
    %sum.0 = phi i32 [ %bit.0, %found_max_mult ], [ %sum.2, %loop_bit_next ], [ %top_bit.3, %found_top_bit_final ]
    %bit.2 = phi i32 [ %bit.0, %found_max_mult ], [ %bit.3, %loop_bit_next ], [ %top_bit.3, %found_top_bit_final ]
    %mult.2 = phi i32 [ %mult.0, %found_max_mult ], [ %mult.3, %loop_bit_next ], [ %top_mult.3, %found_top_bit_final ]
    %bit.3 = lshr i32 %bit.2, 1
    %eq.1 = icmp eq i32 %bit.3, 0
    br i1 %eq.1, label %end_bit, label %loop_bit_body

loop_bit_body:
    %mult.3 = lshr i32 %mult.2, 1
    %smaller.2 = icmp ult i32 %remaining.0, %mult.3
    br i1 %smaller.2, label %loop_bit_next, label %loop_bit_smaller

loop_bit_smaller:
    %remaining.1 = sub i32 %remaining.0, %mult.3
    %sum.1 = add i32 %sum.0, %bit.3
    br label %loop_bit_next

loop_bit_next:
    %remaining.2 = phi i32 [ %remaining.0, %loop_bit_body ], [ %remaining.1, %loop_bit_smaller ]
    %sum.2 = phi i32 [ %sum.0, %loop_bit_body ], [ %sum.1, %loop_bit_smaller ]
    br label %loop_bit_header

end_bit:
    ret i32 %sum.0

zero:
    ret i32 0
}
