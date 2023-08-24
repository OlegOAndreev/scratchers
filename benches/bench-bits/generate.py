#!/usr/bin/env python3

import random
import string

def random_string(min_len, max_len):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def partitions(s):
    ret = 1
    cur = set()
    for c in s:
        if c in cur:
            ret += 1
            cur = set([c])
        else:
            cur.add(c)
    return ret

random.seed(0)
for i in range(10000000):
    s = random_string(10, 100)
    print(s, partitions(s))
