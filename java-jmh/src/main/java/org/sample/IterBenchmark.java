package org.sample;

import org.openjdk.jmh.annotations.*;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@OutputTimeUnit(TimeUnit.NANOSECONDS)
@BenchmarkMode(Mode.AverageTime)
public class IterBenchmark {
    @State(Scope.Thread)
    public static class MyState {
        static final int N = 2500;

        int[] intArr1 = IntStream.range(0, N).toArray();
        int[] intArr2 = IntStream.range(N, N * 2).toArray();
        Integer[] integerArr1 = IntStream.range(0, N).boxed().toArray(Integer[]::new);
        Integer[] integerArr2 = IntStream.range(N, N * 2).boxed().toArray(Integer[]::new);
        List<Integer> integerList1 = IntStream.range(0, N).boxed().collect(Collectors.toList());
        List<Integer> integerList2 = IntStream.range(N, N * 2).boxed().collect(Collectors.toList());
    }

    @Benchmark
    public int testIntArr1(MyState state) {
        int v = 0;
        for (int i = 0; i < state.intArr1.length; i++) {
            v += state.intArr1[i] * 2;
        }
        return v;
    }

    @Benchmark
    public int testIntArr1Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.intArr1.length; i++) {
            v += state.intArr1[i] * state.intArr1[i] * state.intArr1[i];
        }
        return v;
    }

    @Benchmark
    public int testIntArr2(MyState state) {
        int v = 0;
        for (int i = 0; i < state.intArr1.length; i++) {
            v += state.intArr1[i] + state.intArr2[i];
        }
        return v;
    }

    @Benchmark
    public int testIntArr2Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.intArr1.length; i++) {
            v += state.intArr1[i] * state.intArr1[i] * state.intArr1[i]
                    + state.intArr2[i] * state.intArr2[i] * state.intArr2[i];
        }
        return v;
    }

    @Benchmark
    public int testIntegerArr1(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerArr1.length; i++) {
            v += state.integerArr1[i];
        }
        return v;
    }

    @Benchmark
    public int testIntegerArr1Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerArr1.length; i++) {
            v += state.integerArr1[i] * state.integerArr1[i] * state.integerArr1[i];
        }
        return v;
    }

    @Benchmark
    public int testIntegerArr2(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerArr1.length; i++) {
            v += state.integerArr1[i] + state.integerArr2[i];
        }
        return v;
    }

    @Benchmark
    public int testIntegerArr2Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerArr1.length; i++) {
            v += state.integerArr1[i] * state.integerArr1[i] * state.integerArr1[i]
                    + state.integerArr2[i] * state.integerArr2[i] * state.integerArr2[i];
        }
        return v;
    }

    @Benchmark
    public int testIntegerArr2Pow3v2(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerArr1.length; i++) {
            int j = state.integerArr1[i];
            int k = state.integerArr2[i];
            v += j * j * j + k * k * k;
        }
        return v;
    }

    @Benchmark
    public int testIntegerList1(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerList1.size(); i++) {
            v += state.integerList1.get(i);
        }
        return v;
    }

    @Benchmark
    public int testIntegerList1Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerList1.size(); i++) {
            v += state.integerList1.get(i) * state.integerList1.get(i) * state.integerList1.get(i);
        }
        return v;
    }

    @Benchmark
    public int testIntegerList2(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerList1.size(); i++) {
            v += state.integerList1.get(i) + state.integerList2.get(i);
        }
        return v;
    }

    @Benchmark
    public int testIntegerList2Pow3(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerList1.size(); i++) {
            v += state.integerList1.get(i) * state.integerList1.get(i) * state.integerList1.get(i)
                    + state.integerList2.get(i) * state.integerList2.get(i) * state.integerList2.get(i);
        }
        return v;
    }

    @Benchmark
    public int testIntegerList2Pow3v2(MyState state) {
        int v = 0;
        for (int i = 0; i < state.integerList1.size(); i++) {
            int j = state.integerList1.get(i);
            int k = state.integerList2.get(i);
            v += j * j * j + k * k * k;
        }
        return v;
    }

    @Benchmark
    public int testIntegerListForeach1(MyState state) {
        int v = 0;
        for (int j : state.integerList1) {
            v += j;
        }
        return v;
    }

    @Benchmark
    public int testIntegerListForeach1Pow3(MyState state) {
        int v = 0;
        for (int j : state.integerList1) {
            v += j * j * j;
        }
        return v;
    }

    @Benchmark
    public int testIntegerListStream1(MyState state) {
        return state.integerList1.stream().mapToInt(i -> i).sum();
    }

    @Benchmark
    public int testIntegerListStream1Pow3(MyState state) {
        return state.integerList1.stream().mapToInt(i -> i * i * i).sum();
    }
}
