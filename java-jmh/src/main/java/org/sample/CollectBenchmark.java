package org.sample;

import one.util.streamex.StreamEx;
import org.openjdk.jmh.annotations.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@OutputTimeUnit(TimeUnit.NANOSECONDS)
@BenchmarkMode(Mode.AverageTime)
public class CollectBenchmark {
    @State(Scope.Thread)
    public static class MyState {
        static final int N = 2500;
        static final int NUM_KEYS = 20;

        List<Integer> integerList = IntStream.range(0, N).boxed().collect(Collectors.toList());
        List<KeyValue> keyValues = IntStream.range(0, N)
                .mapToObj(_ -> {
                    var key = Integer.toString(ThreadLocalRandom.current().nextInt(NUM_KEYS));
                    var value = UUID.randomUUID().toString();
                    return new KeyValue(key, value);
                })
                .toList();
    }

    @Benchmark
    public int testIntegerListForeach(MyState state) {
        var mapped = new ArrayList<Integer>(state.integerList.size());
        for (Integer v : state.integerList) {
            mapped.add(v * 3);
        }
        return mapped.getLast();
    }

    @Benchmark
    public int testIntegerListMap(MyState state) {
        var mapped = state.integerList.stream()
                .map(i -> i * 3)
                .toList();
        return mapped.getLast();
    }

    @Benchmark
    public int testIntegerListMapWithCapacity(MyState state) {
        var mapped = state.integerList.stream()
                .map(i -> i * 3)
                .collect(Collectors.toCollection(() -> new ArrayList<>(state.integerList.size())));
        return mapped.getLast();
    }

    @Benchmark
    public int testGroupingStream(MyState state) {
//        var map = StreamEx.of(state.keyValues)
//                .mapToEntry(KeyValue::key, KeyValue::value)
//                .grouping();
        var map = state.keyValues.stream()
                .collect(Collectors.groupingBy(
                        KeyValue::key,
                        Collectors.mapping(KeyValue::value, Collectors.toList())));
        return map.get("0").getLast().hashCode();
    }

    @Benchmark
    public int testGroupingStreamEx(MyState state) {
        var map = StreamEx.of(state.keyValues)
                .mapToEntry(KeyValue::key, KeyValue::value)
                .grouping();
        return map.get("0").getLast().hashCode();
    }

    @Benchmark
    public int testGroupingForeach(MyState state) {
        var map = new HashMap<String, List<String>>();
        for (KeyValue kv : state.keyValues) {
            map.computeIfAbsent(kv.key(), _ -> new ArrayList<>()).add(kv.value());
        }
        return map.get("0").getLast().hashCode();
    }

    record KeyValue(String key, String value) {
    }
}
