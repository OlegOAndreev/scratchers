package org.sample;

import org.openjdk.jmh.annotations.*;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.IntStream;

@BenchmarkMode(Mode.Throughput)
public class ReflectCallBenchmark {
    @State(Scope.Thread)
    public static class MyState {
        static final int N = 2500;
        static final int CAPACITY = 1024;

        List<MyRecord> records = IntStream.range(0, N)
                .mapToObj(_ -> new MyRecord(
                        "s1-" + UUID.randomUUID(),
                        "s2-" + UUID.randomUUID(),
                        "s3-" + UUID.randomUUID(),
                        "s4-" + UUID.randomUUID(),
                        "s5-" + UUID.randomUUID(),
                        "s6-" + UUID.randomUUID(),
                        "s7-" + UUID.randomUUID(),
                        "s8-" + UUID.randomUUID(),
                        "s9-" + UUID.randomUUID(),
                        "s10-" + UUID.randomUUID(),
                        "s11-" + UUID.randomUUID(),
                        "s12-" + UUID.randomUUID(),
                        "s13-" + UUID.randomUUID(),
                        "s14-" + UUID.randomUUID(),
                        "s15-" + UUID.randomUUID(),
                        "s16-" + UUID.randomUUID(),
                        "s17-" + UUID.randomUUID()
                ))
                .toList();

        Function<MyRecord, String> loopMethods = createLoopMethodsFunction();
        Function<MyRecord, String> loopFields = createLoopFieldsFunction();
        Function<MyRecord, String> chainMethods = createChainMethodsFunction();

        static {
            testMethods();
        }

        private static Method[] getGetters() {
            var getters = new ArrayList<Method>();
            for (var method : MyRecord.class.getMethods()) {
                if (method.getReturnType().equals(String.class) && !method.getName().equals("toString")) {
                    method.setAccessible(true);
                    getters.add(method);
                }
            }
            getters.sort(Comparator.comparing(Method::getName));
            return getters.toArray(new Method[0]);
        }

        private static Function<MyRecord, String> createLoopMethodsFunction() {
            var gettersArr = getGetters();
            return (obj) -> {
                var sb = new StringBuilder(CAPACITY);
                for (var getter : gettersArr) {
                    try {
                        sb.append(getter.invoke(obj));
                        sb.append(':');
                    } catch (IllegalAccessException | InvocationTargetException e) {
                        throw new RuntimeException(e);
                    }
                }
                return sb.toString();
            };
        }

        private static Function<MyRecord, String> createLoopFieldsFunction() {
            var fields = new ArrayList<Field>();
            for (var field : MyRecord.class.getDeclaredFields()) {
                if (field.getType().equals(String.class)) {
                    field.setAccessible(true);
                    fields.add(field);
                }
            }
            fields.sort(Comparator.comparing(Field::getName));
            var fieldsArr = fields.toArray(new Field[0]);
            return (obj) -> {
                var sb = new StringBuilder(CAPACITY);
                for (var field : fieldsArr) {
                    try {
                        sb.append(field.get(obj));
                        sb.append(':');
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    }
                }
                return sb.toString();
            };
        }

        private static Function<MyRecord, String> createChainMethodsFunction() {
            var getters = getGetters();
            BiConsumer<MyRecord, StringBuilder> appender = (obj, sb) -> {
                try {
                    sb.append(getters[0].invoke(obj));
                    sb.append(':');
                } catch (IllegalAccessException | InvocationTargetException e) {
                    throw new RuntimeException(e);
                }
            };

            for (int i = 1; i < getters.length; i++) {
                var prevAppender = appender;
                var getter = getters[i];
                appender = (obj, sb) -> {
                    prevAppender.accept(obj, sb);
                    try {
                        sb.append(getter.invoke(obj));
                        sb.append(':');
                    } catch (IllegalAccessException | InvocationTargetException e) {
                        throw new RuntimeException(e);
                    }
                };
            }
            var finalAppender = appender;
            return (obj) -> {
                var sb = new StringBuilder(CAPACITY);
                finalAppender.accept(obj, sb);
                return sb.toString();
            };
        }
    }

    @Benchmark
    public int testLoopMethods(MyState state) {
        int v = 0;
        for (MyRecord record : state.records) {
            v += state.loopMethods.apply(record).hashCode();
        }
        return v;
    }

    @Benchmark
    public int testLoopFields(MyState state) {
        int v = 0;
        for (MyRecord record : state.records) {
            v += state.loopFields.apply(record).hashCode();
        }
        return v;
    }

    @Benchmark
    public int testChainMethods(MyState state) {
        int v = 0;
        for (MyRecord record : state.records) {
            v += state.chainMethods.apply(record).hashCode();
        }
        return v;
    }

    record MyRecord(
            String s1,
            String s2,
            String s3,
            String s4,
            String s5,
            String s6,
            String s7,
            String s8,
            String s9,
            String s10,
            String s11,
            String s12,
            String s13,
            String s14,
            String s15,
            String s16,
            String s17
    ) {
    }

    private static void testMethods() {
        MyState state = new MyState();
        String result1 = state.chainMethods.apply(state.records.getFirst());
        String result2 = state.loopMethods.apply(state.records.getFirst());
        String result3 = state.loopFields.apply(state.records.getFirst());
        if (!result1.equals(result2)) {
            throw new RuntimeException("chainMethods got %s, but loopMethods got %s".formatted(result1, result2));
        }
        if (!result1.equals(result3)) {
            throw new RuntimeException("chainMethods got %s, but loopFields got %s".formatted(result1, result3));
        }
    }
}
