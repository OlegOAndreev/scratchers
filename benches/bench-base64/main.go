package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"time"
)

func main() {
	src := []byte("123")
	//src := []byte("1234567890abcdefgh")
	fmt.Printf("%q\n", string(Base64EncodeAvx2(nil, src)))
	fmt.Printf("%q\n", base64.StdEncoding.EncodeToString(src))
	os.Exit(1)

	cpuprofile := os.Getenv("CPUPROFILE")
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer func() {
			_ = f.Close()
		}()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	const totalSize = 1024 * 1024
	//const totalSize = 1024 * 1024 * 1024

	if len(os.Args) < 3 {
		log.Fatalf("Usage: %s <algo> <size>\n", os.Args[0])
	}
	algos := strings.Split(os.Args[1], ",")
	baseSize, err := strconv.Atoi(os.Args[2])
	if err != nil {
		log.Fatalf("Can't parse size %q\n", err)
	}

	log.Printf("Generating samples...\n")
	samples := generateSamples(baseSize, totalSize)
	var encodedSamples [][]byte
	for _, sample := range samples {
		encodedSamples = append(encodedSamples, base64.StdEncoding.AppendEncode(nil, sample))
	}
	output := makeOutput(samples)
	encodedOutput := makeOutput(encodedSamples)
	runtime.GC()

	log.Printf("Warming up...\n")
	for _, algo := range algos {
		checkAlgo(algo, samples, encodedSamples, output, encodedOutput, true)
	}
	log.Printf("Benchmarking...\n")
	for _, algo := range algos {
		checkAlgo(algo, samples, encodedSamples, output, encodedOutput, false)
	}
}

func makeOutput(samples [][]byte) [][]byte {
	var result [][]byte
	for _, sample := range samples {
		result = append(result, make([]byte, 0, len(sample)))
	}
	return result
}

func checkAlgo(algo string, samples [][]byte, encodedSamples [][]byte, output [][]byte, encodedOutput [][]byte,
	quiet bool,
) {
	switch algo {
	case "std":
		benchEncoding(samples, encodedSamples, encodedOutput, algo, quiet, func(dst []byte, src []byte) []byte {
			return base64.StdEncoding.AppendEncode(dst, src)
		})
		benchDecoding(encodedSamples, samples, output, algo, quiet, func(dst []byte, src []byte) []byte {
			result, err := base64.StdEncoding.AppendDecode(dst, src)
			if err != nil {
				log.Fatalf("could not decode %q: %v\n", string(src), err)
			}
			return result
		})
	case "purego":
		benchEncoding(samples, encodedSamples, encodedOutput, algo, quiet, func(dst []byte, src []byte) []byte {
			return Base64Encode(dst, src)
		})
		benchDecoding(encodedSamples, samples, output, algo, quiet, func(dst []byte, src []byte) []byte {
			result, err := Base64Decode(dst, src)
			if err != nil {
				log.Fatalf("could not decode %q: %v\n", string(src), err)
			}
			return result
		})
	case "avx2":
		benchEncoding(samples, encodedSamples, encodedOutput, algo, quiet, func(dst []byte, src []byte) []byte {
			return Base64EncodeAvx2(dst, src)
		})
		benchDecoding(encodedSamples, samples, output, algo, quiet, func(dst []byte, src []byte) []byte {
			result, err := Base64Decode(dst, src)
			if err != nil {
				log.Fatalf("could not decode %q: %v\n", string(src), err)
			}
			return result
		})
	}
}

func benchEncoding(samples [][]byte, encodedSamples [][]byte, encodedOutput [][]byte, algo string, silent bool,
	encode func(dst []byte, src []byte) []byte,
) {
	startTime := time.Now()
	count := 0
	for time.Since(startTime) < time.Second {
		for i := range samples {
			encodedOutput[i] = encode(encodedOutput[i][:0], samples[i])
		}
		count++
		if silent {
			return
		}
	}
	dt := time.Since(startTime) / time.Duration(count)

	//for i := range samples {
	//	if !bytes.Equal(encodedOutput[i], encodedSamples[i]) {
	//		log.Fatalf("Encoding with %s failed for %v: %q instead of %q\n", algo, samples[i], string(encodedOutput[i]),
	//			string(encodedSamples[i]))
	//	}
	//}

	totalSize := 0
	for i := range samples {
		totalSize += len(samples[i])
	}
	log.Printf("Encoded with %s in %dMb/sec\n", algo, int(float64(totalSize)/(dt.Seconds()*1024*1024)))
}

func benchDecoding(samples [][]byte, decodedSamples [][]byte, output [][]byte, algo string, silent bool,
	decode func(dst []byte, src []byte) []byte,
) {
	startTime := time.Now()
	count := 0
	for time.Since(startTime) < time.Second {
		for i := range samples {
			output[i] = decode(output[i][:0], samples[i])
		}
		count++
		if silent {
			return
		}
	}
	dt := time.Since(startTime) / time.Duration(count)

	for i := range samples {
		if !bytes.Equal(output[i], decodedSamples[i]) {
			log.Fatalf("Decoding with %s failed for %q: %v instead of %v\n", algo, samples[i], output[i],
				decodedSamples[i])
		}
	}

	totalSize := 0
	for i := range samples {
		totalSize += len(output[i])
	}
	log.Printf("Decoded with %s in %dMb/sec\n", algo, int(float64(totalSize)/(dt.Seconds()*1024*1024)))
}

type fixedRandSource struct {
}

func (f fixedRandSource) Uint64() uint64 {
	return 1234
}

func generateSamples(baseSize int, totalSize int) [][]byte {
	var result [][]byte
	size := 0
	rnd := rand.New(fixedRandSource{})
	for size < totalSize {
		nextSize := baseSize + rnd.IntN(baseSize)/10
		size += nextSize
		buf := make([]byte, nextSize)
		for i := 0; i < nextSize; i++ {
			buf[i] = byte(rnd.IntN(256))
		}
		result = append(result, buf)
	}
	return result
}
