package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("Usage: %s <data file> <algo>\n", os.Args[0])
	}
	filename := os.Args[1]
	algo := os.Args[2]

	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	startLoad := time.Now()
	scanner := bufio.NewScanner(f)
	var strs []string
	var partitions []int
	for scanner.Scan() {
		v := strings.Split(scanner.Text(), " ")
		if len(v) != 2 {
			continue
		}
		strs = append(strs, v[0])
		p, err := strconv.Atoi(v[1])
		if err != nil {
			log.Fatal(err)
		}
		partitions = append(partitions, p)
	}
	fmt.Printf("Parsed data in %v\n", time.Since(startLoad))

	ourPartitions := make([]int, len(partitions))
	startCompute := time.Now()
	switch algo {
	case "array":
		arrayPartition(strs, ourPartitions)
	case "bitmask":
		bitmaskPartition(strs, ourPartitions)
	case "array_noclear":
		arrayNoclearPartition(strs, ourPartitions)
	}
	computeTime := time.Since(startCompute)
	for i, v := range ourPartitions {
		if partitions[i] != v {
			log.Fatalf("Different partitions for %s (line %d): %d vs %d\n", strs[i], i, partitions[i], v)
		}
	}
	log.Printf("Computed using algo %s in %v\n", algo, computeTime)
}

func arrayPartition(strs []string, partitions []int) {
	var mask [256]bool
	for i, s := range strs {
		for j := 0; j < 256; j++ {
			mask[j] = false
		}
		ret := 1
		for _, b := range s {
			if mask[b] {
				for j := 0; j < 256; j++ {
					mask[j] = false
				}
				ret++
			}
			mask[b] = true
		}
		partitions[i] = ret
	}
}

func bitmaskPartition(strs []string, partitions []int) {
	for i, s := range strs {
		mask := 0
		ret := 1
		for _, b := range s {
			m := 1 << (b - 'A')
			if mask&m != 0 {
				mask = m
				ret++
			} else {
				mask |= m
			}
		}
		partitions[i] = ret
	}
}

func arrayNoclearPartition(strs []string, partitions []int) {
	var prev [256]int32
	for i, s := range strs {
		for j := 0; j < 256; j++ {
			prev[j] = 0
		}
		ret := 1
		for _, b := range s {
			if prev[b] == int32(ret) {
				ret++
			}
			prev[b] = int32(ret)
		}
		partitions[i] = ret
	}
}
