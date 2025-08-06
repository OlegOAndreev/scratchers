package main

import (
	"bytes"
	"fmt"
)

const enableRedZone = true

var redZone = []byte("ThisIsTheRedZonePleaseDoNotOverwriteThisIsTheRedZonePleaseDoNotOverwrite")

func appendRedZone(b []byte) []byte {
	return append(b, redZone...)
}

func checkRedZone(b []byte) []byte {
	n := len(b) - len(redZone)
	if !bytes.Equal(b[n:], redZone) {
		panic(fmt.Sprintf("Red zone overwritten: expected %v, got %v", redZone, b[n:]))
	}
	return b[:n]
}
