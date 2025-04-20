package main

import (
	"crypto/tls"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync/atomic"
)

func main() {
	var port int
	flag.IntVar(&port, "port", 0, "server port")
	var certFile string
	flag.StringVar(&certFile, "cert", "server.crt", "path to certificate")
	var keyFile string
	flag.StringVar(&keyFile, "key", "server.key", "path to private key")
	var keepalive bool
	flag.BoolVar(&keepalive, "keepalive", false, "enable keep-alive")
	flag.Parse()

	var sum atomic.Uint64
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(writer http.ResponseWriter, request *http.Request) {
		defer func() {
			_ = request.Body.Close()
		}()
		buf, err := io.ReadAll(request.Body)
		if err != nil {
			log.Fatalf("could not read body: %v\n", err)
		}
		v := binary.LittleEndian.Uint32(buf)
		res := sum.Add(uint64(v))
		_, err = writer.Write(binary.LittleEndian.AppendUint64(nil, res))
		if err != nil {
			log.Fatalf("could not read body: %v\n", err)
		}
	})
	s := &http.Server{
		Addr:    fmt.Sprintf("localhost:%d", port),
		Handler: mux,
		// Disable HTTP/2
		TLSNextProto: make(map[string]func(*http.Server, *tls.Conn, http.Handler)),
	}
	s.SetKeepAlivesEnabled(keepalive)
	err := s.ListenAndServeTLS(certFile, keyFile)
	if err != nil {
		log.Fatalf("ListenAndServerTLS: %v", err)
	}
}
