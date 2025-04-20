package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"flag"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	var url string
	flag.StringVar(&url, "url", "", "server address")
	var certFile string
	flag.StringVar(&certFile, "cert", "server.crt", "path to certificate")
	var numRequests int
	flag.IntVar(&numRequests, "requests", 10000, "number of requests to run")
	var parallel int
	flag.IntVar(&parallel, "parallel", runtime.NumCPU(), "number of concurrent requests")
	flag.Parse()

	certPool := x509.NewCertPool()
	crt, err := os.ReadFile(certFile)
	if err != nil {
		log.Fatalf("could not read certificate: %v\n", err)
	}
	certPool.AppendCertsFromPEM(crt)
	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: parallel,
			TLSClientConfig: &tls.Config{
				RootCAs: certPool,
			},
		},
	}
	var wg sync.WaitGroup
	wg.Add(parallel)
	startTime := time.Now()
	var expected atomic.Uint64
	expected.Add(doRequest(client, url, 0))
	var res atomic.Uint64
	for i := 0; i < parallel; i++ {
		go func() {
			for j := 0; j < numRequests/parallel; j++ {
				v := rand.Int31()
				ret := doRequest(client, url, v)
				expected.Add(uint64(v))
				atomicMaxUint64(&res, ret)
			}
			defer wg.Done()
		}()
	}

	wg.Wait()
	if expected.Load() != res.Load() {
		log.Fatalf("Got wrong answer: %d instead of %d\n", res.Load(), expected.Load())
	}

	finalRequests := numRequests / parallel * parallel
	deltaTime := time.Since(startTime)
	rps := int(float64(finalRequests) / deltaTime.Seconds())
	log.Printf("%d requests in %v (%d rps)\n", finalRequests, deltaTime, rps)
}

func doRequest(client *http.Client, url string, v int32) uint64 {
	req := binary.LittleEndian.AppendUint32(nil, uint32(v))
	resp, err := client.Post(url, "", bytes.NewReader(req))
	if err != nil {
		log.Fatalf("could not make request: %v\n", err)
	}
	buf, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("could not read body: %v\n", err)
	}
	_ = resp.Body.Close()
	return binary.LittleEndian.Uint64(buf)
}

func atomicMaxUint64(a *atomic.Uint64, v uint64) bool {
	for {
		cur := a.Load()
		if cur >= v {
			return false
		}
		if a.CompareAndSwap(cur, v) {
			return true
		}
	}
}
