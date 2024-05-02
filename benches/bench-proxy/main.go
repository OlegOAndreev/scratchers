package main

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/crypto/ed25519"
	"golang.org/x/crypto/ssh"
	"golang.org/x/sys/unix"
)

const (
	sshUser     = "bench-proxy-user"
	sshPassword = "bench-proxy-very-safe-password-123456"
)

func main() {
	sendAddr := flag.String("send", "", "Send data, must be host:port")
	sshSendAddr := flag.String("ssh-send", "", "Send data to SSH server, must be host:port")
	proxyAddr := flag.String("proxy", "", "Receive and proxy data, must be port:host:port")
	sshProxyPort := flag.Int("ssh-proxy", 0, "Receive ssh connections and process direct-tcpip requests, must be port")
	recvPort := flag.Int("recv", 0, "Receive data, must be port")
	sshRecvPort := flag.Int("ssh-recv", 0, "Receive data as SSH server, must be port")
	// Default buffer size from io.Copy
	bufKb := flag.Int("buf", 32, "Size of buffer to send/proxy/recv, must be in kilobytes")
	sockBufKb := flag.Int("sock-buf", 0, "Set custom buffer for socket send/recv")
	amountMb := flag.Int("amount", 100, "Amount of data to send, must be in megabytes")
	numConns := flag.Int("num-conns", 1, "Send data in N concurrent connections")
	sleepAfter := flag.Int("sleep-after", 0, "Number of seconds to sleep after the last sent byte")
	flag.Parse()

	if *sendAddr != "" {
		multiSend(*numConns, *amountMb*1024*1024, func(amount int) {
			sendTo(*sendAddr, amount, *bufKb*1024, *sockBufKb*1024, time.Duration(*sleepAfter)*time.Second)
		})
	} else if *sshSendAddr != "" {
		multiSend(*numConns, *amountMb*1024*1024, func(amount int) {
			sshSendTo(*sshSendAddr, amount, *bufKb*1024, *sockBufKb*1024, time.Duration(*sleepAfter)*time.Second)
		})
	} else if *recvPort != 0 {
		recvAt(*recvPort, *bufKb*1024, *sockBufKb*1024)
	} else if *sshRecvPort != 0 {
		sshRecvAt(*sshRecvPort, *bufKb*1024, *sockBufKb*1024)
	} else if *proxyAddr != "" {
		s := strings.SplitN(*proxyAddr, ":", 2)
		port, err := strconv.Atoi(s[0])
		if err != nil {
			log.Fatalf("Could not parse port %s: %v\n", s[0], err)
		}
		destAddr := s[1]
		proxyFromTo(port, destAddr, *bufKb*1024, *sockBufKb*1024)
	} else if *sshProxyPort != 0 {
		log.Fatalf("Unimplemented\n")
	} else {
		log.Fatalf("At least one mode of operations must be chosen\n")
	}
}

func multiSend(numConns int, amount int, send func(amount int)) {
	startTime := time.Now()
	var wg sync.WaitGroup
	wg.Add(numConns)
	for i := 0; i < numConns; i++ {
		go func() {
			defer wg.Done()
			send(amount / numConns)
		}()
	}
	wg.Wait()
	dt := time.Since(startTime)
	log.Printf("Sent %d bytes in %v with speed %.1fMb/sec\n", amount, dt, float64(amount)/(1024.0*1024.0*dt.Seconds()))
}

func sendTo(destAddr string, amount int, bufSize int, sockBufSize int, sleepAfter time.Duration) {
	d := net.Dialer{
		Control: getSockBufControl(sockBufSize),
	}
	conn, err := d.Dial("tcp", destAddr)
	if err != nil {
		log.Fatalf("Failed to dial %s: %v\n", destAddr, err)
	}
	defer conn.Close()

	sendImpl(conn, conn, amount, bufSize, destAddr, sleepAfter)
}

func sshSendTo(destAddr string, amount int, bufSize int, sockBufSize int, sleepAfter time.Duration) {
	d := net.Dialer{
		Control: getSockBufControl(sockBufSize),
	}
	conn, err := d.Dial("tcp", destAddr)
	if err != nil {
		log.Fatalf("Failed to dial %s: %v\n", destAddr, err)
	}
	sshC, sshChans, sshReqs, err := ssh.NewClientConn(conn, destAddr, &ssh.ClientConfig{
		User: sshUser,
		Auth: []ssh.AuthMethod{
			ssh.Password(sshPassword),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
	})
	if err != nil {
		log.Fatalf("Failed to open ssh connection to %s: %v\n", destAddr, err)
	}
	cl := ssh.NewClient(sshC, sshChans, sshReqs)
	defer cl.Close()

	sess, err := cl.NewSession()
	if err != nil {
		log.Fatalf("Failed to open ssh session")
	}
	defer sess.Close()
	stdin, err := sess.StdinPipe()
	if err != nil {
		log.Fatalf("Failed to get stdin for ssh shell: %v\n", err)
	}
	stdout, err := sess.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to get stdout for ssh shell: %v\n", err)
	}
	err = sess.Shell()
	if err != nil {
		log.Fatalf("Failed to start ssh shell: %v\n", err)
	}
	sendImpl(stdin, stdout, amount, bufSize, destAddr, sleepAfter)
}

func sendImpl(w io.Writer, r io.Reader, amount int, bufSize int, dest string, sleepAfter time.Duration) {
	log.Printf("Sending %d bytes to %s...\n", amount, dest)
	err := writeUint64(w, uint64(amount))
	if err != nil {
		log.Fatalf("Failed to write amount to %s: %v\n", dest, err)
	}

	buf := make([]byte, bufSize)
	h := fnv.New64()
	for j := 0; j < bufSize; j++ {
		buf[j] = byte(j & 0xFF)
	}
	for i := 0; i < amount; i += bufSize {
		toWrite := bufSize
		if toWrite > amount-i {
			toWrite = amount - i
		}
		_, err = w.Write(buf[:toWrite])
		if err != nil {
			log.Fatalf("Failed to write to %s: %v\n", dest, err)
		}
		_, _ = h.Write(buf[:toWrite])
	}

	ourSum := h.Sum64()

	gotSum, err := readUint64(r)
	if err != nil {
		log.Fatalf("Failed to read sum from %s: %v\n", dest, err)
	}
	if ourSum != gotSum {
		log.Fatalf("Checksum mismatch: %d vs %d\n", ourSum, gotSum)
	}

	time.Sleep(sleepAfter)
}

func recvAt(port int, bufSize int, sockBufSize int) {
	lc := net.ListenConfig{
		Control: getSockBufControl(sockBufSize),
	}
	l, err := lc.Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Failed to listen at %d: %v\n", port, err)
	}
	log.Printf("Listening at :%d...\n", port)

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Fatalf("Failed to accept at %d: %v\n", port, err)
		}
		go func() {
			defer conn.Close()

			recvImpl(conn, conn, bufSize, port)
		}()
	}
}

func sshRecvAt(port int, bufSize int, sockBufSize int) {
	lc := net.ListenConfig{
		Control: getSockBufControl(sockBufSize),
	}
	l, err := lc.Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Failed to listen at %d: %v\n", port, err)
	}
	log.Printf("Listening at :%d...\n", port)

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Fatalf("Failed to accept at %d: %v\n", port, err)
		}
		go func() {
			defer conn.Close()
			conf := &ssh.ServerConfig{
				PasswordCallback: func(conn ssh.ConnMetadata, password []byte) (*ssh.Permissions, error) {
					if conn.User() == sshUser && string(password) == sshPassword {
						return &ssh.Permissions{}, nil
					}
					return nil, errors.New("wrong user or password")
				},
			}
			_, privKey, err := ed25519.GenerateKey(rand.Reader)
			if err != nil {
				log.Printf("Failed to open generate ssh key: %v\n", err)
				return
			}
			signer, err := ssh.NewSignerFromKey(privKey)
			if err != nil {
				log.Printf("Failed to open generate ssh key: %v\n", err)
				return
			}
			conf.AddHostKey(signer)
			sConn, channels, requests, err := ssh.NewServerConn(conn, conf)
			if err != nil {
				log.Printf("Failed to open ssh conn: %v\n", err)
				return
			}
			defer sConn.Close()
			go ssh.DiscardRequests(requests)

			newChannel := <-channels
			if newChannel == nil || newChannel.ChannelType() != "session" {
				log.Printf("Got wrong channel type: %v\n", newChannel.ChannelType())
				_ = newChannel.Reject(ssh.UnknownChannelType, "only one session is accepted")
				return
			}

			channel, chRequests, err := newChannel.Accept()
			if err != nil {
				log.Printf("Failed to open ssh session: %v\n", err)
				return
			}
			req := <-chRequests
			log.Printf("Got request %s\n", req.Type)

			err = req.Reply(true, nil)
			if err != nil {
				log.Printf("Could not reply to request: %v\n", err)
				return
			}
			go ssh.DiscardRequests(chRequests)

			recvImpl(channel, channel, bufSize, port)
		}()
	}
}

func getSockBufControl(sockBufSize int) func(network, address string, c syscall.RawConn) error {
	return func(network, address string, c syscall.RawConn) error {
		return c.Control(func(fd uintptr) {
			if sockBufSize != 0 {
				if err := unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_RCVBUF, sockBufSize); err != nil {
					log.Printf("Setting SO_RCVBUF failed: %v\n", err)
				}
				if err := unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_SNDBUF, sockBufSize); err != nil {
					log.Printf("Setting SO_SNDBUF failed: %v\n", err)
				}
			}
			rcvBufSize, err := unix.GetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_RCVBUF)
			if err != nil {
				log.Printf("Reading SO_RCVBUF failed: %v\n", err)
			} else {
				log.Printf("SO_RCVBUF: %d\n", rcvBufSize)
			}
			sndBufSize, err := unix.GetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_SNDBUF)
			if err != nil {
				log.Printf("Reading SO_SNDBUF failed: %v\n", err)
			} else {
				log.Printf("SO_SNDBUF: %d\n", sndBufSize)
			}
		})
	}
}

func recvImpl(w io.Writer, r io.Reader, bufSize int, port int) {
	startTime := time.Now()
	amount, err := readUint64(r)
	if err != nil {
		log.Printf("Failed to read amount at %d: %v\n", port, err)
		return
	}

	buf := make([]byte, bufSize)
	h := fnv.New64()
	for i := 0; i < int(amount); i += bufSize {
		toRead := bufSize
		if toRead > int(amount)-i {
			toRead = int(amount) - i
		}
		_, err = io.ReadFull(r, buf[:toRead])
		if err != nil {
			log.Printf("Failed to read at %d: %v\n", port, err)
			return
		}
		_, _ = h.Write(buf[:toRead])
	}
	gotSum := h.Sum64()

	err = writeUint64(w, gotSum)
	if err != nil {
		log.Printf("Failed to write sum at %d: %v\n", port, err)
		return
	}
	log.Printf("Processed %d bytes in %v\n", amount, time.Since(startTime))
}

func proxyFromTo(port int, destAddr string, bufSize int, sockBufSize int) {
	lc := net.ListenConfig{
		Control: getSockBufControl(sockBufSize),
	}
	l, err := lc.Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Failed to listen at %d: %v\n", port, err)
	}
	log.Printf("Listening at :%d and copying to %s...\n", port, destAddr)

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Fatalf("Failed to accept at %d: %v\n", port, err)
		}
		go func() {
			defer conn.Close()
			d := net.Dialer{
				Control: getSockBufControl(sockBufSize),
			}
			targetConn, err := d.Dial("tcp", destAddr)
			if err != nil {
				log.Printf("Failed to dial %s: %v\n", destAddr, err)
				return
			}
			defer targetConn.Close()

			startTime := time.Now()
			errCh := make(chan error)
			go func() {
				_, err := io.CopyBuffer(conn, targetConn, make([]byte, bufSize))
				errCh <- err
			}()
			_, err = io.CopyBuffer(targetConn, conn, make([]byte, bufSize))
			if err != nil {
				log.Printf("Error copying from client to target: %v\n", err)
			}
			err = <-errCh
			if err != nil {
				log.Printf("Error copying from target to client: %v\n", err)
			}
			log.Printf("Copied in %v\n", time.Since(startTime))
		}()
	}
}

func readUint64(r io.Reader) (uint64, error) {
	var buf [8]byte
	_, err := io.ReadFull(r, buf[:])
	if err != nil {
		return 0, fmt.Errorf("failed to read uint64: %w", err)
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func writeUint64(w io.Writer, v uint64) error {
	_, err := w.Write(binary.LittleEndian.AppendUint64(nil, v))
	if err != nil {
		return fmt.Errorf("failed to write uint64: %w", err)
	}
	return nil
}
