package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"time"
	"unsafe"

	"golang.org/x/crypto/pbkdf2"

	"github.com/gotd/ige"
)

const authKeyLen = 256

func main() {
	passcode := flag.String("passcode", "", "passcode if there is any set")
	tdataPath := flag.String("tdata", "", "path to tdata directory")
	findFromStr := flag.String("from", "", "find files starting from date in YYYY-MM-DD format")
	findToStr := flag.String("to", "", "find files until date in YYYY-MM-DD format")
	largerThanKb := flag.Int("larger", 50, "find files larger than given number of kilobytes")
	outDir := flag.String("out", "", "directory to output files")
	flag.Parse()

	var findFrom time.Time
	var err error
	if *findFromStr != "" {
		findFrom, err = time.Parse(time.DateOnly, *findFromStr)
		if err != nil {
			log.Fatalf("Could not parse -from: %v\n", err)
		}
	}
	var findTo time.Time
	if *findToStr != "" {
		findTo, err = time.Parse(time.DateOnly, *findToStr)
		if err != nil {
			log.Fatalf("Could not parse -to: %v\n", err)
		}
	} else {
		findTo = time.Now().Add(time.Hour * 24)
	}
	largerThan := int64(*largerThanKb) * 1024

	keyDataPath := getKeyDataFile(*tdataPath)
	keyData, err := readTdfFile(keyDataPath)
	if err != nil {
		log.Fatalf("Could not read %s: %v\n", keyDataPath, err)
	}
	localKey, err := parseKeyData(keyData, *passcode)
	if err != nil {
		log.Fatalf("Could not parse key data from %s: %v\n", keyDataPath, err)
	}

	var filenames []string
	if len(flag.Args()) == 0 {
		defaultParam := getCachePath(*tdataPath)
		filenames = append(filenames, resolveParam(defaultParam, findFrom, findTo, largerThan)...)
	} else {
		for _, p := range flag.Args() {
			filenames = append(filenames, resolveParam(p, findFrom, findTo, largerThan)...)
		}
	}

	if *outDir == "" {
		if err := os.MkdirAll("out", 0755); err != nil {
			log.Fatalf("Could not create output directory: %v\n", err)
		}
		if err := os.Chdir("out"); err != nil {
			log.Fatalf("Could not chdir to output directory: %v\n", err)
		}
	} else {
		if err := os.Chdir(*outDir); err != nil {
			log.Fatalf("Could not chdir to output directory: %v\n", err)
		}
	}

	for _, filename := range filenames {
		if err := decryptTdef(filename, localKey); err != nil {
			log.Fatalf("Could not decrypt %s: %v\n", filename, err)
		}
	}
}

func getKeyDataFile(tdataPath string) string {
	if tdataPath == "" {
		tdataPath = getDefaultTdataPath()
	}
	return path.Join(tdataPath, "key_datas")
}

func getCachePath(tdataPath string) string {
	if tdataPath == "" {
		tdataPath = getDefaultTdataPath()
	}
	return path.Join(tdataPath, "user_data", "cache")
}

func getDefaultTdataPath() string {
	if runtime.GOOS == "darwin" {
		return os.Getenv("HOME") + "/Library/Containers/org.telegram.desktop/Data/Library/Application Support/Telegram Desktop/tdata"
	} else if runtime.GOOS == "linux" {
		return os.Getenv("HOME") + "/.local/share/TelegramDesktop/tdata"
	} else {
		log.Fatalf("Don't know the path to tdata, please ")
	}
	return ""
}

func resolveParam(p string, from time.Time, to time.Time, largerThan int64) []string {
	s, err := os.Stat(p)
	if os.IsNotExist(err) {
		log.Fatalf("File or directory %s does not exist\n", p)
	}
	if !s.IsDir() {
		return []string{p}
	}
	var ret []string
	err = filepath.Walk(p, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && info.ModTime().Before(to) && !info.ModTime().Before(from) && info.Size() >= largerThan {
			ret = append(ret, path)
		}
		return nil
	})
	if err != nil {
		log.Fatalf("Could not list %s: %v\n", p, err)
	}
	return ret
}

type tdfHeader struct {
	Magic   [4]byte
	Version uint32
}

func readTdfFile(filename string) ([]byte, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header tdfHeader
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("reading header: %v", err)
	}
	if string(header.Magic[:]) != "TDF$" {
		return nil, fmt.Errorf("wrong magic")
	}
	log.Printf("Key data version: %d\n", header.Version)

	contents, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	if len(contents) <= 16 {
		return nil, fmt.Errorf("got only %d bytes", len(contents))
	}
	const md5Len = 16
	data := contents[:len(contents)-md5Len]
	checksum := contents[len(contents)-md5Len:]
	h := md5.New()
	_, _ = h.Write(data)
	_ = binary.Write(h, binary.LittleEndian, uint32(len(data)))
	_ = binary.Write(h, binary.LittleEndian, header.Version)
	_, _ = h.Write(header.Magic[:])
	if !bytes.Equal(h.Sum(nil), checksum) {
		return nil, fmt.Errorf("wrong checksum")
	}

	return data, nil
}

func parseKeyData(keyData []byte, passcode string) ([]byte, error) {
	r := bytes.NewReader(keyData)
	var l uint32
	if err := binary.Read(r, binary.BigEndian, &l); err != nil {
		return nil, fmt.Errorf("could not read salt len: %v", err)
	}
	salt := make([]byte, l)
	if _, err := r.Read(salt); err != nil {
		return nil, fmt.Errorf("could not read salt: %v", err)
	}

	if err := binary.Read(r, binary.BigEndian, &l); err != nil {
		return nil, fmt.Errorf("could not read key len: %v", err)
	}
	keyEncrypted := make([]byte, l)
	if _, err := r.Read(keyEncrypted); err != nil {
		return nil, fmt.Errorf("could not read key: %v", err)
	}

	if err := binary.Read(r, binary.BigEndian, &l); err != nil {
		return nil, fmt.Errorf("could not read key len: %v", err)
	}
	if r.Len() != int(l) {
		return nil, fmt.Errorf("did not read all data: %d bytes remaining", r.Len()-int(l))
	}

	h := sha512.New()
	_, _ = h.Write(salt)
	_, _ = h.Write([]byte(passcode))
	_, _ = h.Write(salt)
	pass := h.Sum(nil)
	iterations := 1
	if passcode != "" {
		iterations = 100000
	}
	passcodeKey := pbkdf2.Key(pass, salt, iterations, authKeyLen, sha512.New)
	log.Printf("Got %d bytes salt from key data, %d bytes passcode key\n", len(salt), len(passcodeKey))

	localKeyHash := keyEncrypted[:16]
	localKeyEncryptedData := keyEncrypted[16:]
	localKeyData := aesDecryptLocal(passcodeKey, localKeyHash, localKeyEncryptedData)
	if !bytes.Equal(simpleSha1(localKeyData)[:16], localKeyHash) {
		return nil, fmt.Errorf("wrong decrypted key")
	}

	localKeyLen := binary.LittleEndian.Uint32(localKeyData[:4])
	log.Printf("Got local key len %d\n", localKeyLen-4)
	return localKeyData[4:localKeyLen], nil
}

type tdefPreHeader struct {
	Magic [4]byte
	Salt  [64]byte
}

type tdefHeader struct {
	Format   byte
	Reserved [15]byte
	Checksum [32]byte
}

func decryptTdef(filename string, key []byte) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var preHeader tdefPreHeader
	if err := binary.Read(f, binary.LittleEndian, &preHeader); err != nil {
		return fmt.Errorf("reading pre header: %v", err)
	}
	if string(preHeader.Magic[:]) != "TDEF" {
		return fmt.Errorf("wrong magic")
	}

	var header tdefHeader
	var headerBytes [unsafe.Sizeof(header)]byte
	if _, err := io.ReadFull(f, headerBytes[:]); err != nil {
		return fmt.Errorf("reading header: %v", err)
	}

	ctr := prepareCtrState(key, preHeader.Salt[:])
	decryptionOffset := 0

	ctr.Decrypt(headerBytes[:], decryptionOffset)
	decryptionOffset += len(headerBytes)
	if err := binary.Read(bytes.NewReader(headerBytes[:]), binary.LittleEndian, &header); err != nil {
		log.Fatalf("Internal error converting header bytes: %v", err)
	}
	h := sha256.New()
	_, _ = h.Write(key)
	_, _ = h.Write(preHeader.Salt[:])
	_, _ = h.Write([]byte{header.Format})
	_, _ = h.Write(header.Reserved[:])
	if !bytes.Equal(h.Sum(nil), header.Checksum[:]) {
		return fmt.Errorf("wrong header checksum")
	}

	if header.Format != 0 {
		return fmt.Errorf("unknown TDEF format %d", header.Format)
	}
	for _, c := range header.Reserved {
		if c != 0 {
			return fmt.Errorf("non-zero bytes in reserved header: %v", header.Reserved)
		}
	}

	data, err := io.ReadAll(f)
	if err != nil {
		return fmt.Errorf("could not read file: %v", err)
	}
	ctr.Decrypt(data, decryptionOffset)

	outName := outFilename(filename, data)
	if err := os.WriteFile(outName, data, 0666); err != nil {
		return fmt.Errorf("could not write file: %v", err)
	}

	fmt.Printf("All ok with %s\n", filename)

	return nil
}

func aesDecryptLocal(authKey []byte, msgKey []byte, ciphertext []byte) []byte {
	aesKey, aesIV := prepareAESOldMtp(authKey, msgKey, false)
	c, err := aes.NewCipher(aesKey)
	if err != nil {
		log.Fatalf("Internal error initializing AES cipher: %v\n", err)
	}
	plaintext := make([]byte, len(ciphertext))
	ige.DecryptBlocks(c, aesIV, plaintext, ciphertext)
	return plaintext
}

func prepareAESOldMtp(authKey []byte, msgKey []byte, send bool) (aesKey []byte, aesIV []byte) {
	x := 0
	if !send {
		x = 8
	}

	var data [48]byte
	copy(data[:16], msgKey[:16])
	copy(data[16:48], authKey[x:x+32])
	sha1A := simpleSha1(data[:])

	copy(data[:16], authKey[x+32:x+48])
	copy(data[16:32], msgKey[:16])
	copy(data[32:48], authKey[x+48:x+64])
	sha1B := simpleSha1(data[:])

	copy(data[:32], authKey[x+64:x+96])
	copy(data[32:48], msgKey[:16])
	sha1C := simpleSha1(data[:])

	copy(data[:16], msgKey[:16])
	copy(data[16:48], authKey[x+96:x+128])
	sha1D := simpleSha1(data[:])

	aesKey = make([]byte, 32)
	copy(aesKey[:8], sha1A[:8])
	copy(aesKey[8:20], sha1B[8:20])
	copy(aesKey[20:32], sha1C[4:16])
	aesIV = make([]byte, 32)
	copy(aesIV[:12], sha1A[8:20])
	copy(aesIV[12:20], sha1B[:8])
	copy(aesIV[20:24], sha1C[16:20])
	copy(aesIV[24:32], sha1D[:8])
	return
}

type ctrState struct {
	cipher cipher.Block
	key    []byte
	iv     []byte
	curIv  []byte
}

const ctrIvSize = 16

func prepareCtrState(key []byte, salt []byte) *ctrState {
	h := sha256.New()
	_, _ = h.Write(key[:len(key)/2])
	_, _ = h.Write(salt[:len(salt)/2])
	ctrKey := h.Sum(nil)
	c, err := aes.NewCipher(ctrKey)
	if err != nil {
		log.Fatalf("Internal error initializing AES cipher: %v\n", err)
	}

	h.Reset()
	_, _ = h.Write(key[len(key)/2:])
	_, _ = h.Write(salt[len(salt)/2:])
	ctrIv := h.Sum(nil)

	return &ctrState{
		cipher: c,
		key:    ctrKey,
		iv:     ctrIv[:ctrIvSize],
		curIv:  make([]byte, ctrIvSize),
	}
}

func (c *ctrState) Decrypt(ciphertext []byte, offset int) {
	const blockSize = 16
	blockIndex := offset / blockSize
	c.setCurIv(blockIndex)
	ctr := cipher.NewCTR(c.cipher, c.curIv)
	ctr.XORKeyStream(ciphertext, ciphertext)
}

func (c *ctrState) setCurIv(blockIndex int) {
	copy(c.curIv, c.iv)
	if blockIndex == 0 {
		return
	}
	digits := ctrIvSize
	increment := uint64(blockIndex)
	for {
		digits--
		increment += uint64(c.curIv[digits])
		c.curIv[digits] = byte(increment & 0xFF)
		increment >>= 8
		if digits == 0 || increment == 0 {
			break
		}
	}
}

func simpleSha1(data []byte) []byte {
	h := sha1.New()
	_, _ = h.Write(data)
	return h.Sum(nil)
}

func outFilename(p string, data []byte) string {
	s, err := os.Stat(p)
	if err != nil {
		log.Fatalf("Could not stat already read file: %v\n", err)
	}
	name := s.ModTime().Format("2006-01-02-15-04-05") + "-" + path.Base(p)

	var jpegPrefixes = [][]byte{
		{0xFF, 0xD8, 0xFF, 0xDB},
		{0xFF, 0xD8, 0xFF, 0xE0},
		{0xFF, 0xD8, 0xFF, 0xE1},
		{0xFF, 0xD8, 0xFF, 0xEE},
	}
	var pngPrefix = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	var gzipPrefix = []byte{0x1F, 0x8B}
	var mp4Prefix = []byte{0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D}
	var mkvPrefix = []byte{0x1A, 0x45, 0xDF, 0xA3}
	var riffPrefix = []byte{'R', 'I', 'F', 'F'}
	var webpPrefix = []byte{'W', 'E', 'B', 'P'}

	for _, p := range jpegPrefixes {
		if bytes.HasPrefix(data, p) {
			return name + ".jpg"
		}
	}
	if bytes.HasPrefix(data, pngPrefix) {
		return name + ".png"
	}
	if bytes.HasPrefix(data, gzipPrefix) {
		return name + ".gz"
	}
	if bytes.HasPrefix(data, mp4Prefix) || (len(data) > 4 && bytes.HasPrefix(data[4:], mp4Prefix)) {
		return name + ".mp4"
	}
	if bytes.HasPrefix(data, mkvPrefix) {
		return name + ".mkv"
	}
	if bytes.HasPrefix(data, riffPrefix) && len(data) > 8 && bytes.HasPrefix(data[8:], webpPrefix) {
		return name + ".webp"
	}
	return name + ".out"
}
