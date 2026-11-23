package hnsw

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand/v2"
	"testing"
)

const hashRows, hashQueries, hashDim = 5000, 100, 50

// hashVectors returns rows of dim values, each row scaled to unit length.
// The values come from a PCG stream through integer arithmetic, a square
// root and a division, each exact or correctly rounded, so they are the
// same on every platform. The generator of the other tests, clustered, goes
// through NormFloat64, which calls math.Log and math.Exp; their code differs
// between amd64 and arm64 (math.Log is assembly on amd64 and Go on arm64,
// math.Exp has separate assembly on each).
func hashVectors(rows, dim int) []float32 {
	r := rand.NewPCG(1, 2)
	out := make([]float32, rows*dim)
	for i := range rows {
		row := out[i*dim : (i+1)*dim]
		var ss float64
		for j := range row {
			row[j] = float32(int32(r.Uint64()>>40)-1<<23) / (1 << 23) // uniform on [-1, 1)
			// The conversion rounds the product before the addition, so a
			// compiler that fuses a multiply with an add (Go allows it, and
			// arm64 does it) cannot fuse them here. The product of two
			// float32 values is exact in float64 in any case.
			ss += float64(float64(row[j]) * float64(row[j]))
		}
		norm := math.Sqrt(ss)
		for j := range row {
			row[j] = float32(float64(row[j]) / norm)
		}
	}
	return out
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// TestFirstVersionHashes pins the build, the file format and the search of
// the module's first version, the code of recserve at commit 35a47a2, where
// these hashes were taken (docs/DECISIONS.md D006). A change that fails them
// changes results; the hashes are never updated to let a change pass.
//
// Hashed are the input vectors, WriteTo's output for a one-goroutine build,
// and the hits of 100 queries at ef 10, 50, 100 and 400, and at ef 100 with
// odd IDs skipped, which takes the other branch of searchLayer's stopping
// rule. Each search asks for k = ef hits; a search for fewer runs the same
// way and returns a prefix of them.
//
// The values hold on amd64 with vec.Dot32 in AVX2 assembly and in Go
// (GODEBUG=cpu.avx2=off), and on arm64.
func TestFirstVersionHashes(t *testing.T) {
	all := hashVectors(hashRows+hashQueries, hashDim)
	var raw []byte
	for _, v := range all {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	if got, want := sha256Hex(raw), "3b5de973ea5f92633aa91dfc63c8316571e7b2ec9aab04b1ffc6bc65352f98e6"; got != want {
		t.Fatalf("input vectors: sha256 %s, want %s", got, want)
	}
	data, queries := all[:hashRows*hashDim], all[hashRows*hashDim:]
	ix, err := Build(data, hashDim, Config{M: 16, EfConstruction: 200, Seed: 1, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := ix.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	if got, want := sha256Hex(buf.Bytes()), "7def2b6bc20d4aadc9155efb95edbaac93d04ec2f5f91be035cbec335d40a04d"; got != want {
		t.Errorf("WriteTo: sha256 %s, want %s", got, want)
	}
	for _, c := range []struct {
		name string
		ef   int
		skip func(int32) bool
		want string
	}{
		{"ef 10", 10, nil, "7935e8a4dcb8200b269bcdfe10ab97908d00451a3b130b4e1fb27be6240e0d91"},
		{"ef 50", 50, nil, "1bef313ba12c57585d6bc5388ef0ea6a90c91dd2af59a2518fd3c5944b037e3a"},
		{"ef 100", 100, nil, "70133f94925d8462bc0fa77f10c31a6ebc92c801694741cc400f8c651fbb14a6"},
		{"ef 400", 400, nil, "927e126939028b9212f445a9020209193d2fa9c2f5d15b02bceb21a764f2dba7"},
		{"ef 100, odd IDs skipped", 100, oddIDs, "e1d19f1e00e9a948ff9f74638ee411c71cf437788f2494598173b9d3ec079119"},
	} {
		var b []byte
		for i := range hashQueries {
			for _, h := range ix.Search(queries[i*hashDim:(i+1)*hashDim], c.ef, c.ef, c.skip) {
				b = binary.LittleEndian.AppendUint32(b, uint32(h.ID))
				b = binary.LittleEndian.AppendUint64(b, math.Float64bits(h.Score))
			}
		}
		if got := sha256Hex(b); got != c.want {
			t.Errorf("%s: sha256 %s, want %s", c.name, got, c.want)
		}
	}
}
