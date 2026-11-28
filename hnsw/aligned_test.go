package hnsw

import (
	"bytes"
	"encoding/binary"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

// aligned returns a copy of b that starts at a multiple of 8, and a copy
// that starts one byte after one.
func aligned(b []byte) (ok, shifted []byte) {
	at := func(skip int) []byte {
		words := make([]uint64, len(b)/8+2)
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 8*len(words))
		out := raw[skip : skip+len(b) : skip+len(b)]
		copy(out, b)
		return out
	}
	return at(0), at(1)
}

// TestViewEqualsBuild views the graph of TestFirstVersionHashes, written by
// WriteAligned: the viewed graph writes the same WriteTo file as the built
// one, and its searches return the same hits in every case that test pins.
func TestViewEqualsBuild(t *testing.T) {
	all := hashVectors(hashRows+hashQueries, hashDim)
	data, queries := all[:hashRows*hashDim], all[hashRows*hashDim:]
	ix, err := Build(data, hashDim, Config{M: 16, EfConstruction: 200, Seed: 1, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	var file, built, viewed bytes.Buffer
	if _, err := ix.WriteAligned(&file); err != nil {
		t.Fatal(err)
	}
	buf, _ := aligned(file.Bytes())
	v, err := View(buf, data, hashDim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ix.WriteTo(&built); err != nil {
		t.Fatal(err)
	}
	if _, err := v.WriteTo(&viewed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(viewed.Bytes(), built.Bytes()) {
		t.Fatal("the viewed graph writes another file than the built one")
	}
	for _, c := range []struct {
		ef   int
		skip func(int32) bool
	}{{10, nil}, {50, nil}, {100, nil}, {400, nil}, {100, oddIDs}, {10, func(id int32) bool { return id%100 != 0 }}} {
		for i := range hashQueries {
			q := queries[i*hashDim : (i+1)*hashDim]
			if a, b := ix.Search(q, c.ef, c.ef, c.skip), v.Search(q, c.ef, c.ef, c.skip); !slices.Equal(a, b) {
				t.Fatalf("ef %d, query %d: viewed %v, built %v", c.ef, i, b, a)
			}
		}
	}
}

func TestViewRejects(t *testing.T) {
	var file, old bytes.Buffer
	if _, err := looseIx(t).WriteAligned(&file); err != nil {
		t.Fatal(err)
	}
	if _, err := looseIx(t).WriteTo(&old); err != nil {
		t.Fatal(err)
	}
	saved := file.Bytes()
	corrupt := slices.Clone(saved)
	corrupt[len(corrupt)-1] = 200 // the last node's level, above the cap
	// n, dim, M, Mmax0, efConstruction, entry, top, len(upper), where n
	// times dim wraps around to the 8 values of the data.
	wrapping := []byte(alignedMagic)
	for _, v := range []int64{4 - 1<<63, 2, 16, 32, 1, 0, 0, 0} {
		wrapping = binary.LittleEndian.AppendUint64(wrapping, uint64(v))
	}
	for name, c := range map[string]struct {
		buf  []byte
		data []float32
		dim  int
		want string
	}{
		"truncated":            {saved[:len(saved)-1], loose.data, loose.dim, "bytes, want"},
		"trailing":             {append(slices.Clone(saved), 0), loose.data, loose.dim, "bytes, want"},
		"corrupt":              {corrupt, loose.data, loose.dim, "level"},
		"in WriteTo's format":  {old.Bytes(), loose.data, loose.dim, "magic"},
		"without a header":     {saved[:40], loose.data, loose.dim, "no header"},
		"data of another size": {saved, loose.data[:len(loose.data)-loose.dim], loose.dim, "rows"},
		"wrapping row count":   {wrapping, make([]float32, 8), 2, "rows"},
	} {
		buf, _ := aligned(c.buf)
		if _, err := View(buf, c.data, c.dim); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error with %q", name, err, c.want)
		}
	}
	buf, shifted := aligned(saved)
	if _, err := View(shifted, loose.data, loose.dim); err == nil || !strings.Contains(err.Error(), "multiple of 4") {
		t.Errorf("misaligned buffer: got %v, want an error naming the alignment", err)
	}
	if _, err := View(buf, loose.data, loose.dim); err != nil {
		t.Errorf("the saved graph: %v", err)
	}
}
