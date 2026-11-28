package hnsw

import (
	"bytes"
	"encoding/binary"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

// aligned returns a copy of b that starts at a multiple of 4, and a copy
// that starts one byte after one.
func aligned(b []byte) (ok, shifted []byte) {
	at := func(skip int) []byte {
		words := make([]uint32, len(b)/4+2)
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*len(words))
		out := raw[skip : skip+len(b) : skip+len(b)]
		copy(out, b)
		return out
	}
	return at(0), at(1)
}

// viewOf writes ix with WriteAligned and views it over data.
func viewOf(t *testing.T, ix *Index, data []float32, dim int) (*Index, []byte) {
	t.Helper()
	var file bytes.Buffer
	if _, err := ix.WriteAligned(&file); err != nil {
		t.Fatal(err)
	}
	buf, _ := aligned(file.Bytes())
	v, err := View(buf, data, dim)
	if err != nil {
		t.Fatal(err)
	}
	return v, buf
}

// sameGraph fails unless v writes the same WriteTo file as ix.
func sameGraph(t *testing.T, ix, v *Index) {
	t.Helper()
	var built, viewed bytes.Buffer
	if _, err := ix.WriteTo(&built); err != nil {
		t.Fatal(err)
	}
	if _, err := v.WriteTo(&viewed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(viewed.Bytes(), built.Bytes()) {
		t.Fatal("the viewed graph writes another file than the built one")
	}
}

// sameHits fails unless ix and v return the same hits for every query of
// TestFirstVersionHashes in each case it pins.
func sameHits(t *testing.T, ix, v *Index, queries []float32) {
	t.Helper()
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

func hashGraph(t *testing.T) (ix *Index, data, queries []float32) {
	t.Helper()
	all := hashVectors(hashRows+hashQueries, hashDim)
	data, queries = all[:hashRows*hashDim], all[hashRows*hashDim:]
	ix, err := Build(data, hashDim, Config{M: 16, EfConstruction: 200, Seed: 1, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	return ix, data, queries
}

// TestViewEqualsBuild views the graph of TestFirstVersionHashes: the viewed
// graph writes the same WriteTo file as the built one, returns the same
// hits in every case that test pins, and holds its arrays in buf itself.
func TestViewEqualsBuild(t *testing.T) {
	ix, data, queries := hashGraph(t)
	v, buf := viewOf(t, ix, data, hashDim)
	sameGraph(t, ix, v)
	sameHits(t, ix, v, queries)
	off := len(alignedMagic) + headerLen
	for name, at := range map[string]struct {
		got  unsafe.Pointer
		want int
	}{
		"links0":   {unsafe.Pointer(unsafe.SliceData(v.links0)), off},
		"upperOff": {unsafe.Pointer(unsafe.SliceData(v.upperOff)), off + 4*len(v.links0)},
		"upper":    {unsafe.Pointer(unsafe.SliceData(v.upper)), off + 4*(len(v.links0)+len(v.upperOff))},
		"levels":   {unsafe.Pointer(unsafe.SliceData(v.levels)), len(buf) - v.Len()},
	} {
		if at.got != unsafe.Pointer(&buf[at.want]) {
			t.Errorf("%s is not buf at offset %d", name, at.want)
		}
	}
}

// TestViewSizes views graphs whose node counts are not multiples of 4, so
// that levels placed before the int32 arrays would misalign them, and an
// empty graph.
func TestViewSizes(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 5, 1001} {
		d := newDataset(uint64(n)+7, n, 1, 8, 4, 0.3)
		ix, err := Build(d.data, d.dim, looseCfg)
		if err != nil {
			t.Fatal(err)
		}
		v, buf := viewOf(t, ix, d.data, d.dim)
		sameGraph(t, ix, v)
		base := uintptr(unsafe.Pointer(&buf[0]))
		for name, s := range map[string][]int32{"links0": v.links0, "upperOff": v.upperOff, "upper": v.upper} {
			if len(s) > 0 && (uintptr(unsafe.Pointer(&s[0]))-base)%4 != 0 {
				t.Errorf("n %d: %s at an offset that is not a multiple of 4", n, name)
			}
		}
		if a, b := ix.Search(d.query(0), 3, 10, nil), v.Search(d.query(0), 3, 10, nil); !slices.Equal(a, b) {
			t.Errorf("n %d: viewed %v, built %v", n, b, a)
		}
	}
}

// TestViewRejects gives View buffers that are cut, extended, corrupt, in
// WriteTo's format, misaligned or with a header that does not match the
// data: each is an error that names its cause.
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
