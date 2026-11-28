//go:build linux

package hnsw

import (
	"bytes"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// TestViewMappedFile views a graph file mapped read-only, as a server maps
// it: any write into the mapping would crash the test.
func TestViewMappedFile(t *testing.T) {
	ix, data, queries := hashGraph(t)
	var file bytes.Buffer
	if _, err := ix.WriteAligned(&file); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "graph")
	if err := os.WriteFile(path, file.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	buf, err := syscall.Mmap(int(f.Fd()), 0, file.Len(), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(buf)
	v, err := View(buf, data, hashDim)
	if err != nil {
		t.Fatal(err)
	}
	sameGraph(t, ix, v)
	sameHits(t, ix, v, queries)
}
