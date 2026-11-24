package hnsw_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/sHyanorn/vecindex/hnsw"
)

// Build a graph over four unit vectors, save it, read it back over the same
// vectors and search it, leaving node 0 out.
func Example() {
	data := []float32{ // node i is row i
		1, 0,
		0.8, 0.6,
		0, 1,
		-1, 0,
	}
	ix, err := hnsw.Build(data, 2, hnsw.Config{M: 16, EfConstruction: 200, Seed: 1, Workers: 1})
	if err != nil {
		log.Fatal(err)
	}
	var file bytes.Buffer
	if _, err := ix.WriteTo(&file); err != nil {
		log.Fatal(err)
	}
	ix, err = hnsw.Read(&file, data, 2)
	if err != nil {
		log.Fatal(err)
	}
	skip := func(id int32) bool { return id == 0 }
	for _, h := range ix.Search([]float32{1, 0}, 2, 100, skip) {
		fmt.Printf("node %d, inner product %.1f\n", h.ID, h.Score)
	}
	// Output:
	// node 1, inner product 0.8
	// node 2, inner product 0.0
}
