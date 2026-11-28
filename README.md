# vecindex

Approximate nearest neighbor search in Go: `hnsw`, a hierarchical navigable small world graph (Malkov and Yashunin, arXiv:1603.09320) searched by float32 inner product, and `vec`, which holds the vector kernels it runs on and the `Hit` type of its results. In v0.1.0 the two packages are `internal/hnsw` and `internal/vec` of [recserve](https://github.com/sHyanorn/recserve) at commit 35a47a2, unchanged except for the import path and a package comment; the module adds tests. Later versions keep its float32 search and its file format bit for bit; the hash test below checks this on one configuration. The graph is built once over a fixed set of vectors: nothing can be added or removed after `Build`.

## Use

```bash
go get github.com/sHyanorn/vecindex
```

```go
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
// node 1, inner product 0.8
// node 2, inner product 0.0
```

The example is `hnsw/example_test.go`, so `go test` checks its output. `Search` returns the best k of the nodes it finds, by inner product, ties to the lower ID. `ef` is the size of the result list of the layer-0 search, raised to k if it is smaller: a larger ef is slower and finds more of the true neighbors. A skip function keeps nodes out of the results but not out of the traversal. A built index is safe for concurrent searches.

`WriteAligned` writes the same graph in a second format, which `View` uses in place: the index's arrays are the bytes of the buffer, so a file mapped into memory, for example with `syscall.Mmap`, serves searches without being read into the heap. The buffer must start at a multiple of 4, as a mapped file does, and stay mapped and unchanged while the index is in use; a mapped file truncated meanwhile can crash the process, so replace a file by writing a new one and renaming it. `View` checks the graph as `Read` does, reading its levels, offsets and every list before it returns. A viewed graph writes the same `WriteTo` file and returns the same hits as the graph it was written from, also from a file mapped read-only on Linux (`hnsw/aligned_test.go`, `hnsw/mmap_test.go`).

## What the tests pin

`hnsw/hash_test.go` builds a graph over 5,000 fixed vectors of 50 values with M 16, efConstruction 200, seed 1 and one goroutine. It checks the SHA-256 of the input (those vectors and 100 more used as queries), of `WriteTo`'s output, and of the hits of the 100 queries at ef 10, 50, 100 and 400, at ef 100 with odd IDs skipped, and at ef 10 with only the IDs divisible by 100 kept. The values were taken with recserve's code at 35a47a2. They hold with `vec.Dot32` in AVX2 assembly and in Go on amd64, and on arm64. The tests of `vec` check that `Dot` and `Dot32` equal their definitions to the bit, in Go and, on CPUs with AVX2 and FMA, in assembly, for lengths 0 to 300, and under `go test -fuzz` on random lengths and values. docs/DECISIONS.md D006 gives the reasons for each choice.

## Tests and benchmarks

```bash
go test -race ./...
GODEBUG=cpu.avx2=off go test ./...   # vec's Go kernels in place of the amd64 assembly
go test -run '^$' -bench . -benchmem ./vec
test -z "$(gofmt -l .)" && go vet ./...
```

CI runs gofmt, go vet and the race tests on amd64 and arm64, the tests once more on amd64 with AVX2 off, and each benchmark once.

## Limits

- No insertion or deletion after `Build`; a changed set of vectors needs a new build.
- Only a one-goroutine build is reproducible. A build with `Workers` above 1 can give a different graph each time.
- Pruning can remove every link to a node, so a few nodes may be unreachable on layer 0 and never returned: 3 of 3,000 in one of the tests' data sets, built with M 4. Many identical vectors make this much worse: most copies of a vector may never be found.
- Similarity is the inner product. For cosine, scale the vectors to unit length first.
- The file holds the graph without the vectors, and `Read` and `View` need the rows the graph was built over: they check their count and dimension, not their values. `Build`, `Read` and `View` keep the caller's slice without copying it. `WriteTo`'s file starts with the bytes `recserve-hnsw-1\n`, kept because the format is pinned; `WriteAligned`'s starts with `vecindex-hnsw-2\n`. `View` needs a little-endian CPU.
- Input is not validated further: a query must have the graph's dimension (a shorter one is scored on a prefix of each row, a longer one panics), and NaN values are not rejected.
- `vec.Matrix.TopK` searches IDs 1 and up, because recserve reserves row 0.
