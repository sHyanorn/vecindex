# CLAUDE.md

This repository is a Go module of two packages: `hnsw`, an HNSW graph searched by float32 inner product, and `vec`, the vector kernels it runs on. Other projects import it at pinned versions, so its results matter to the bit.

## Read before starting

1. `docs/DECISIONS.md`: decisions already made; do not overturn them on your own. To change one, explain why and get approval first
2. `README.md`: what the module guarantees and what the tests pin

## Working practices

- Write a plan before starting a change
- Commit messages in English following Conventional Commits, with a short subject and at most a brief body
- Append new technical decisions to DECISIONS.md
- Everything committed (code, comments, docs, commit messages) is in English and concise: say what changed and why, not the step-by-step process

## Common commands

```bash
go test -race ./...                            # all tests
GODEBUG=cpu.avx2=off go test ./...             # vec's Go kernels in place of the amd64 assembly
go test -run '^$' -bench . -benchmem ./vec     # benchmarks
go test -fuzz=FuzzDot32 -fuzztime=30s ./vec    # a kernel against its definition
test -z "$(gofmt -l .)" && go vet ./...
```

## Hard rules

- The float32 search and the file format stay bit for bit as in v0.1.0; new behavior is added beside the existing functions and formats
- The hash tests in `hnsw/hash_test.go` are never updated to make a change pass
- A kernel in assembly equals its Go code to the bit, and its tests compare both with the definition written out element by element
- In Go code that must match assembly to the bit, a product that is not exact is converted explicitly, so that the compiler cannot fuse it with an addition (D004)
- Performance changes carry before and after benchmarks in `bench-results/`, with the machine they ran on
