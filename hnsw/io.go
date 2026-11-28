package hnsw

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unsafe"
)

// fileMagic starts a saved index; the layout follows it in little-endian:
// n, dim, M, Mmax0, efConstruction, entry, top and len(upper) as int64,
// then the levels as bytes and links0, upperOff and upper as int32.
const fileMagic = "recserve-hnsw-1\n"

// alignedMagic starts a graph written by WriteAligned. The same eight int64
// follow it, then links0, upperOff and upper as int32 and the levels as
// bytes, all little-endian. The magic and the header take 80 bytes, so every
// int32 lies at an offset that is a multiple of 4.
const alignedMagic = "vecindex-hnsw-2\n"

const headerLen = 8 * 8

// WriteTo writes the graph, without the vectors.
func (x *Index) WriteTo(w io.Writer) (int64, error) {
	return x.write(w, fileMagic, []any{x.levels, x.links0, x.upperOff, x.upper})
}

// WriteAligned writes the graph, without the vectors, in the layout that
// View reads in place.
func (x *Index) WriteAligned(w io.Writer) (int64, error) {
	return x.write(w, alignedMagic, []any{x.links0, x.upperOff, x.upper, x.levels})
}

func (x *Index) write(w io.Writer, magic string, body []any) (int64, error) {
	bw := bufio.NewWriter(w)
	cw := &countWriter{w: bw}
	head := []int64{int64(x.n), int64(x.dim), int64(x.m), int64(x.m0), int64(x.efc), int64(x.entry), int64(x.top), int64(len(x.upper))}
	if _, err := io.WriteString(cw, magic); err != nil {
		return cw.n, err
	}
	for _, v := range append([]any{head}, body...) {
		if err := binary.Write(cw, binary.LittleEndian, v); err != nil {
			return cw.n, err
		}
	}
	return cw.n, bw.Flush()
}

// newIndex checks the header that follows a magic against data and dim and
// returns an index without its arrays.
func newIndex(head []int64, data []float32, dim int) (*Index, error) {
	n, d, m, m0, efc, entry, top, nUpper := head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7]
	// The row count is compared with the data's, not multiplied by dim: a
	// product could wrap around to the data's length.
	switch {
	case int64(dim) != d || dim <= 0 || len(data)%dim != 0 || n != int64(len(data)/dim):
		return nil, fmt.Errorf("hnsw: index of %d rows of %d values, data has %d values of rows of %d", n, d, len(data), dim)
	case m < 2 || m > 1<<10 || (m0 != m && m0 != 2*m) || efc < 1 || top < 0 || top > maxLevel || nUpper < 0 || nUpper > n*maxLevel*(1+m):
		return nil, errors.New("hnsw: header values out of range")
	}
	return &Index{dim: dim, n: int(n), data: data, m: int(m), m0: int(m0), efc: int(efc), entry: int32(entry), top: int(top)}, nil
}

// ready checks the graph and makes it searchable.
func (x *Index) ready() (*Index, error) {
	if err := x.Check(); err != nil {
		return nil, err
	}
	x.pool.New = func() any { return x.newSearcher() }
	return x, nil
}

// Read reads a graph written by WriteTo and attaches it to data, the rows
// it was built over, dim values each. The graph is checked before use.
func Read(r io.Reader, data []float32, dim int) (*Index, error) {
	br := bufio.NewReader(r)
	magic := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, fmt.Errorf("hnsw: magic: %w", err)
	}
	if string(magic) != fileMagic {
		return nil, fmt.Errorf("hnsw: not a saved index: magic %q, want %q", magic, fileMagic)
	}
	head := make([]int64, 8)
	if err := binary.Read(br, binary.LittleEndian, head); err != nil {
		return nil, fmt.Errorf("hnsw: header: %w", err)
	}
	x, err := newIndex(head, data, dim)
	if err != nil {
		return nil, err
	}
	n, m0, nUpper := head[0], head[3], head[7]
	x.levels = make([]uint8, n)
	x.links0 = make([]int32, n*(1+m0))
	x.upperOff = make([]int32, n)
	x.upper = make([]int32, nUpper)
	for _, v := range []any{x.levels, x.links0, x.upperOff, x.upper} {
		if err := binary.Read(br, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("hnsw: body: %w", err)
		}
	}
	switch _, err := br.ReadByte(); {
	case err == nil:
		return nil, errors.New("hnsw: trailing data")
	case !errors.Is(err, io.EOF):
		return nil, fmt.Errorf("hnsw: after the graph: %w", err)
	}
	return x.ready()
}

// View returns the graph in buf, as WriteAligned wrote it, attached to data,
// the rows it was built over, dim values each. The index uses buf in place,
// without copying: buf must start at an address that is a multiple of 4, as
// a mapped file does, and stay mapped and unchanged while the index is in
// use. A mapped file that is truncated meanwhile crashes the process with
// SIGBUS; replace a file by writing a new one and renaming it. The graph is
// checked before use: Check reads the levels, the offsets and every list.
func View(buf []byte, data []float32, dim int) (*Index, error) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		return nil, errors.New("hnsw: View needs a little-endian CPU")
	}
	if len(buf) < len(alignedMagic)+headerLen {
		return nil, fmt.Errorf("hnsw: %d bytes hold no header", len(buf))
	}
	if magic := string(buf[:len(alignedMagic)]); magic != alignedMagic {
		return nil, fmt.Errorf("hnsw: not an aligned graph: magic %q, want %q", magic, alignedMagic)
	}
	if uintptr(unsafe.Pointer(&buf[0]))%4 != 0 {
		return nil, errors.New("hnsw: buffer does not start at a multiple of 4")
	}
	head := make([]int64, 8)
	for i := range head {
		head[i] = int64(binary.LittleEndian.Uint64(buf[len(alignedMagic)+8*i:]))
	}
	x, err := newIndex(head, data, dim)
	if err != nil {
		return nil, err
	}
	// newIndex bounds n by the data, M and Mmax0 by 2^11 and len(upper) by
	// n·31·2^10, so the size cannot wrap around in int64; once it equals
	// len(buf), every count fits in an int.
	n, n0, nUp := head[0], head[0]*(1+head[3]), head[7]
	off := len(alignedMagic) + headerLen
	if want := int64(off) + 4*(n0+n+nUp) + n; int64(len(buf)) != want {
		return nil, fmt.Errorf("hnsw: aligned graph of %d bytes, want %d", len(buf), want)
	}
	x.links0, off = int32s(buf, off, int(n0))
	x.upperOff, off = int32s(buf, off, int(n))
	x.upper, off = int32s(buf, off, int(nUp))
	x.levels = buf[off:]
	return x.ready()
}

// int32s returns the n int32 at byte offset off of buf, in place, and the
// offset after them; the caller has checked that buf holds them. With n 0,
// off may be len(buf), where &buf[off] would panic.
func int32s(buf []byte, off, n int) ([]int32, int) {
	if n == 0 {
		return nil, off
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&buf[off])), n), off + 4*n
}

type countWriter struct {
	w io.Writer
	n int64
}

func (c *countWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}
