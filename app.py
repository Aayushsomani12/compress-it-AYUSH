#!/usr/bin/env python3
"""
compress_tool.py
Single-file compression utility with:
- Huffman coding (bit-packed)
- LZW (16-bit codes)
- Simple RLE (escape-based)
- CLI + PyQt5 GUI (dark glassmorphic / neumorphic styling)

Usage (CLI):
  python compress_tool.py compress input.bin --alg huffman
  python compress_tool.py decompress input.bin.cmp
  python compress_tool.py --gui
"""

import sys
import os
import argparse
import struct
from collections import Counter, deque
import heapq
from typing import Dict, Tuple, Optional, List

# GUI imports (optional; only required if --gui)
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

MAGIC = b"CMPV1"
ALG_HUFFMAN = 1
ALG_LZW = 2
ALG_RLE = 3

# ---------------------------
# Bit writer / reader helpers
# ---------------------------
class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self._current = 0
        self._count = 0  # number of bits in current (0..7)

    def write_bit(self, bit: int):
        if bit not in (0,1):
            raise ValueError("bit must be 0 or 1")
        self._current = (self._current << 1) | bit
        self._count += 1
        if self._count == 8:
            self.buffer.append(self._current)
            self._current = 0
            self._count = 0

    def write_bits_from_str(self, s: str):
        for ch in s:
            self.write_bit(1 if ch == '1' else 0)

    def flush(self):
        if self._count > 0:
            # pad remaining bits on the right (low bits)
            self._current <<= (8 - self._count)
            self.buffer.append(self._current)
            self._current = 0
            self._count = 0

    def get_bytes(self) -> bytes:
        return bytes(self.buffer)

    def bit_length(self) -> int:
        return len(self.buffer) * 8  # after flush you'd want to compute correctly

class BitReader:
    def __init__(self, data: bytes, bit_length: int):
        self.data = data
        self.bit_length = bit_length
        self._pos = 0  # bit index

    def read_bit(self) -> Optional[int]:
        if self._pos >= self.bit_length:
            return None
        byte_index = self._pos // 8
        bit_index = 7 - (self._pos % 8)  # we wrote MSB-first
        val = (self.data[byte_index] >> bit_index) & 1
        self._pos += 1
        return val

    def read_bits(self, n: int) -> List[int]:
        res = []
        for _ in range(n):
            b = self.read_bit()
            if b is None:
                raise EOFError("Not enough bits")
            res.append(b)
        return res

# ---------------------------
# Huffman implementation
# ---------------------------
class HuffNode:
    __slots__ = ("freq","byte","left","right")
    def __init__(self, freq:int, byte:Optional[int]=None, left=None, right=None):
        self.freq = freq
        self.byte = byte
        self.left = left
        self.right = right
    def is_leaf(self): return self.byte is not None
    # for heapq
    def __lt__(self, other): return self.freq < other.freq

def build_huffman_tree(data: bytes) -> HuffNode:
    freq = Counter(data)
    heap = []
    for b, f in freq.items():
        heapq.heappush(heap, HuffNode(f, b))
    if len(heap) == 0:
        return HuffNode(0, 0)  # degenerate
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = HuffNode(a.freq + b.freq, None, a, b)
        heapq.heappush(heap, parent)
    return heap[0]

def make_codes(node: HuffNode, prefix: str="", d: Dict[int,str]=None) -> Dict[int,str]:
    if d is None:
        d = {}
    if node.is_leaf():
        d[node.byte] = prefix or "0"  # single-symbol edge case
    else:
        make_codes(node.left, prefix + "0", d)
        make_codes(node.right, prefix + "1", d)
    return d

def serialize_huffman_tree(node: HuffNode) -> bytes:
    # Preorder: 0x00 for internal, 0x01 + byte for leaf
    out = bytearray()
    def walk(n: HuffNode):
        if n.is_leaf():
            out.append(0x01)
            out.append(n.byte)
        else:
            out.append(0x00)
            walk(n.left)
            walk(n.right)
    walk(node)
    return bytes(out)

def deserialize_huffman_tree(buf: bytes) -> HuffNode:
    # Return node and new offset
    idx = 0
    def walk() -> HuffNode:
        nonlocal idx
        if idx >= len(buf):
            raise ValueError("Malformed Huffman tree")
        marker = buf[idx]; idx += 1
        if marker == 0x01:
            if idx >= len(buf):
                raise ValueError("Malformed Huffman tree leaf")
            b = buf[idx]; idx += 1
            return HuffNode(0, b)
        elif marker == 0x00:
            left = walk()
            right = walk()
            return HuffNode(0, None, left, right)
        else:
            raise ValueError("Bad marker in tree")
    node = walk()
    return node

def huffman_compress_bytes(data: bytes) -> Tuple[bytes, bytes, int]:
    # returns (tree_bytes, payload_bytes, payload_bit_length)
    tree = build_huffman_tree(data)
    codes = make_codes(tree)
    writer = BitWriter()
    for b in data:
        code = codes[b]
        writer.write_bits_from_str(code)
    writer.flush()
    payload = writer.get_bytes()
    bits = (len(payload)-0) * 8
    # But we may have padded at end, we should compute actual bit-length by summing code lengths
    total_bits = sum(len(codes[b]) for b in data)
    return serialize_huffman_tree(tree), payload, total_bits

def huffman_decompress_bytes(tree_bytes: bytes, payload: bytes, payload_bits: int) -> bytes:
    root = deserialize_huffman_tree(tree_bytes)
    reader = BitReader(payload, payload_bits)
    out = bytearray()
    node = root
    while True:
        bit = reader.read_bit()
        if bit is None:
            break
        node = node.left if bit == 0 else node.right
        if node.is_leaf():
            out.append(node.byte)
            node = root
    return bytes(out)

# ---------------------------
# LZW implementation (16-bit codes)
# ---------------------------
def lzw_compress_bytes(data: bytes) -> bytes:
    # dictionary: bytes -> code (int)
    # initialize dictionary with all 1-byte sequences
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(256)}
    w = b""
    codes = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        codes.append(dictionary[w])
    # write codes as big-endian 2-byte words (16-bit)
    out = bytearray()
    out.extend(struct.pack(">Q", len(codes)))  # count of codes
    for code in codes:
        out.extend(struct.pack(">H", code & 0xFFFF))
    return bytes(out)

def lzw_decompress_bytes(buf: bytes) -> bytes:
    stream = memoryview(buf)
    cnt = struct.unpack_from(">Q", stream, 0)[0]
    pos = 8
    codes = []
    for _ in range(cnt):
        code = struct.unpack_from(">H", stream, pos)[0]; pos += 2
        codes.append(code)
    # decompress
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(256)}
    result = bytearray()
    if not codes:
        return bytes(result)
    w = dictionary[codes[0]]
    result.extend(w)
    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[:1]
        else:
            raise ValueError("Bad LZW code")
        result.extend(entry)
        # add w+entry[0]
        dictionary[dict_size] = w + entry[:1]
        dict_size += 1
        w = entry
    return bytes(result)

# ---------------------------
# Simple RLE (escape-based)
# ---------------------------
# Format: literals copied until an ESC byte 0x00 encountered.
# ESC sequences:
#   0x00 0x00 => literal single 0x00
#   0x00 <count> <value> => run of <count> copies of <value> (count >= 4 recommended)
def rle_compress_bytes(data: bytes) -> bytes:
    ESC = 0x00
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        # check run
        j = i + 1
        while j < n and data[j] == data[i] and j - i < 255:
            j += 1
        run_len = j - i
        if run_len >= 4:
            out.append(ESC)
            out.append(run_len & 0xFF)
            out.append(data[i])
            i = j
        else:
            # write literals until next run or end, escaping 0x00
            out.append(data[i]) if data[i] != ESC else (out.extend([ESC, ESC]))
            i += 1
    return bytes(out)

def rle_decompress_bytes(buf: bytes) -> bytes:
    ESC = 0x00
    out = bytearray()
    i = 0
    n = len(buf)
    while i < n:
        b = buf[i]; i += 1
        if b == ESC:
            if i >= n:
                raise ValueError("RLE malformed")
            nb = buf[i]; i += 1
            if nb == ESC:
                out.append(ESC)
            else:
                # run
                if i >= n:
                    raise ValueError("RLE malformed run")
                val = buf[i]; i += 1
                out.extend(bytes([val]) * nb)
        else:
            out.append(b)
    return bytes(out)

# ---------------------------
# File container format
# ---------------------------
def write_container(output_path: str, alg_id: int, orig_name: str, orig_size: int, payload_blocks: List[Tuple[bytes, dict]]):
    """
    payload_blocks: algorithm-specific pieces. We'll write a common format:
    MAGIC
    alg_id (1 byte)
    filename_len (2 bytes) filename (utf-8)
    orig_size (8 bytes)
    Then algorithm-specific:
      For Huffman: tree_len (4), tree_bytes, payload_bit_len (8), payload_bytes_len (8), payload_bytes
      For LZW: payload_bytes_len (8), payload_bytes
      For RLE: payload_bytes_len (8), payload_bytes
    """
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("B", alg_id))
        nameb = orig_name.encode("utf-8")
        f.write(struct.pack(">H", len(nameb)))
        f.write(nameb)
        f.write(struct.pack(">Q", orig_size))
        if alg_id == ALG_HUFFMAN:
            tree_bytes, payload_bytes, payload_bits = payload_blocks[0]
            f.write(struct.pack(">I", len(tree_bytes)))
            f.write(tree_bytes)
            f.write(struct.pack(">Q", payload_bits))
            f.write(struct.pack(">Q", len(payload_bytes)))
            f.write(payload_bytes)
        elif alg_id == ALG_LZW:
            payload_bytes = payload_blocks[0][0]
            f.write(struct.pack(">Q", len(payload_bytes)))
            f.write(payload_bytes)
        elif alg_id == ALG_RLE:
            payload_bytes = payload_blocks[0][0]
            f.write(struct.pack(">Q", len(payload_bytes)))
            f.write(payload_bytes)
        else:
            raise ValueError("Unknown algorithm id")

def read_container(path: str):
    with open(path, "rb") as f:
        data = f.read()
    mv = memoryview(data)
    pos = 0
    if data[:5] != MAGIC:
        raise ValueError("Not a CMPV1 file")
    pos += 5
    alg_id = mv[pos]; pos += 1
    name_len = struct.unpack_from(">H", mv, pos)[0]; pos += 2
    name = bytes(mv[pos:pos+name_len]).decode("utf-8"); pos += name_len
    orig_size = struct.unpack_from(">Q", mv, pos)[0]; pos += 8
    if alg_id == ALG_HUFFMAN:
        tree_len = struct.unpack_from(">I", mv, pos)[0]; pos += 4
        tree_bytes = bytes(mv[pos:pos+tree_len]); pos += tree_len
        payload_bits = struct.unpack_from(">Q", mv, pos)[0]; pos += 8
        payload_len = struct.unpack_from(">Q", mv, pos)[0]; pos += 8
        payload_bytes = bytes(mv[pos:pos+payload_len]); pos += payload_len
        return {
            "alg": alg_id, "name": name, "orig_size": orig_size,
            "tree": tree_bytes, "payload_bytes": payload_bytes, "payload_bits": payload_bits
        }
    elif alg_id == ALG_LZW:
        payload_len = struct.unpack_from(">Q", mv, pos)[0]; pos += 8
        payload_bytes = bytes(mv[pos:pos+payload_len]); pos += payload_len
        return {"alg": alg_id, "name": name, "orig_size": orig_size, "payload_bytes": payload_bytes}
    elif alg_id == ALG_RLE:
        payload_len = struct.unpack_from(">Q", mv, pos)[0]; pos += 8
        payload_bytes = bytes(mv[pos:pos+payload_len]); pos += payload_len
        return {"alg": alg_id, "name": name, "orig_size": orig_size, "payload_bytes": payload_bytes}
    else:
        raise ValueError("Unknown algorithm id in file")

# ---------------------------
# High-level compress / decompress
# ---------------------------
def compress_file(path: str, out_path: Optional[str] = None, alg: str = "huffman"):
    if out_path is None:
        out_path = path + ".cmp"
    alg = alg.lower()
    with open(path, "rb") as f:
        data = f.read()
    if alg == "huffman":
        tree_bytes, payload_bytes, payload_bits = huffman_compress_bytes(data)
        write_container(out_path, ALG_HUFFMAN, os.path.basename(path), len(data),
                        [(tree_bytes, payload_bytes, payload_bits)])
    elif alg == "lzw":
        payload_bytes = lzw_compress_bytes(data)
        write_container(out_path, ALG_LZW, os.path.basename(path), len(data),
                        [(payload_bytes, {})])
    elif alg == "rle":
        payload_bytes = rle_compress_bytes(data)
        write_container(out_path, ALG_RLE, os.path.basename(path), len(data),
                        [(payload_bytes, {})])
    else:
        raise ValueError("Unknown algorithm name")
    return out_path

def decompress_file(path: str, out_dir: Optional[str] = None):
    info = read_container(path)
    alg = info["alg"]
    name = info["name"]
    orig_size = info["orig_size"]
    if out_dir is None:
        out_dir = os.path.dirname(path) or "."
    out_path = os.path.join(out_dir, name)
    if alg == ALG_HUFFMAN:
        tree = info["tree"]
        payload_bytes = info["payload_bytes"]
        payload_bits = info["payload_bits"]
        out = huffman_decompress_bytes(tree, payload_bytes, payload_bits)
    elif alg == ALG_LZW:
        out = lzw_decompress_bytes(info["payload_bytes"])
    elif alg == ALG_RLE:
        out = rle_decompress_bytes(info["payload_bytes"])
    else:
        raise ValueError("Unknown alg in file")
    # validation (optionally compare orig size)
    if len(out) != orig_size:
        # still write file but warn
        print(f"Warning: decompressed size {len(out)} != original {orig_size}")
    with open(out_path, "wb") as f:
        f.write(out)
    return out_path

# ---------------------------
# CLI
# ---------------------------
def cli_main():
    parser = argparse.ArgumentParser(description="Simple compression utility (Huffman, LZW, RLE) with GUI")
    sub = parser.add_subparsers(dest="cmd", required=False)
    parser.add_argument("--gui", action="store_true", help="Run GUI")
    c1 = sub.add_parser("compress", help="Compress file")
    c1.add_argument("input", help="Input file path")
    c1.add_argument("--alg", choices=["huffman","lzw","rle"], default="huffman", help="Algorithm")
    c1.add_argument("--out", help="Output path (optional)")

    c2 = sub.add_parser("decompress", help="Decompress file")
    c2.add_argument("input", help="Compressed file")
    c2.add_argument("--outdir", help="Output directory (optional)")

    args = parser.parse_args()
    if args.gui:
        if not PYQT_AVAILABLE:
            print("PyQt5 not installed. Install with: pip install PyQt5")
            return 2
        run_gui()
        return 0
    if args.cmd == "compress":
        out = compress_file(args.input, args.out, args.alg)
        print("Wrote", out)
    elif args.cmd == "decompress":
        out = decompress_file(args.input, args.outdir)
        print("Wrote", out)
    else:
        parser.print_help()

# ---------------------------
# PyQt5 GUI (glassmorphic/neumorphic dark theme)
# ---------------------------
def run_gui():
    if not PYQT_AVAILABLE:
        raise RuntimeError("PyQt5 not installed")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CompressIt")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CompressIt —AYUSH")
        self.setMinimumSize(700, 420)
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # top card
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QGridLayout(card)
        card_layout.setContentsMargins(20,20,20,20)
        card_layout.setSpacing(12)

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select a file...")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self.browse)

        self.alg_combo = QtWidgets.QComboBox()
        self.alg_combo.addItems(["huffman","lzw","rle"])

        btn_compress = QtWidgets.QPushButton("Compress")
        btn_compress.clicked.connect(self.gui_compress)
        btn_decompress = QtWidgets.QPushButton("Decompress")
        btn_decompress.clicked.connect(self.gui_decompress)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setValue(0)

        card_layout.addWidget(QtWidgets.QLabel("File"), 0, 0)
        card_layout.addWidget(self.path_edit, 0, 1)
        card_layout.addWidget(btn_browse, 0, 2)
        card_layout.addWidget(QtWidgets.QLabel("Algorithm"), 1, 0)
        card_layout.addWidget(self.alg_combo, 1, 1)
        card_layout.addWidget(btn_compress, 2, 1)
        card_layout.addWidget(btn_decompress, 2, 2)
        card_layout.addWidget(self.progress, 3, 0, 1, 3)

        # log area
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        layout.addWidget(card)
        layout.addWidget(self.log)

    def apply_style(self):
        # Dark glassmorphic / neumorphic-ish stylesheet
        self.setStyleSheet("""
            QWidget { background: #0f1220; color: #e6eef6; font-family: Inter, Arial; }
            #card {
                background: rgba(255,255,255,0.03);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.04);
                padding: 10px;
                /* soft inner glow */
                box-shadow: 0px 6px 18px rgba(2,6,23,0.8);
            }
            QLineEdit, QComboBox, QTextEdit {
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.03);
                padding: 8px;
                border-radius: 10px;
            }
            QPushButton {
                padding: 8px 12px;
                border-radius: 10px;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.03), stop:1 rgba(255,255,255,0.01));
                border: 1px solid rgba(255,255,255,0.06);
            }
            QPushButton:hover { border: 1px solid rgba(255,255,255,0.12); }
            QProgressBar {
                background: rgba(255,255,255,0.02);
                border-radius: 8px;
                height: 12px;
            }
            QProgressBar::chunk {
                background: qlineargradient(spread:pad, x1:0,y1:0,x2:1,y2:0, stop:0 rgba(90,200,250,0.7), stop:1 rgba(80,180,230,0.9));
                border-radius: 8px;
            }
        """)
        # set window translucency if supported (not required)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, False)
        # drop shadow effect for card
        effect = QtWidgets.QGraphicsDropShadowEffect(blurRadius=22, xOffset=0, yOffset=6)
        effect.setColor(QtGui.QColor(0,0,0,200))
        for child in self.findChildren(QtWidgets.QFrame):
            child.setGraphicsEffect(effect)

    def browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file")
        if path:
            self.path_edit.setText(path)

    def gui_compress(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.log.append("Select a valid file.")
            return
        alg = self.alg_combo.currentText()
        self.log.append(f"Compressing {path} with {alg} ...")
        QtCore.QCoreApplication.processEvents()
        try:
            out = compress_file(path, None, alg)
            sz_in = os.path.getsize(path)
            sz_out = os.path.getsize(out)
            ratio = (1 - sz_out/sz_in) if sz_in else 0
            self.log.append(f"Done: {out} (in {sz_in}B -> out {sz_out}B) ratio saved {ratio:.2%}")
            self.progress.setValue(100)
        except Exception as e:
            self.log.append("Error: " + str(e))
            self.progress.setValue(0)

    def gui_decompress(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.log.append("Select a valid compressed file.")
            return
        self.log.append(f"Decompressing {path} ...")
        QtCore.QCoreApplication.processEvents()
        try:
            out = decompress_file(path)
            self.log.append(f"Done: extracted to {out}")
            self.progress.setValue(100)
        except Exception as e:
            self.log.append("Error: " + str(e))
            self.progress.setValue(0)

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    cli_main()
