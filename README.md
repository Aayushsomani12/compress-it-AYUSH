# 📦 Compress Tool

A Python-based file compression utility that supports **Huffman Coding, LZW, and RLE** algorithms.
The tool works in both **Command Line (CLI)** and a modern **PyQt5 Graphical User Interface (GUI)** with dark glassmorphic styling.

---

## ✨ Features

* 🔹 Compress and decompress any file (text or binary).
* 🔹 Algorithms implemented:

  * **Huffman Coding** (bit-packed)
  * **LZW** (16-bit codes)
  * **Run-Length Encoding (RLE)** (escape-based)
* 🔹 Dual Interface:

  * **CLI** for developers and scripting.
  * **PyQt5 GUI** for ease of use.
* 🔹 Container format stores algorithm info, filename, and original size.
* 🔹 Progress bar and logs in GUI.

---

## 🚀 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/compress-tool.git
   cd compress-tool
   ```

2. Install requirements:

   ```bash
   pip install pyqt5
   ```

*(If you only want CLI, PyQt5 is optional.)*

---

## ⚡ Usage

### CLI Mode

#### Compress a file

```bash
python compress_tool.py compress input.txt --alg huffman
```

➡️ Output: `input.txt.cmp`

#### Decompress a file

```bash
python compress_tool.py decompress input.txt.cmp
```

#### Supported algorithms

* `huffman`
* `lzw`
* `rle`

---

### GUI Mode

Run:

```bash
python compress_tool.py --gui
```

Features in GUI:

* Browse and select file.
* Choose algorithm from dropdown.
* **Compress** or **Decompress** with one click.
* Progress bar + compression ratio shown.

---

## 📂 Project Structure

```
compress_tool.py   # Main script (CLI + GUI + algorithms)
README.md          # Documentation
```

---

## 📊 Example

Compression (Huffman):

```bash
python compress_tool.py compress sample.txt --alg huffman
Wrote sample.txt.cmp
```

Decompression:

```bash
python compress_tool.py decompress sample.txt.cmp
Wrote sample.txt
```

GUI:

* Select `sample.txt` → Huffman → Compress
* Output: `sample.txt.cmp`

---

## 🛠 Future Improvements

* Add advanced algorithms (e.g., DEFLATE, BWT).
* Multi-threading for large files.
* Batch compression/decompression.
* Compression ratio graphs in GUI.

---


---

👉 Do you want me to also add a **“Screenshots” section** with placeholders where you can later insert GUI images or GIF from your screen recording?
