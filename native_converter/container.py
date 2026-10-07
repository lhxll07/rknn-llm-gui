"""Recovered RKLLM v3 outer container, based on local 1.3.1 observations.

The outer directory follows GGUF v3. Strings and the magic use a repeating
format mask; numeric values and tensor payloads do not. Matrix directory
offsets describe the ordinary data region, NOT the appended matrix payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import struct
from typing import BinaryIO


FIELD_MASK = bytes.fromhex(
    "9da9e670cce76f6662774b69b88b45a1a6fb80a77a92d6f3e80044f8da42e34e"
    "71ff7dcd82069a6495838fa465eadead6d1cf661be8502fc496cf554b43b5dc3"
    "32071db1881910750b76373fb54c91c80a72d21ebf99ecae21af2da809d41fb0"
    "8d3d9f2815d7e1c76a6bd9bc1bc41217a54d13358eee258184205c98367fcf7"
    "35147118987fe7b182f6efdd193cb03f122f0f23c5958c1904aed8660b2232e5"
    "6a314bb5bd89b16ba745fdff47e30c9ac9401e968509e318cfa97b6550fa0e2a"
    "2b35ee057ce29d0050d7c3a2cca34e40ceb04d55239c04633789626db38272b63"
    "f9efd3b924e59c48434f8a0ebdf75a67aa7941c2ab533eb72adcc51a0840ddc6"
)
MAGIC = b"GGUF"
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
MAX_ITEMS = 2_000_000
MAX_STRING = 64 * 1024 * 1024


def mask_bytes(data: bytes) -> bytes:
    return bytes(value ^ FIELD_MASK[index % len(FIELD_MASK)]
                 for index, value in enumerate(data))


def align(value: int, alignment: int = 32) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("Alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass
class Entry:
    key: str
    kind: int
    value: object
    element_kind: int | None = None


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # logical order, e.g. output_channels, input_channels
    kind: int
    offset: int


class Reader:
    """Read only the directory; keep tensor payloads on disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.buffer = None
        try:
            if self.path.stat().st_size < 24:
                raise ValueError("Truncated RKLLM header")
            self.buffer = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self.position = 0
            if mask_bytes(self.take(4)) != MAGIC:
                raise ValueError("Unsupported RKLLM magic")
            self.version = self.unpack("I")
            if self.version != 3:
                raise ValueError(f"Unsupported outer container version: {self.version}")
            count = self.count()
            metadata_count = self.count()
            self.entries = []
            keys = set()
            for _ in range(metadata_count):
                key = self.string().decode("utf-8")
                if key in keys:
                    raise ValueError(f"Duplicate metadata key: {key}")
                keys.add(key)
                kind = self.unpack("I")
                element = None
                if kind == 9:
                    element = self.unpack("I")
                    if element == 9:
                        raise ValueError("Nested metadata arrays are unsupported")
                    value = [self.value(element) for _ in range(self.count())]
                else:
                    value = self.value(kind)
                self.entries.append(Entry(key, kind, value, element))
            self.metadata = {entry.key: entry.value for entry in self.entries}
            self.tensors = []
            names = set()
            for _ in range(count):
                name = self.string().decode("utf-8")
                if name in names:
                    raise ValueError(f"Duplicate tensor name: {name}")
                names.add(name)
                dimensions = self.unpack("I")
                if not 1 <= dimensions <= 4:
                    raise ValueError("Unsupported tensor rank")
                shape = tuple(reversed([self.unpack("Q") for _ in range(dimensions)]))
                if any(dimension == 0 for dimension in shape):
                    raise ValueError("Zero-sized tensors are unsupported")
                self.tensors.append(TensorInfo(name, shape, self.unpack("I"), self.unpack("Q")))
            self.alignment = self.metadata.get("general.alignment", 32)
            self.data_offset = align(self.position, self.alignment)
            if self.data_offset > len(self.buffer):
                raise ValueError("Tensor directory extends beyond the file")
        except Exception:
            self.close()
            raise

    def take(self, size: int) -> bytes:
        if size < 0 or self.position + size > len(self.buffer):
            raise ValueError("Truncated RKLLM directory")
        result = self.buffer[self.position:self.position + size]
        self.position += size
        return result

    def unpack(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def count(self) -> int:
        value = self.unpack("Q")
        if value > MAX_ITEMS:
            raise ValueError("Unreasonably large directory or array")
        return value

    def string(self) -> bytes:
        size = self.unpack("Q")
        if size > MAX_STRING:
            raise ValueError("Unreasonably large string")
        return mask_bytes(self.take(size))

    def value(self, kind: int):
        if kind == 8:
            return self.string()
        if kind not in SCALARS:
            raise ValueError(f"Unsupported metadata type: {kind}")
        return self.unpack(SCALARS[kind])

    def close(self):
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def encode_string(value: str | bytes) -> bytes:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(data)) + mask_bytes(data)


def encode_value(kind: int, value) -> bytes:
    if kind == 8:
        return encode_string(value)
    if kind not in SCALARS:
        raise ValueError(f"Unsupported metadata type: {kind}")
    return struct.pack("<" + SCALARS[kind], value)


def write_directory(file: BinaryIO, entries: list[Entry], tensors: list[TensorInfo],
                    alignment: int = 32) -> int:
    file.write(mask_bytes(MAGIC))
    file.write(struct.pack("<IQQ", 3, len(tensors), len(entries)))
    for entry in entries:
        file.write(encode_string(entry.key))
        file.write(struct.pack("<I", entry.kind))
        if entry.kind == 9:
            file.write(struct.pack("<IQ", entry.element_kind, len(entry.value)))
            for value in entry.value:
                file.write(encode_value(entry.element_kind, value))
        else:
            file.write(encode_value(entry.kind, entry.value))
    for tensor in tensors:
        file.write(encode_string(tensor.name))
        file.write(struct.pack("<I", len(tensor.shape)))
        for dimension in reversed(tensor.shape):
            file.write(struct.pack("<Q", dimension))
        file.write(struct.pack("<IQ", tensor.kind, tensor.offset))
    end = align(file.tell(), alignment)
    file.write(b"\0" * (end - file.tell()))
    return end
