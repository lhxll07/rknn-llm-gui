"""Read the observed RKLLM 1.3.1 NPU program without SDK/FlatBuffers modules.

Field slots/types were recovered by observing public generated accessors.
Only the observed RLLM v1 / RKRK model envelope is accepted. This is an
inspector, not an instruction compiler or an execution API.
"""

from collections import Counter
import hashlib
import math
from pathlib import Path
import struct
import zipfile

from .container import Reader, align

MAX_PROGRAM = 256 * 1024 * 1024
MAX_VECTOR = 1_000_000
CATEGORIES = {0: "UNDEFINED", 1: "INPUT", 2: "OUTPUT", 3: "INTERNAL",
              4: "WEIGHT", 5: "COEFF", 6: "LUT", 7: "REGCFG", 8: "MISC",
              9: "REGCMD", 10: "REGTASK"}
TASK = struct.Struct("<8IQ")


class Table:
    def __init__(self, data, position):
        self.data, self.position = data, position
        self.check(position, 4)
        self.vtable = position - self.read("i", position)
        self.check(self.vtable, 4)
        self.vsize = self.read("H", self.vtable)
        self.osize = self.read("H", self.vtable + 2)
        if self.vsize < 4 or self.vsize % 2 or self.osize < 4:
            raise ValueError("Invalid FlatBuffers table")
        self.check(self.vtable, self.vsize)
        self.check(position, self.osize)

    def check(self, offset, size):
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("FlatBuffers range extends beyond model section")

    def read(self, fmt, offset):
        self.check(offset, struct.calcsize("<" + fmt))
        return struct.unpack_from("<" + fmt, self.data, offset)[0]

    def field(self, slot):
        index = 4 + slot * 2
        if index >= self.vsize:
            return None
        offset = self.read("H", self.vtable + index)
        if not offset:
            return None
        if offset >= self.osize:
            raise ValueError("Field lies outside its FlatBuffers object")
        return self.position + offset

    def scalar(self, slot, fmt, default=0):
        offset = self.field(slot)
        return default if offset is None else self.read(fmt, offset)

    def indirect(self, offset):
        target = offset + self.read("I", offset)
        self.check(target, 4)
        return target

    def string_at(self, offset):
        target = self.indirect(offset)
        size = self.read("I", target)
        if size > 1024 * 1024:
            raise ValueError("Unreasonably large program string")
        self.check(target + 4, size + 1)
        if self.data[target + 4 + size] != 0:
            raise ValueError("Unterminated program string")
        return bytes(self.data[target + 4:target + 4 + size]).decode("utf-8")

    def string(self, slot):
        offset = self.field(slot)
        return None if offset is None else self.string_at(offset)

    def child(self, slot):
        offset = self.field(slot)
        return None if offset is None else Table(self.data, self.indirect(offset))

    def vector(self, slot, width):
        offset = self.field(slot)
        if offset is None:
            return 0, 0
        target = self.indirect(offset)
        count = self.read("I", target)
        if count > MAX_VECTOR:
            raise ValueError("Unreasonably large program vector")
        self.check(target + 4, count * width)
        return target + 4, count

    def ints(self, slot):
        start, count = self.vector(slot, 4)
        return list(struct.unpack_from("<" + "i" * count, self.data, start)) if count else []

    def bytes_vector(self, slot):
        start, count = self.vector(slot, 1)
        return bytes(self.data[start:start + count])

    def strings(self, slot):
        start, count = self.vector(slot, 4)
        return [self.string_at(start + i * 4) for i in range(count)]

    def children(self, slot):
        start, count = self.vector(slot, 4)
        return [Table(self.data, self.indirect(start + i * 4)) for i in range(count)]


def _matmul_tensor(t):
    if t is None:
        return None
    return {"layout": t.scalar(0, "b"), "quant_type": t.scalar(1, "b"),
            "group_size": t.scalar(2, "i"), "quant_params_hex": t.bytes_vector(3).hex()}


def _operator(t):
    p = t.child(3)
    return {"type": t.string(0), "name": t.string(1), "domain_id": t.scalar(2, "i"),
            "inputs": t.ints(4), "outputs": t.ints(5),
            "matmul": None if p is None else {
                "type": p.scalar(0, "b"), "shapes": [s.ints(0) for s in p.children(1)],
                "input_a": _matmul_tensor(p.child(2)), "input_b": _matmul_tensor(p.child(3)),
                "output_c": _matmul_tensor(p.child(4))},
            "jobs": [{"task_start": j.ints(0), "task_number": j.ints(1),
                      "lut_number": j.ints(2)} for j in t.children(6)]}


def _tensor(t):
    category = t.scalar(2, "b")
    return {"name": t.string(5), "type": t.scalar(0, "b"), "layout": t.scalar(1, "b"),
            "category": category, "category_name": CATEGORIES.get(category, "UNKNOWN"),
            "shape": t.ints(3), "original_shape": t.ints(4),
            "plan_size": t.scalar(6, "I"), "plan_offset": t.scalar(7, "I")}


def read_program(path: Path) -> bytes:
    if path.suffix == ".rkplan":
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("program.bin")
            if info.file_size > MAX_PROGRAM:
                raise ValueError("NPU program too large")
            return archive.read("program.bin")
    if path.suffix == ".rkllm":
        with Reader(path) as reader:
            ordinary_end = 0
            for tensor in reader.tensors:
                if tensor.name == "token_embd.weight" or tensor.kind == 0:
                    ordinary_end = max(ordinary_end, align(tensor.offset + math.prod(tensor.shape)
                                                          * (2 if tensor.kind == 1 else 4)))
            start = reader.data_offset + ordinary_end
            if start + 256 > len(reader.buffer):
                raise ValueError("Missing program header")
            h = struct.unpack_from("<5Q", reader.buffer, start)
            size = 256 + sum(h[2:])
            if size > MAX_PROGRAM or start + size > len(reader.buffer):
                raise ValueError("Invalid program section sizes")
            return reader.buffer[start:start + size]
    if path.stat().st_size > MAX_PROGRAM:
        raise ValueError("NPU program too large")
    return path.read_bytes()


def inspect_program(program: bytes, include_commands=False) -> dict:
    if len(program) < 264:
        raise ValueError("Truncated NPU program")
    magic, version, model_size, task_size, command_size = struct.unpack_from("<5Q", program)
    if (magic != 0x4D4C4C52 or version != 1 or len(program) != 256 + model_size + task_size + command_size
            or model_size < 8 or task_size % TASK.size or command_size % 8):
        raise ValueError("Unsupported NPU envelope")
    data = program[256:256 + model_size]
    if data[4:8] != b"RKRK":
        raise ValueError("Unsupported program model identifier")
    m = Table(data, struct.unpack_from("<I", data)[0])
    operators = [_operator(t) for t in m.children(3)]
    tensors = [_tensor(t) for t in m.children(4)]
    address_regs = []
    for t in m.children(5):
        start, count = t.vector(1, 16)
        values = [dict(zip(("address_offset", "regcmd_word", "core_mask", "core_id"),
                           struct.unpack_from("<4i", data, start + i * 16))) for i in range(count)]
        address_regs.append({"name": t.string(0), "entries": values})
    domains = [{"domain_id": d.scalar(0, "i"), "operators": d.strings(1),
                "internal_size": d.scalar(2, "i"), "constant_size": d.scalar(3, "i"),
                "regcmd_size": d.scalar(4, "i"), "regtask_size": d.scalar(5, "i"),
                "total_dma_size": d.scalar(6, "i")} for d in m.children(6)]
    task_start = 256 + model_size
    command_start = task_start + task_size
    tasks = []
    words = list(struct.unpack_from("<" + "Q" * (command_size // 8), program, command_start))
    for i in range(task_size // TASK.size):
        values = TASK.unpack_from(program, task_start + i * TASK.size)
        task = dict(zip(("flags", "operator_index", "enable_mask", "interrupt_mask", "interrupt_clear",
                         "interrupt_status", "regcfg_amount", "regcfg_offset", "regcmd_offset"), values))
        end = task["regcmd_offset"] + task["regcfg_amount"] * 8
        if task["regcmd_offset"] % 8 or end > command_size:
            raise ValueError("Task register commands lie outside command section")
        tasks.append(task)
    decoded = [{"word": i, "target": (w >> 48) & 0xffff,
                "register": w & 0xffff, "value": (w >> 16) & 0xffffffff} for i, w in enumerate(words)]
    for table in address_regs:
        for entry in table["entries"]:
            i = entry["regcmd_word"]
            if not 0 <= i < len(words):
                raise ValueError("Address relocation outside command section")
            entry.update({"target": decoded[i]["target"], "register": decoded[i]["register"]})
    result = {"program_bytes": len(program), "program_sha256": hashlib.sha256(program).hexdigest(),
              "envelope_version": version, "section_sizes": {"model": model_size, "tasks": task_size,
                                                               "regcmd": command_size},
              "model_version": m.scalar(0, "I"), "target": m.string(1), "core_mode": m.scalar(2, "B"),
              "operators": operators, "tensors": tensors, "address_relocations": address_regs,
              "domains": domains, "tasks": tasks,
              "regcmd_words": len(words), "register_targets": dict(Counter(f"0x{d['target']:04x}" for d in decoded))}
    if include_commands:
        result["commands"] = decoded
    return result
