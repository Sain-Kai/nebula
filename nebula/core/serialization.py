"""
Binary serialization parser for GGUF and SafeTensors model formats.
Extracts model architecture topology, tensor dimension descriptors, and quantization types
directly from binary file headers with zero heavy framework dependencies.
"""

from __future__ import annotations
import json
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, BinaryIO


@dataclass
class TensorDescriptor:
    """Descriptor of a tensor inside a serialized model file."""
    name: str
    shape: List[int]
    dtype: str
    offset: int
    size_bytes: int


@dataclass
class SerializedModelSummary:
    """Parsed metadata and tensor topology from a model file."""
    format_type: str  # 'GGUF', 'SafeTensors', 'Unknown'
    version: int
    tensor_count: int
    architecture: str
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    num_experts: Optional[int] = None
    tensors: List[TensorDescriptor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_type": self.format_type,
            "version": self.version,
            "tensor_count": self.tensor_count,
            "architecture": self.architecture,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "num_experts": self.num_experts,
            "metadata_keys": list(self.metadata.keys()),
            "tensor_sample": [
                {"name": t.name, "shape": t.shape, "dtype": t.dtype, "offset": t.offset}
                for t in self.tensors[:10]
            ],
        }


class GGUFParser:
    """Parser for binary GGUF (GGML Universal Format) file headers."""

    GGUF_MAGIC = 0x46554747  # b"GGUF" in little-endian

    GGML_TYPE_NAMES = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        12: "Q4_K",
        14: "Q6_K",
    }

    @classmethod
    def parse_header(cls, file_obj: BinaryIO) -> SerializedModelSummary:
        """Parses magic, metadata KV, and tensor descriptors from open binary file."""
        file_obj.seek(0)
        magic_bytes = file_obj.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("File too short for GGUF header")

        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != cls.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

        version = struct.unpack("<I", file_obj.read(4))[0]
        tensor_count = struct.unpack("<Q", file_obj.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", file_obj.read(8))[0]

        metadata: Dict[str, Any] = {}

        def read_string() -> str:
            str_len = struct.unpack("<Q", file_obj.read(8))[0]
            return file_obj.read(str_len).decode("utf-8", errors="replace")

        # Parse metadata keys and values
        for _ in range(min(metadata_kv_count, 100)):
            key = read_string()
            val_type = struct.unpack("<I", file_obj.read(4))[0]

            if val_type == 0:  # UINT8
                metadata[key] = struct.unpack("<B", file_obj.read(1))[0]
            elif val_type in (1, 2):  # INT8, UINT16
                metadata[key] = struct.unpack("<H", file_obj.read(2))[0]
            elif val_type in (3, 4):  # INT32, UINT32
                metadata[key] = struct.unpack("<I", file_obj.read(4))[0]
            elif val_type == 5:  # FLOAT32
                metadata[key] = struct.unpack("<f", file_obj.read(4))[0]
            elif val_type == 6:  # BOOL
                metadata[key] = bool(struct.unpack("<B", file_obj.read(1))[0])
            elif val_type == 7:  # STRING
                metadata[key] = read_string()
            elif val_type in (8, 9, 10):  # INT64, UINT64
                metadata[key] = struct.unpack("<Q", file_obj.read(8))[0]
            else:
                break

        # Read tensor info
        tensors: List[TensorDescriptor] = []
        for _ in range(min(tensor_count, 200)):
            t_name = read_string()
            n_dims = struct.unpack("<I", file_obj.read(4))[0]
            dims = [struct.unpack("<Q", file_obj.read(8))[0] for _ in range(n_dims)]
            t_type = struct.unpack("<I", file_obj.read(4))[0]
            offset = struct.unpack("<Q", file_obj.read(8))[0]

            type_str = cls.GGML_TYPE_NAMES.get(t_type, f"TYPE_{t_type}")
            tensors.append(
                TensorDescriptor(
                    name=t_name,
                    shape=dims,
                    dtype=type_str,
                    offset=offset,
                    size_bytes=0,
                )
            )

        arch = metadata.get("general.architecture", "transformer")
        d_model = metadata.get(f"{arch}.embedding_length")
        n_layers = metadata.get(f"{arch}.block_count")
        n_heads = metadata.get(f"{arch}.attention.head_count")
        n_experts = metadata.get(f"{arch}.expert_count")

        return SerializedModelSummary(
            format_type="GGUF",
            version=version,
            tensor_count=tensor_count,
            architecture=arch,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            num_experts=n_experts,
            tensors=tensors,
            metadata=metadata,
        )


class SafeTensorsParser:
    """Parser for SafeTensors model file headers."""

    @classmethod
    def parse_header(cls, file_obj: BinaryIO) -> SerializedModelSummary:
        file_obj.seek(0)
        len_bytes = file_obj.read(8)
        if len(len_bytes) < 8:
            raise ValueError("File too short for SafeTensors header")

        header_len = struct.unpack("<Q", len_bytes)[0]
        json_bytes = file_obj.read(header_len)
        header_data = json.loads(json_bytes.decode("utf-8"))

        tensors: List[TensorDescriptor] = []
        metadata = header_data.get("__metadata__", {})

        for name, info in header_data.items():
            if name == "__metadata__":
                continue
            shape = info.get("shape", [])
            dtype = info.get("dtype", "F32")
            offsets = info.get("data_offsets", [0, 0])
            size_b = offsets[1] - offsets[0] if len(offsets) == 2 else 0

            tensors.append(
                TensorDescriptor(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    offset=offsets[0] if offsets else 0,
                    size_bytes=size_b,
                )
            )

        return SerializedModelSummary(
            format_type="SafeTensors",
            version=1,
            tensor_count=len(tensors),
            architecture=metadata.get("format", "transformer"),
            tensors=tensors,
            metadata=metadata,
        )
