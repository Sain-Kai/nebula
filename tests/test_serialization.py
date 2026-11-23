import io
import json
import struct
import pytest
from nebula.core.serialization import GGUFParser, SafeTensorsParser


def test_safetensors_parser():
    header_dict = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [320, 64],
            "data_offsets": [0, 40960],
        },
    }
    header_json = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_json)

    bio = io.BytesIO()
    bio.write(struct.pack("<Q", header_len))
    bio.write(header_json)
    bio.seek(0)

    summary = SafeTensorsParser.parse_header(bio)
    assert summary.format_type == "SafeTensors"
    assert summary.tensor_count == 1
    assert summary.tensors[0].name == "model.embed_tokens.weight"
    assert summary.tensors[0].shape == [320, 64]


def test_gguf_parser():
    bio = io.BytesIO()
    bio.write(struct.pack("<I", GGUFParser.GGUF_MAGIC))
    bio.write(struct.pack("<I", 3))  # version
    bio.write(struct.pack("<Q", 1))  # tensor_count
    bio.write(struct.pack("<Q", 1))  # metadata_count

    # Write key: "general.architecture" -> "llama"
    key = "general.architecture".encode("utf-8")
    bio.write(struct.pack("<Q", len(key)))
    bio.write(key)
    bio.write(struct.pack("<I", 7))  # string type
    val = "llama".encode("utf-8")
    bio.write(struct.pack("<Q", len(val)))
    bio.write(val)

    # Write tensor: "token_embd.weight", 2 dims [64, 320], type F32 (0), offset 0
    t_name = "token_embd.weight".encode("utf-8")
    bio.write(struct.pack("<Q", len(t_name)))
    bio.write(t_name)
    bio.write(struct.pack("<I", 2))  # n_dims
    bio.write(struct.pack("<Q", 64))
    bio.write(struct.pack("<Q", 320))
    bio.write(struct.pack("<I", 0))  # type F32
    bio.write(struct.pack("<Q", 0))  # offset

    bio.seek(0)
    summary = GGUFParser.parse_header(bio)

    assert summary.format_type == "GGUF"
    assert summary.version == 3
    assert summary.architecture == "llama"
    assert len(summary.tensors) == 1
    assert summary.tensors[0].dtype == "F32"
