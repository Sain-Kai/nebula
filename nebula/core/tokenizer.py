"""
Tokenizer implementation providing granular tokenization inspection and byte teardowns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TokenMetadata:
    """Detailed metadata for an individual token in the sequence."""
    index: int
    token_id: int
    text: str
    byte_values: List[int]
    is_special: bool = False
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "token_id": self.token_id,
            "text": self.text,
            "byte_values": self.byte_values,
            "hex": " ".join(f"0x{b:02X}" for b in self.byte_values),
            "is_special": self.is_special,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class NebulaTokenizer:
    """
    Byte-level tokenizer designed for mechanistic interpretability.
    Provides complete byte-to-token transparent mappings, position tracing,
    and subword boundary inspection.
    """

    BOS_TOKEN_ID = 256
    EOS_TOKEN_ID = 257
    PAD_TOKEN_ID = 258
    MASK_TOKEN_ID = 259
    VOCAB_SIZE = 260

    def __init__(self, vocab_size: int = 260):
        self.vocab_size = max(vocab_size, self.VOCAB_SIZE)
        self.special_tokens = {
            self.BOS_TOKEN_ID: "<|bos|>",
            self.EOS_TOKEN_ID: "<|eos|>",
            self.PAD_TOKEN_ID: "<|pad|>",
            self.MASK_TOKEN_ID: "<|mask|>",
        }
        self.reverse_special = {v: k for k, v in self.special_tokens.items()}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encodes text into a sequence of token IDs."""
        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.BOS_TOKEN_ID)
        
        # UTF-8 byte encoding
        utf8_bytes = text.encode("utf-8")
        token_ids.extend(list(utf8_bytes))
        
        if add_eos:
            token_ids.append(self.EOS_TOKEN_ID)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back into string format, safely rendering special tokens."""
        raw_bytes = bytearray()
        result_parts: List[str] = []

        for tid in token_ids:
            if tid in self.special_tokens:
                if raw_bytes:
                    result_parts.append(raw_bytes.decode("utf-8", errors="replace"))
                    raw_bytes = bytearray()
                result_parts.append(self.special_tokens[tid])
            elif 0 <= tid < 256:
                raw_bytes.append(tid)
            else:
                if raw_bytes:
                    result_parts.append(raw_bytes.decode("utf-8", errors="replace"))
                    raw_bytes = bytearray()
                result_parts.append(f"<|token_{tid}|>")

        if raw_bytes:
            result_parts.append(raw_bytes.decode("utf-8", errors="replace"))

        return "".join(result_parts)

    def analyze(self, text: str, add_bos: bool = True) -> List[TokenMetadata]:
        """Performs teardown analysis of tokenization with offsets and byte signatures."""
        metadata_list: List[TokenMetadata] = []
        current_index = 0

        if add_bos:
            metadata_list.append(
                TokenMetadata(
                    index=current_index,
                    token_id=self.BOS_TOKEN_ID,
                    text="<|bos|>",
                    byte_values=[],
                    is_special=True,
                    start_char=0,
                    end_char=0,
                )
            )
            current_index += 1

        # Walk characters and track byte boundaries
        char_offset = 0
        for char in text:
            char_bytes = list(char.encode("utf-8"))
            char_len = len(char)
            for byte_val in char_bytes:
                # Human-friendly representation
                if 32 <= byte_val <= 126 and byte_val != 32:
                    repr_str = chr(byte_val)
                elif byte_val == 32:
                    repr_str = "␣"
                elif byte_val == 10:
                    repr_str = "↵"
                elif byte_val == 9:
                    repr_str = "⇥"
                else:
                    repr_str = f"\\x{byte_val:02x}"

                metadata_list.append(
                    TokenMetadata(
                        index=current_index,
                        token_id=byte_val,
                        text=repr_str,
                        byte_values=[byte_val],
                        is_special=False,
                        start_char=char_offset,
                        end_char=char_offset + char_len,
                    )
                )
                current_index += 1
            char_offset += char_len

        return metadata_list
