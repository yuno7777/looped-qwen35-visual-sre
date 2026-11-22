"""Dataset and multimodal collator with exact assistant-only loss masking for Qwen3.5 Looped SFT."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_TAG_RE = re.compile(r"(?:<image>\s*\n?)+")


class LoopedSFTDataset(Dataset):
    """Loads Qwen3.5 multimodal SFT JSONL files produced by `cot_dataset_builder/dataset_clean.jsonl`."""

    def __init__(self, jsonl_path: str | Path, min_samples: int = 0) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {self.jsonl_path}")

        records: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" not in obj or not isinstance(obj["messages"], list) or len(obj["messages"]) < 2:
                    raise ValueError(f"Invalid record at line {line_no} in {self.jsonl_path}: missing 'messages'")
                records.append(obj)

        if not records:
            raise ValueError(f"SFT dataset is empty: {self.jsonl_path}")

        self.base_records = records
        if min_samples > len(records):
            repeats = (min_samples + len(records) - 1) // len(records)
            self.records = (records * repeats)[:min_samples]
        else:
            self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_paths(self, image_paths: list[str] | None) -> list[Path]:
        if not image_paths:
            return []
        resolved: list[Path] = []
        search_roots = [self.jsonl_path.parent]
        if self.jsonl_path.parent.name == "splits":
            search_roots.append(self.jsonl_path.parent.parent)
        for raw in image_paths:
            p = Path(raw)
            if not p.is_absolute():
                for root in search_roots:
                    if (root / p).exists():
                        p = root / p
                        break
            if not p.exists():
                raise FileNotFoundError(f"Image referenced in {self.jsonl_path} not found: {p}")
            resolved.append(p)
        return resolved

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        return {
            "id": rec.get("id", f"row_{idx}"),
            "messages": rec["messages"],
            "image_paths": self._resolve_image_paths(rec.get("images")),
        }


class MultimodalSFTCollator:
    """Formats Qwen3.5 conversations (text-only or image+text) with assistant-only loss masking."""

    def __init__(self, processor: Any, max_length: int = 2048) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self._assistant_header_ids: list[int] = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    @staticmethod
    def _build_chat_messages(
        raw_messages: list[dict[str, Any]],
        pil_images: list[Image.Image],
    ) -> list[dict[str, Any]]:
        """Convert JSONL messages + PIL images into Qwen3.5 AutoProcessor chat format."""
        formatted: list[dict[str, Any]] = []
        img_cursor = 0

        for msg in raw_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                text_str = content if isinstance(content, str) else str(content)
                has_tag = "<image>" in text_str
                if pil_images and img_cursor < len(pil_images) and (has_tag or img_cursor == 0):
                    # Attach 1 image per `<image>` user turn (supports multi-turn crop_zoom)
                    if has_tag and len(pil_images) > 1:
                        imgs_for_turn = [pil_images[img_cursor]]
                        img_cursor += 1
                    else:
                        imgs_for_turn = pil_images[img_cursor:]
                        img_cursor = len(pil_images)
                    clean_text = IMAGE_TAG_RE.sub("", text_str).strip()
                    blocks: list[dict[str, Any]] = [{"type": "image", "image": img} for img in imgs_for_turn]
                    if clean_text:
                        blocks.append({"type": "text", "text": clean_text})
                    formatted.append({"role": "user", "content": blocks})
                elif isinstance(content, str):
                    formatted.append({"role": "user", "content": [{"type": "text", "text": content}]})
                else:
                    formatted.append({"role": role, "content": content})
            else:
                formatted.append({"role": role, "content": content})

        return formatted

    def _find_assistant_start(self, full_ids: list[int], prefix_len: int) -> int:
        """Locate the exact token index right after `<|im_start|>assistant\\n`."""
        hdr = self._assistant_header_ids
        hdr_len = len(hdr)
        if full_ids[prefix_len : prefix_len + hdr_len] == hdr:
            return prefix_len + hdr_len

        # Fallback: search for the last occurrence of `<|im_start|>assistant\n`
        for pos in range(len(full_ids) - hdr_len, -1, -1):
            if full_ids[pos : pos + hdr_len] == hdr:
                return pos + hdr_len

        return prefix_len

    def _encode_single(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        pil_images = [Image.open(p).convert("RGB") for p in item["image_paths"]]
        try:
            messages = self._build_chat_messages(item["messages"], pil_images)

            # Find last assistant turn so system + user turns are masked with -100
            last_assistant_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
            prefix_messages = messages[:last_assistant_idx]
            full_messages = messages[: last_assistant_idx + 1]

            prefix_inputs = self.processor.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            full_inputs = self.processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            for img in pil_images:
                img.close()

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prefix_len = int(prefix_inputs["input_ids"].shape[1])

        assistant_start = self._find_assistant_start(input_ids.tolist(), prefix_len)

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = input_ids.clone()
        labels[: min(assistant_start, labels.shape[0])] = -100

        encoded: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "mm_token_type_ids" in full_inputs and full_inputs["mm_token_type_ids"] is not None:
            mm_ids = full_inputs["mm_token_type_ids"][0]
            if mm_ids.shape[0] > self.max_length:
                mm_ids = mm_ids[: self.max_length]
            encoded["mm_token_type_ids"] = mm_ids

        if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
            encoded["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs and full_inputs["image_grid_thw"] is not None:
            encoded["image_grid_thw"] = full_inputs["image_grid_thw"]

        return encoded

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_items = [self._encode_single(item) for item in batch]
        max_seq = max(x["input_ids"].shape[0] for x in encoded_items)
        bs = len(encoded_items)

        input_ids = torch.full((bs, max_seq), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_seq), dtype=torch.long)
        labels = torch.full((bs, max_seq), -100, dtype=torch.long)

        has_mm = any("mm_token_type_ids" in x for x in encoded_items)
        mm_token_type_ids = torch.zeros((bs, max_seq), dtype=torch.int32) if has_mm else None

        pixel_values_list: list[torch.Tensor] = []
        image_grid_list: list[torch.Tensor] = []

        for i, enc in enumerate(encoded_items):
            seq_len = enc["input_ids"].shape[0]
            input_ids[i, :seq_len] = enc["input_ids"]
            attention_mask[i, :seq_len] = enc["attention_mask"]
            labels[i, :seq_len] = enc["labels"]
            if mm_token_type_ids is not None and "mm_token_type_ids" in enc:
                mm_token_type_ids[i, :seq_len] = enc["mm_token_type_ids"]
            if "pixel_values" in enc:
                pixel_values_list.append(enc["pixel_values"])
            if "image_grid_thw" in enc:
                image_grid_list.append(enc["image_grid_thw"])

        collated: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if mm_token_type_ids is not None:
            collated["mm_token_type_ids"] = mm_token_type_ids
        if pixel_values_list:
            collated["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_list:
            collated["image_grid_thw"] = torch.cat(image_grid_list, dim=0)

        return collated
