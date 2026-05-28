#!/usr/bin/env python3
"""SAM3 frame-labeling worker.

This script is intentionally process-oriented: dg-labeller launches it with a
model Python environment, and the FastAPI app never imports SAM3 or holds model
state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core import frame_queue


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


@dataclass
class OntologyItem:
    class_id: int
    class_name: str
    prompt: str


@dataclass
class Detection:
    class_id: int
    class_name: str
    prompt: str
    score: float
    box_xyxy: list[float]
    mask: np.ndarray | None = None


class Sam3ImageLabeler:
    """Thin upstream-SAM3 image labeler isolated to this worker process."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        sam3_repo: str | Path | None,
        device: str | None,
        dtype: str | None,
    ) -> None:
        if sam3_repo:
            sys.path.insert(0, str(Path(sam3_repo).resolve()))

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "SAM3 is not importable. Check DGL_MODEL_PYTHON and DGL_SAM3_REPO."
            ) from exc

        import torch

        model_kwargs: dict[str, Any] = {}
        if checkpoint_path is not None:
            model_kwargs["checkpoint_path"] = str(checkpoint_path)
            model_kwargs["load_from_HF"] = False
        if device is not None:
            model_kwargs["device"] = device

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = None
        if dtype is not None:
            dtype_map = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
            }
            if dtype not in dtype_map:
                raise ValueError(f"Unsupported dtype: {dtype}")
            if dtype_map[dtype] in {torch.bfloat16, torch.float16}:
                self.autocast_dtype = dtype_map[dtype]

        model = build_sam3_image_model(**model_kwargs)
        self.processor = Sam3Processor(model, device=self.device)
        self._text_cache: dict[str, dict[str, Any]] = {}

    def _inference_context(self) -> Any:
        if self.autocast_dtype is None or self.device == "cpu":
            return nullcontext()

        import torch

        return torch.autocast(self.device.split(":", maxsplit=1)[0], dtype=self.autocast_dtype)

    def cache_text_prompts(self, ontology: list[OntologyItem]) -> None:
        for item in ontology:
            self._text_outputs_for(item.prompt)

    def label_image(
        self,
        image_path: str | Path,
        ontology: list[OntologyItem],
        score_threshold: float,
    ) -> tuple[tuple[int, int], list[Detection]]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        with self._inference_context():
            state = self.processor.set_image(image)

        detections: list[Detection] = []
        for item in ontology:
            output = self._set_cached_text_prompt(state=state, prompt=item.prompt)
            detections.extend(_parse_output(output, item, score_threshold))

        return (width, height), detections

    def _set_cached_text_prompt(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        text_outputs = self._text_outputs_for(prompt)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
        with self._inference_context():
            return self.processor._forward_grounding(state)

    def _text_outputs_for(self, prompt: str) -> dict[str, Any]:
        if prompt not in self._text_cache:
            with self._inference_context():
                self._text_cache[prompt] = self.processor.model.backbone.forward_text(
                    [prompt],
                    device=self.device,
                )
        return self._text_cache[prompt]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label queued frames with SAM3.")
    parser.add_argument("--frames-db", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--ontology-json", required=True)
    parser.add_argument("--producer-done-file", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--sam3-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    job_dir = Path(args.job_dir)
    frames_db = Path(args.frames_db)
    producer_done = Path(args.producer_done_file)
    ontology = _load_ontology(Path(args.ontology_json))

    log(
        "startup "
        f"worker_id={args.worker_id} device={args.device} dtype={args.dtype} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"sam3_repo={'set' if args.sam3_repo else 'unset'} "
        f"checkpoint={'set' if args.checkpoint else 'unset'} "
        f"ontology_items={len(ontology)} save_masks={args.save_masks}"
    )

    labeler = Sam3ImageLabeler(
        checkpoint_path=args.checkpoint,
        sam3_repo=args.sam3_repo,
        device=args.device,
        dtype=args.dtype,
    )
    labeler.cache_text_prompts(ontology)
    log(f"ready worker_id={args.worker_id}")

    claimed = 0
    labeled = 0
    failed = 0
    while True:
        frame = frame_queue.claim_ready_frame(frames_db, args.worker_id)
        if frame is None:
            if producer_done.exists() and not frame_queue.has_open_work(frames_db):
                break
            time.sleep(args.poll_sec)
            continue

        claimed += 1
        frame_started = time.monotonic()
        log(f"claim frame_id={frame['frame_id']} path={frame['path']}")
        try:
            image_key = _safe_image_key(frame["frame_id"])
            image_size, detections = labeler.label_image(
                image_path=frame["path"],
                ontology=ontology,
                score_threshold=args.score_threshold,
            )
            mask_file = None
            if args.save_masks:
                mask_file = _save_masks(job_dir / "sam3" / "masks", image_key, detections)
            annotation_path = _write_annotation_json(
                job_dir / "sam3" / "annotations",
                image_key,
                Path(frame["path"]),
                image_size,
                detections,
                mask_file,
            )
            frame_queue.mark_labeled(
                frames_db,
                frame["frame_id"],
                str(annotation_path),
                mask_file,
            )
            labeled += 1
            elapsed = time.monotonic() - frame_started
            log(
                "labeled "
                f"frame_id={frame['frame_id']} detections={len(detections)} "
                f"annotation={annotation_path} elapsed_sec={elapsed:.2f}"
            )
        except Exception as exc:
            failed += 1
            log(f"failed frame_id={frame['frame_id']} error={type(exc).__name__}: {exc}")
            traceback.print_exc()
            frame_queue.mark_failed(frames_db, frame["frame_id"], f"{type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - started
    log(
        "summary "
        f"worker_id={args.worker_id} claimed={claimed} labeled={labeled} "
        f"failed={failed} elapsed_sec={elapsed:.2f}"
    )


def _load_ontology(path: Path) -> list[OntologyItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        OntologyItem(
            class_id=int(item["class_id"]),
            class_name=str(item["class_name"]),
            prompt=str(item["prompt"]),
        )
        for item in raw
    ]


def _safe_image_key(frame_id: str) -> str:
    return frame_id.replace("/", "_").replace("\\", "_")


def _parse_output(output: dict[str, Any], item: OntologyItem, score_threshold: float) -> list[Detection]:
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))
    masks = _to_numpy(output.get("masks"))
    if boxes is None or scores is None:
        return []

    boxes = np.atleast_2d(boxes)
    scores = np.atleast_1d(scores).astype(float)

    parsed: list[Detection] = []
    for index, score in enumerate(scores):
        if score < score_threshold or index >= len(boxes):
            continue
        mask = None
        if masks is not None and index < len(masks):
            mask = np.asarray(masks[index]).squeeze() > 0
        parsed.append(
            Detection(
                class_id=item.class_id,
                class_name=item.class_name,
                prompt=item.prompt,
                score=float(score),
                box_xyxy=[float(value) for value in boxes[index].tolist()],
                mask=mask,
            )
        )
    return parsed


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _save_masks(mask_dir: Path, image_key: str, detections: list[Detection]) -> str | None:
    masks = [det.mask.astype(np.uint8) for det in detections if det.mask is not None]
    if not masks:
        return None
    mask_path = mask_dir / f"{image_key}.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mask_path.with_suffix(mask_path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, masks=np.stack(masks, axis=0))
    tmp.replace(mask_path)
    return str(mask_path)


def _write_annotation_json(
    annotation_dir: Path,
    image_key: str,
    image_path: Path,
    image_size: tuple[int, int],
    detections: list[Detection],
    mask_file: str | None,
) -> Path:
    record = {
        "image": str(image_path),
        "image_key": image_key,
        "width": image_size[0],
        "height": image_size[1],
        "mask_file": mask_file,
        "annotations": [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "prompt": det.prompt,
                "score": det.score,
                "bbox_xyxy": det.box_xyxy,
                "has_mask": det.mask is not None,
            }
            for det in detections
        ],
    }
    annotation_path = annotation_dir / f"{image_key}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = annotation_path.with_suffix(annotation_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tmp.replace(annotation_path)
    return annotation_path

if __name__ == "__main__":
    main()
