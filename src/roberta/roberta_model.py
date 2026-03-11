import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from src.roberta.binary_model import RobertaBinaryClassifier

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "roberta_label1")
MAX_TOKEN_LEN = 128
MAX_CHAR_LEN = 4096
TOKENIZE_MAP_BATCH_SIZE = 32
SUPPORTED_TARGET_LABELS = {"label0", "label1"}


class RobertaModel:
    """
    Single-task RoBERTa wrapper. Train one independent model per label.

    Typical usage:
        # BS model
        rob_bs = RobertaModel(target_label="label1")
        rob_bs.train(texts_train, labels_1=y_train)
        rob_bs.save_model("roberta_bs")

        # AI model
        rob_ai = RobertaModel(target_label="label0")
        rob_ai.train(texts_train, labels_0=y_train_ai)
        rob_ai.save_model("roberta_ai")
    """

    def __init__(self, base_model_id: str = "roberta-base", target_label: str = "label1") -> None:
        if target_label not in SUPPORTED_TARGET_LABELS:
            raise ValueError(f"target_label must be one of {sorted(SUPPORTED_TARGET_LABELS)}")
        self.base_model_id = base_model_id
        self.target_label = target_label
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[RobertaBinaryClassifier] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_dataset(
        self,
        texts: npt.NDArray[np.str_],
        labels: npt.NDArray[np.int_],
    ) -> Dataset:
        ds = Dataset.from_dict(
            {
                "text": [str(t)[:MAX_CHAR_LEN] for t in texts],
                "labels": list(labels),
            }
        )

        def tokenize(batch):
            enc = self.tokenizer(batch["text"], truncation=True, max_length=MAX_TOKEN_LEN)
            enc["labels"] = batch["labels"]
            return enc

        return ds.map(
            tokenize,
            batched=True,
            batch_size=TOKENIZE_MAP_BATCH_SIZE,
            remove_columns=["text"],
        )

    def _check_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded. Call train() or load_model() first.")

    def _model_device(self) -> torch.device:
        self._check_loaded()
        return next(self.model.parameters()).device

    def _resolve_labels(
        self,
        labels: Optional[npt.NDArray[np.int_]],
        labels_0: Optional[npt.NDArray[np.int_]],
        labels_1: Optional[npt.NDArray[np.int_]],
    ) -> npt.NDArray[np.int_]:
        if labels is not None:
            return labels
        if self.target_label == "label0" and labels_0 is not None:
            return labels_0
        if self.target_label == "label1" and labels_1 is not None:
            return labels_1
        raise ValueError(
            f"No labels provided for target_label='{self.target_label}'. "
            "Pass labels=... or the matching labels_0/labels_1 argument."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: npt.NDArray[np.str_],
        labels: Optional[npt.NDArray[np.int_]] = None,
        labels_1: Optional[npt.NDArray[np.int_]] = None,
        labels_0: Optional[npt.NDArray[np.int_]] = None,
        output_dir: str = DEFAULT_MODEL_DIR,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Train a single-task RoBERTa model for self.target_label.

        :param texts: Array of input strings.
        :param labels: Preferred generic labels argument for the active task.
        :param labels_1: Backwards-compatible alias when target_label='label1'.
        :param labels_0: Backwards-compatible alias when target_label='label0'.
        :param output_dir: Where to write HuggingFace trainer checkpoints.
        :param num_epochs: Number of training epochs.
        :param batch_size: Per-device batch size.
        :param learning_rate: AdamW learning rate.
        """
        y = self._resolve_labels(labels, labels_0, labels_1)

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.model = RobertaBinaryClassifier(base_model_id=self.base_model_id)
        self.model.encoder.gradient_checkpointing_enable()
        train_device = os.environ.get("ROBERTA_TRAIN_DEVICE", "cpu").strip().lower()
        use_cpu = train_device != "mps"

        ds = self._make_dataset(texts, y)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            optim="adafactor",
            save_strategy="no",
            logging_strategy="epoch",
            dataloader_pin_memory=False,
            report_to=[],
            use_cpu=use_cpu,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=ds,
            data_collator=collator,
        )
        trainer.train()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        texts: npt.NDArray[np.str_],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float64]:
        """
        Return P(target_label = 1) for each text, shape (N,).
        """
        self._check_loaded()
        self.model.eval()
        model_device = self._model_device()

        all_probs = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(model_device) for k, v in enc.items()}
            outputs = self.model(**enc)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs).astype(np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model_save_name: str) -> bool:
        """
        Save model weights and tokenizer.

        save path: <model_save_name>/
        """
        if self.model is None or self.tokenizer is None:
            return False

        save_dir = self._name_to_dir(model_save_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "task_config.json"), "w", encoding="utf-8") as f:
            json.dump({"target_label": self.target_label}, f)
        return True

    def load_model(self, model_save_name: str, model_dir: str = None) -> None:
        """
        Load model weights and tokenizer from a directory.
        """
        save_dir = model_dir if model_dir else self._name_to_dir(model_save_name)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        config_path = os.path.join(save_dir, "task_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            trained_target = cfg.get("target_label")
            if trained_target and trained_target != self.target_label:
                raise ValueError(
                    f"Model in {save_dir} was trained for '{trained_target}', "
                    f"but this instance is configured for '{self.target_label}'."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        self.model = RobertaBinaryClassifier(base_model_id=self.base_model_id)

        safetensors_path = os.path.join(save_dir, "model.safetensors")
        bin_path = os.path.join(save_dir, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            raise ValueError(
                f"No model weights found in {save_dir} "
                f"(expected model.safetensors or pytorch_model.bin)."
            )

        self.model.load_state_dict(state)
        self.model.eval()

    def _name_to_dir(self, name: str) -> str:
        return name
