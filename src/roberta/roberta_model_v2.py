import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.roberta.binary_model import RobertaBinaryClassifier

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "roberta_label1_v2")
DEFAULT_MAX_TOKEN_LEN = 384
MAX_CHAR_LEN = 4096
TOKENIZE_MAP_BATCH_SIZE = 32
SUPPORTED_TARGET_LABELS = {"label0", "label1"}


class _WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class RobertaModelV2:
    """
    RoBERTa binary classifier wrapper with stronger probability calibration controls:
    - train/validation split + early stopping
    - class-weighted loss
    - label smoothing
    - temperature scaling on held-out validation logits
    """

    def __init__(
        self,
        base_model_id: str = "roberta-base",
        target_label: str = "label1",
        max_token_len: Optional[int] = None,
    ) -> None:
        if target_label not in SUPPORTED_TARGET_LABELS:
            raise ValueError(f"target_label must be one of {sorted(SUPPORTED_TARGET_LABELS)}")
        self.base_model_id = base_model_id
        self.target_label = target_label
        self.max_token_len = int(
            max_token_len or os.environ.get("ROBERTA_MAX_TOKEN_LEN", DEFAULT_MAX_TOKEN_LEN)
        )
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[RobertaBinaryClassifier] = None
        self.temperature: float = 1.0

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
            enc = self.tokenizer(batch["text"], truncation=True, max_length=self.max_token_len)
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

    @staticmethod
    def _compute_class_weights(y_train: npt.NDArray[np.int_]) -> torch.Tensor:
        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        weights = len(y_train) / (2.0 * counts)
        return torch.tensor(weights, dtype=torch.float32)

    @torch.no_grad()
    def _collect_logits(self, texts: npt.NDArray[np.str_], batch_size: int = 32) -> np.ndarray:
        self._check_loaded()
        self.model.eval()
        all_logits = []
        device = next(self.model.parameters()).device
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k,v in enc.items()} # # required to enable training on gpu. Doesnt affect cpu training
            outputs = self.model(**enc)
            all_logits.append(outputs["logits"].cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    def _fit_temperature(
        self,
        val_texts: npt.NDArray[np.str_],
        val_labels: npt.NDArray[np.int_],
    ) -> None:
        logits = self._collect_logits(val_texts)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)

        # Optimize a single scalar temperature > 0 in log-space.
        log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            temperature = torch.exp(log_t)
            loss = torch.nn.functional.cross_entropy(logits_t / temperature, labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(torch.exp(log_t).item())

    def train(
        self,
        texts: npt.NDArray[np.str_],
        labels: Optional[npt.NDArray[np.int_]] = None,
        labels_1: Optional[npt.NDArray[np.int_]] = None,
        labels_0: Optional[npt.NDArray[np.int_]] = None,
        output_dir: str = DEFAULT_MODEL_DIR,
        num_epochs: int = 4,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        val_size: float = 0.05,
        early_stopping_patience: int = 2,
        label_smoothing: float = 0.05,
        use_eval_during_training: bool = False,
        optimizer: str = "adafactor",
    ) -> None:
        y = self._resolve_labels(labels, labels_0, labels_1).astype(int)
        texts = np.asarray(texts, dtype=str)

        if len(texts) != len(y):
            raise ValueError("texts and labels must have same length")
        if len(texts) < 20:
            raise ValueError("Need at least 20 samples to train RobertaModelV2 robustly.")

        x_train, x_val, y_train, y_val = train_test_split(
            texts,
            y,
            test_size=val_size,
            random_state=42,
            stratify=y,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.model = RobertaBinaryClassifier(base_model_id=self.base_model_id)
        use_grad_ckpt = os.environ.get("ROBERTA_GRAD_CHECKPOINTING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if use_grad_ckpt:
            self.model.encoder.gradient_checkpointing_enable()
        train_device = os.environ.get("ROBERTA_TRAIN_DEVICE", "cpu").strip().lower()
        use_cpu = train_device == "cpu"

        ds_train = self._make_dataset(x_train, y_train)
        ds_val = self._make_dataset(x_val, y_val)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        class_weights = self._compute_class_weights(y_train)

        eval_strategy = "epoch" if use_eval_during_training else "no"
        save_strategy = "epoch" if use_eval_during_training else "no"
        load_best = bool(use_eval_during_training)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            optim=optimizer,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_strategy="epoch",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to=[],
            label_smoothing_factor=label_smoothing,
            use_cpu=use_cpu,
        )

        callbacks = []
        if use_eval_during_training:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        trainer = _WeightedLossTrainer(
            model=self.model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            callbacks=callbacks,
            class_weights=class_weights,
        )
        trainer.train()

        # Calibrate probability sharpness on held-out validation split.
        self._fit_temperature(x_val, y_val)

    @torch.no_grad()
    def predict(
        self,
        texts: npt.NDArray[np.str_],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float64]:
        self._check_loaded()
        self.model.eval()

        device = next(self.model.parameters()).device
        all_probs = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt",
            )
            enc = {k:v.to(device) for k,v in enc.items()} # required to enable predicting on gpu. Doesnt affect cpu prediction
            outputs = self.model(**enc)
            logits = outputs["logits"] / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs).astype(np.float64)

    def save_model(self, model_save_name: str) -> bool:
        if self.model is None or self.tokenizer is None:
            return False

        save_dir = self._name_to_dir(model_save_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "task_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_label": self.target_label,
                    "temperature": float(self.temperature),
                    "max_token_len": int(self.max_token_len),
                },
                f,
            )
        return True

    def load_model(self, model_save_name: str, model_dir: str = None) -> None:
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
        self.temperature = 1.0
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            trained_target = cfg.get("target_label")
            if trained_target and trained_target != self.target_label:
                raise ValueError(
                    f"Model in {save_dir} was trained for '{trained_target}', "
                    f"but this instance is configured for '{self.target_label}'."
                )
            if "temperature" in cfg:
                self.temperature = float(cfg["temperature"])
            if "max_token_len" in cfg:
                self.max_token_len = int(cfg["max_token_len"])

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
