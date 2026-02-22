import importlib
import numpy as np
import torch
from typing import Any, Tuple

from transformers import AutoConfig, AutoTokenizer
from deepchem.models.torch_models.hf_models import HuggingFaceModel


class DNABERTModel(HuggingFaceModel):
    def __init__(
        self,
        task: str,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        n_tasks: int = 1,
        **kwargs,
    ):
        self.n_tasks = n_tasks

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )

        if task == "mlm":
            class_key = "AutoModelForMaskedLM"
        else:
            class_key = "AutoModelForSequenceClassification"

        if class_key not in config.auto_map:
            raise ValueError(
                f"{class_key} not found in config.auto_map."
            )

        mapping = config.auto_map[class_key]
        _, class_path = mapping.split("--")
        module_name, class_name = class_path.rsplit(".", 1)

        base_module = config.__class__.__module__.rsplit(".", 1)[0]
        full_module_path = f"{base_module}.{module_name}"

        module = importlib.import_module(full_module_path)
        model_class = getattr(module, class_name)

        if task == "classification":
            config.num_labels = 2 if n_tasks == 1 else n_tasks
        elif task == "regression":
            config.num_labels = n_tasks
            config.problem_type = "regression"

        model = model_class.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )

        super(DNABERTModel, self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        X, y, w = batch

        X_flat = np.array(X).ravel().tolist()

        tokens = self.tokenizer(
            X_flat,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in tokens.items()}

        y_tensor = None

        if y is not None:
            y_tensor = torch.from_numpy(np.asarray(y)).to(self.device)

            if self.task == "classification" and self.n_tasks == 1:
                y_tensor = y_tensor.view(-1).long()
            else:
                y_tensor = y_tensor.float()

            inputs["labels"] = y_tensor

        return inputs, y_tensor, w