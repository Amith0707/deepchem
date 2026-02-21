from typing import Dict, Any, Tuple, Optional
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

try:
    import torch
    import torch.nn as nn
    import numpy as np
    has_torch = True
except:
    has_torch = False


class DNABERTModel(HuggingFaceModel):

    def __init__(
        self,
        task: str,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        n_tasks: int = 1,
        **kwargs
    ):
        self.task = task
        self.n_tasks = n_tasks

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load backbone ONLY
        backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        hidden_size = backbone.config.hidden_size

        # Attach head manually
        if task == "classification":
            num_labels = 2 if n_tasks == 1 else n_tasks
            self.classifier = nn.Linear(hidden_size, num_labels)

        elif task == "regression":
            self.classifier = nn.Linear(hidden_size, n_tasks)

        else:
            raise ValueError("Invalid task")

        self.backbone = backbone

        super(DNABERTModel, self).__init__(
            model=backbone,   # pass backbone to parent
            task=task,
            tokenizer=tokenizer,
            **kwargs
        )

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):

        sequences_batch, y, w = batch

        if isinstance(sequences_batch, np.ndarray):
            sequences_list = sequences_batch.tolist()
        else:
            sequences_list = list(sequences_batch)

        tokens = self.tokenizer(
            sequences_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        for key in tokens:
            tokens[key] = tokens[key].to(self.device)

        outputs = self.backbone(**tokens)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled_output)

        loss = None

        if y is not None:
            y_tensor = torch.from_numpy(y)

            if self.task == "regression":
                y_tensor = y_tensor.float().to(self.device)
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), y_tensor)

            elif self.task == "classification":
                if self.n_tasks == 1:
                    y_tensor = y_tensor.long().to(self.device)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, y_tensor)
                else:
                    y_tensor = y_tensor.float().to(self.device)
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, y_tensor)

        return {"loss": loss, "logits": logits}, y, w
