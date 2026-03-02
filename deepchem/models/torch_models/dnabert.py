import os
import glob
import logging
import torch
import numpy as np
import importlib
from typing import Any, Tuple
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from huggingface_hub import constants as hf_constants
from deepchem.models.torch_models.hf_models import HuggingFaceModel

logger = logging.getLogger(__name__)

class DNABERT2(HuggingFaceModel):
    """
    DNABERT-2 integration for DeepChem. 
    Supports: mlm, classification, regression, mtr, and feature_extractor.
    """
    def __init__(
        self,
        task: str,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        n_tasks: int = 1,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ):
        self.n_tasks = n_tasks
        self.model_name = model_name
        self.task = task

        # Step 1: Force download to HF cache so we can patch the bert_layers.py file
        # The AutoModel call is a bit of a hack but it's the only way to ensure 
        # the remote code is on disk before we try to fix the ALiBi tensor error.
        try:
            _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            # We don't care if this fails, we just need the files in HF_HOME
            _ = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass 

        # Step 2: Apply the ALiBi patch to fix the 'meta' device crash
        self._patch_remote_code()

        # Step 3: Normal HF Setup
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Force the bypass of Triton/Flash-Attention
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.pad_token_id = tokenizer.pad_token_id
        
        # Build the correct model head
        if task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        elif task == "feature_extractor":
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        else:
            # Classification/Regression/MTR logic
            if task == "classification":
                config.num_labels = 2 if n_tasks == 1 else n_tasks
            else:
                config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True)

        super(DNABERT2, self).__init__(model=model, task=task, tokenizer=tokenizer, **kwargs)

    def _patch_remote_code(self):
        """
        Surgically fix the ALiBi initialization in the cached bert_layers.py.
        The authors didn't pass 'device' to rebuild_alibi_tensor, which 
        kills the model on modern transformers/torch versions.
        """
        modules_cache = os.path.join(hf_constants.HF_HOME, "modules", "transformers_modules")
        pattern = os.path.join(modules_cache, "**", "bert_layers.py")
        files = glob.glob(pattern, recursive=True)

        old_line = "self.rebuild_alibi_tensor(size=config.alibi_starting_size)"
        new_line = "self.rebuild_alibi_tensor(size=config.alibi_starting_size, device='cpu')"

        for p in files:
            with open(p, "r") as f:
                content = f.read()
            if old_line in content:
                with open(p, "w") as f:
                    f.write(content.replace(old_line, new_line))
                logger.info(f"Applied ALiBi device patch to {p}")

    def _prepare_batch(self, batch):
        X, y, w = batch
        # Handle the fact that DeepChem datasets might wrap strings in arrays
        sequences = np.array(X).ravel().tolist()
        
        tokens = self.tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt")
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

# import gc
# import glob
# import logging
# import os
# from typing import Any, Tuple

# import numpy as np

# logger = logging.getLogger(__name__)

# try:
#     import torch
#     has_torch = True
# except ImportError:
#     has_torch = False

# try:
#     from huggingface_hub import constants as hf_constants
#     has_huggingface_hub = True
# except ImportError:
#     has_huggingface_hub = False

# try:
#     import transformers
#     has_transformers = True
# except ImportError:
#     has_transformers = False

# try:
#     from deepchem.models.torch_models.hf_models import HuggingFaceModel
#     has_deepchem = True
# except ImportError:
#     has_deepchem = False # ?????


# def _patch_dnabert2_cache(
#     model_name: str = "zhihan1996/DNABERT-2-117M"
# ) -> None:
#     """Apply minimal targeted patches to DNABERT-2's cached custom files.

#     DNABERT-2 ships with two compatibility issues on modern environments:

#     1. **ALiBi meta-device conflict**: ``BertEncoder.__init__`` calls
#        ``rebuild_alibi_tensor`` with no ``device`` argument, causing a
#        ``RuntimeError`` when HuggingFace's lazy loader initialises tensors
#        on the ``meta`` device (PyTorch >= 2.0 / transformers >= 4.38).
#        The fix passes ``device='cpu'`` — a parameter the authors already
#        defined — so the tensor is built on CPU and moved to the correct
#        device during the first forward pass via the authors' own device
#        catch-up logic in ``BertEncoder.forward``.

#     2. **Flash-Attention / Triton**: controlled externally via
#        ``config.attention_probs_dropout_prob > 0`` so no file-level
#        patch is needed. # main thing --------------------------------------- need to remove this comment

#     Both patches are idempotent — safe to run multiple times.

#     Parameters
#     ----------
#     model_name : str
#         HuggingFace model identifier. Used only for error messages.
#     """
#     if not has_huggingface_hub:
#         raise ImportError(
#             "huggingface_hub is required. "
#             "Install with: pip install huggingface_hub"
#         )

#     modules_cache = os.path.join(
#         hf_constants.HF_HOME, "modules", "transformers_modules"
#     )
#     pattern = os.path.join(modules_cache, "**", "bert_layers.py")
#     matches = glob.glob(pattern, recursive=True)

#     if not matches:
#         raise FileNotFoundError(
#             f"bert_layers.py not found in HuggingFace modules cache at "
#             f"'{modules_cache}'. Ensure the tokenizer has been downloaded "
#             f"first via AutoTokenizer.from_pretrained('{model_name}', "
#             f"trust_remote_code=True)."
#         )

#     _OLD = "self.rebuild_alibi_tensor(size=config.alibi_starting_size)"
#     _NEW = (
#         "self.rebuild_alibi_tensor("
#         "size=config.alibi_starting_size, device='cpu')"
#     )

#     for path in matches:
#         with open(path, "r") as fh:
#             src = fh.read()
#         if _OLD in src:
#             src = src.replace(_OLD, _NEW)
#             with open(path, "w") as fh:
#                 fh.write(src)
#             logger.debug("DNABERT-2 ALiBi patch applied: %s", path)
#         else:
#             logger.debug("DNABERT-2 ALiBi patch already applied: %s", path)


# class DNABERT2(HuggingFaceModel):
#     """DNABERT-2 Model for DNA sequence analysis.

#     DNABERT-2 is a foundation model for DNA sequences based on the
#     MosaicBERT architecture with Byte-Pair Encoding (BPE) tokenization.
#     It replaces the k-mer tokenization used in the original DNABERT with
#     a data-driven BPE vocabulary that generalises across species and
#     sequence types.

#     This wrapper integrates DNABERT-2 into DeepChem and resolves several
#     environment-specific compatibility issues transparently:

#     * **Flash-Attention / Triton**: disabled via the authors' own
#       documented ``attention_probs_dropout_prob`` config flag so no
#       Triton installation is required and the model runs on both CPU
#       and GPU.
#     * **ALiBi meta-device conflict**: patched by supplying the ``device``
#       argument the authors already defined but never passed at init.
#     * **Environment-agnostic caching**: uses
#       ``huggingface_hub.constants.HF_HOME`` rather than hard-coded
#       paths, so the wrapper works on Kaggle, Colab, and local machines.

#     The model supports five tasks:

#     * ``mlm`` — masked language modelling (pre-training)
#     * ``classification`` — binary or multi-class sequence classification
#     * ``regression`` — single-target scalar regression
#     * ``mtr`` — multi-task regression
#     * ``feature_extractor`` — returns CLS-token embeddings

#     Parameters
#     ----------
#     task : str
#         Learning task. One of ``'mlm'``, ``'classification'``,
#         ``'regression'``, ``'mtr'``, or ``'feature_extractor'``.
#     model_name : str, optional
#         HuggingFace model identifier.
#         Defaults to ``'zhihan1996/DNABERT-2-117M'``.
#     n_tasks : int, optional
#         Number of prediction targets. Used for ``'classification'``,
#         ``'regression'``, and ``'mtr'``. Defaults to ``1``.
#     attention_probs_dropout_prob : float, optional
#         Any value ``> 0`` disables the Triton Flash-Attention kernel
#         and uses a pure-PyTorch attention implementation instead.
#         This is the authors' own documented mechanism — see
#         ``configuration_bert.py`` in the model repo. Defaults to
#         ``0.1``.
#     **kwargs
#         Additional keyword arguments forwarded to
#         :class:`~deepchem.models.torch_models.hf_models.HuggingFaceModel`.

#     Raises
#     ------
#     ImportError
#         If ``torch``, ``transformers``, or ``huggingface_hub`` are not
#         installed.
#     ValueError
#         If an unsupported ``task`` string is provided.

#     Examples
#     --------
#     **Binary classification — promoter detection**

#     >>> import deepchem as dc
#     >>> import numpy as np
#     >>> from deepchem.models.torch_models.dnabert import DNABERT2

#     >>> sequences = ["ATGCGTACGTAGCTAGCTAGCTAGCGTA",
#     ...              "GCTAGCTAGCTAGCTAGCTAGCTAGC"]
#     >>> labels = np.array([1, 0])
#     >>> dataset = dc.data.NumpyDataset(X=sequences, y=labels)

#     >>> model = DNABERT2(task='classification', n_tasks=1)
#     >>> loss = model.fit(dataset, nb_epoch=1)

#     **Regression**

#     >>> labels = np.array([0.82, 0.31])
#     >>> dataset = dc.data.NumpyDataset(X=sequences, y=labels)
#     >>> model = DNABERT2(task='regression', n_tasks=1)
#     >>> loss = model.fit(dataset, nb_epoch=1)

#     **Multi-task regression**

#     >>> labels = np.array([[0.82, 0.5], [0.31, 0.7]])
#     >>> dataset = dc.data.NumpyDataset(X=sequences, y=labels)
#     >>> model = DNABERT2(task='mtr', n_tasks=2)
#     >>> loss = model.fit(dataset, nb_epoch=1)

#     **Feature extraction**

#     >>> model = DNABERT2(task='feature_extractor')
#     >>> embeddings = model.predict(dataset)  # shape (N, 768)

#     **Masked language modelling**

#     >>> model = DNABERT2(task='mlm')
#     >>> loss = model.fit(dataset, nb_epoch=1)

#     References
#     ----------
#     .. Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H.
#        (2023). DNABERT-2: Efficient Foundation Model and Benchmark For
#        Multi-Species Genome. arXiv:2306.15006.
#     """

#     def __init__(
#         self,
#         task: str,
#         model_name: str = "zhihan1996/DNABERT-2-117M",
#         n_tasks: int = 1,
#         attention_probs_dropout_prob: float = 0.1,
#         **kwargs,
#     ):
#         if not has_torch:
#             raise ImportError(
#                 "PyTorch is required. Install: pip install torch"
#             )
#         if not has_transformers:
#             raise ImportError(
#                 "transformers is required. "
#                 "Install: pip install 'transformers>=4.29,<5'"
#             )
#         if not has_huggingface_hub:
#             raise ImportError(
#                 "huggingface_hub is required. "
#                 "Install: pip install huggingface_hub"
#             )

#         _SUPPORTED = {
#             "mlm", "classification", "regression", "mtr",
#             "feature_extractor"
#         }
#         if task not in _SUPPORTED:
#             raise ValueError(
#                 f"Unsupported task '{task}'. "
#                 f"Choose one of: {sorted(_SUPPORTED)}"
#             )

#         self.n_tasks = n_tasks
#         self.model_name = model_name

#         from transformers import (
#             AutoModel,
#             AutoModelForMaskedLM,
#             AutoModelForSequenceClassification,
#             AutoTokenizer,
#             BertConfig,
#         )

#         # Pull bert_layers.py into the HF modules cache before patching.
#         # AutoModel is the only call that reliably triggers the download of
#         # bert_layers.py on all platforms (Kaggle, Colab, local).
#         # The model init may fail due to missing pad_token_id in the raw
#         # config — that's expected and safe to swallow since we only need
#         # the file on disk.
#         try:
#             _ = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#             del _
#         except Exception:
#             pass
#         gc.collect()

#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#         )
#         config = BertConfig.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#         )

#         _patch_dnabert2_cache(model_name)

#         config.attention_probs_dropout_prob = attention_probs_dropout_prob
#         config.pad_token_id = tokenizer.pad_token_id
#         config.is_decoder = False

#         if task == "classification":
#             config.num_labels = 2 if n_tasks == 1 else n_tasks
#             config.problem_type = (
#                 "single_label_classification"
#                 if n_tasks == 1
#                 else "multi_label_classification"
#             )
#         elif task in ("regression", "mtr"):
#             config.num_labels = n_tasks
#             config.problem_type = "regression"

#         if task == "mlm":
#             config.tie_word_embeddings=False
#             model = AutoModelForMaskedLM.from_pretrained(
#                 model_name, config=config, trust_remote_code=True
#             )
#         elif task == "feature_extractor":
#             model = AutoModel.from_pretrained(
#                 model_name, config=config, trust_remote_code=True
#             )
#         else:
#             model = AutoModelForSequenceClassification.from_pretrained(
#                 model_name, config=config, trust_remote_code=True
#             )

#         super(DNABERT2, self).__init__(
#             model=model,
#             task=task,
#             tokenizer=tokenizer,
#             **kwargs,
#         )

#     def predict(self, dataset, **kwargs):
#         """Run inference.

#         Parameters
#         ----------
#         dataset : dc.data.Dataset
#             Dataset whose ``X`` contains DNA sequences as plain strings.

#         Returns
#         -------
#         np.ndarray
#             * ``classification`` — ``(N, num_labels)`` logits
#             * ``regression`` / ``mtr`` — ``(N, n_tasks)`` values
#             * ``mlm`` — ``(N, seq_len, vocab_size)`` logits
#             * ``feature_extractor`` — ``(N, 768)`` CLS-token embeddings
#         """
#         if self.task == "feature_extractor":
#             return self._predict_embeddings(dataset)
#         return super(DNABERT2, self).predict(dataset, **kwargs)

#     def fit(self, dataset, nb_epoch: int = 1, **kwargs):
#         """Train the model.

#         Parameters
#         ----------
#         dataset : dc.data.Dataset
#             Dataset whose ``X`` contains DNA sequences as plain strings.
#         nb_epoch : int, optional
#             Number of training epochs. Defaults to ``1``.

#         Returns
#         -------
#         float
#             Mean training loss over the last epoch.
#         """
#         if self.task == "feature_extractor":
#             raise ValueError(
#                 "fit() is not supported for task='feature_extractor'. "
#                 "Use task='mlm' for pre-training or 'classification' / "
#                 "'regression' for fine-tuning."
#             )
#         return super(DNABERT2, self).fit(dataset, nb_epoch=nb_epoch, **kwargs)

#     def _predict_embeddings(self, dataset) -> np.ndarray:
#         """Return CLS-token embeddings for every sequence in *dataset*.

#         Parameters
#         ----------
#         dataset : dc.data.Dataset
#             Dataset whose ``X`` contains DNA sequences as plain strings.

#         Returns
#         -------
#         np.ndarray, shape (N, hidden_size)
#             One CLS-token vector per input sequence.
#         """
#         self.model.eval()
#         embeddings = []

#         with torch.no_grad():
#             for seq in dataset.X:
#                 tokens = self.tokenizer(
#                     seq,
#                     return_tensors="pt",
#                     truncation=True,
#                     max_length=512,
#                     padding=True,
#                 )
#                 tokens = {k: v.to(self.device) for k, v in tokens.items()}
#                 outputs = self.model(**tokens)
#                 cls_vec = outputs[0][:, 0, :]
#                 embeddings.append(cls_vec.cpu().numpy())

#         return np.vstack(embeddings)

#     def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
#         """Prepare a batch for the model.

#         Handles:

#         * DNA sequences stored as raw strings in ``X``
#         * Correct label dtype per task — ``long`` for single-label
#           classification, ``float`` for everything else
#         * MLM dynamic masking via the tokenizer's data collator

#         Parameters
#         ----------
#         batch : tuple
#             ``(X, y, w)`` triple from a DeepChem DataLoader.

#         Returns
#         -------
#         tuple
#             ``(inputs_dict, y_tensor, w)`` ready for ``model.forward``.
#         """
#         X, y, w = batch
#         sequences = np.array(X[0]).ravel().tolist()

#         tokens = self.tokenizer(
#             sequences,
#             padding=True,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt",
#         )

#         if self.task == "mlm":
#             input_ids, labels = self.data_collator.torch_mask_tokens(
#                 tokens["input_ids"]
#             )
#             inputs = {
#                 "input_ids": input_ids.to(self.device),
#                 "attention_mask": tokens["attention_mask"].to(self.device),
#                 "labels": labels.to(self.device),
#             }
#             return inputs, None, w

#         inputs = {k: v.to(self.device) for k, v in tokens.items()}

#         if y is not None:
#             # y_tensor = torch.from_numpy(np.asarray(y))
#             y_tensor = torch.from_numpy(np.asarray(y[0]))
#             if self.task == "classification" and self.n_tasks == 1:
#                 y_tensor = y_tensor.view(-1).long().to(self.device)
#             else:
#                 y_tensor = y_tensor.float().to(self.device)
#             inputs["labels"] = y_tensor
#         else:
#             y_tensor = None

#         return inputs, y_tensor, w