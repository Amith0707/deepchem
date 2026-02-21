from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
import torch.nn as nn
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import (AutoModelForMaskedLM, AutoTokenizer, 
                          DataCollatorForLanguageModeling, BertConfig)

class DNABERTModel(HuggingFaceModel):
    """DNABERT Model for DNA Sequenece analysis.
    
    DNABERT is a transformer-based model pretrained on genomic sequences.
    It can be used for both pretraining embeddings and fine tuning for downstream genomic applications
    such as promoter, sequence classification, splice site detection and sequenec rgeression. 
    
    The model supports multiple task types:
    -  `mlm`- Masked Language Modeling for pretraining.
    -  `regression`- Single or multi-task regression.
    -  `classification`- Single or multi-label classification.
    
    Parameters
    -----------
    task: str
        The learning task type. Supported tasks:
        - `mlm`- masked language modeling.
        -  `regression`- Regression Tasks (e.g- Binding Affinity Prediction)
        -  `classification`- Classification Tasks(e.g Promoter Detection)

    model_name: str, optional(default "zhihan1996/DNABERT-2-117M")
        Hugging Face model identifier or local path
    
    n_tasks: int, default 1
        Number of prediction targets for a multitask learning model

    config : Dict[Any, Any], optional (default {})
        Additional configuration parameters for the model

    Example
    --------
    ### Need to fill- when testing is done

    Notes
    --------
    - DNABERT-2 uses k-mer tokenization optimized for DNA Sequences.
    - The model expects uppercase DNA Sequences (A,C,G,T).
    - For best results, sequences should be between 50-512 base pairs.

    References
    ----------
    .. [1] Zhou, Z., et al. "DNABERT-2: Efficient Foundation Model for 
       Multi-Species Genome." arXiv preprint arXiv:2306.15006 (2023).
    """

    def __init__(
            self,
            task: str,
            model_name: str = 'zhihan1996/DNABERT-2-117M',
            n_tasks: int = 1,
            config: Dict[Any, Any] = {},
            **kwargs
    ):
        self.n_tasks = n_tasks
        self.model_name = model_name
        self.task = task
        
        model_config = BertConfig.from_pretrained(model_name, **config)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        base_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )

        if task in ['classification', 'regression']:
            hidden_size = model_config.hidden_size
            out_features = 2 if (task == 'classification' and n_tasks == 1) else n_tasks
            base_model.classifier = nn.Linear(hidden_size, out_features)
            
        super(DNABERTModel, self).__init__(
            model=base_model,
            task=task,
            tokenizer=tokenizer,
            **kwargs
        )
        
        if task == 'mlm':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Prepares a batch of DNA sequences for the model.

        Handles different label formats based on task type:
        - Classification (single-task): uses long int for CrossEntropyLoss
        - Classification (multi-task): uses float for BCEWithLogitsLoss
        - Regression: uses float
        - MLM: masks tokens for pretraining
        """
        sequences_batch, y, w = batch
        
        if isinstance(sequences_batch, np.ndarray):
            sequences_flat = sequences_batch.flatten()
            sequences_list = [str(s) for s in sequences_flat]
        elif isinstance(sequences_batch, (list, tuple)):
            sequences_list = [str(s) for s in sequences_batch]
        else:
            sequences_list = [str(sequences_batch)]
        
        tokens = self.tokenizer(
            sequences_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        if self.task == 'mlm':
            inputs, labels = self.data_collator.torch_mask_tokens(tokens['input_ids'])
            tokens_device = {k: v.to(self.device) for k, v in tokens.items()}
            inputs_dict = {
                'input_ids': inputs.to(self.device), 
                'labels': labels.to(self.device),
                'attention_mask': tokens_device['attention_mask']
            }
            return inputs_dict, None, w
        
        tokens_device = {k: v.to(self.device) for k, v in tokens.items()}
        
        y_tensor = None
        if y is not None:
            if isinstance(y, np.ndarray):
                y_flat = y.flatten()
            else:
                y_flat = np.array(y).flatten()
            
            y_tensor = torch.from_numpy(y_flat).to(self.device)
            
            if self.task == 'classification' and self.n_tasks == 1:
                y_tensor = y_tensor.long().squeeze()
            else:
                y_tensor = y_tensor.float()
                if y_tensor.dim() == 1 and self.n_tasks > 1:
                    y_tensor = y_tensor.unsqueeze(1)
        
        tokens_device['labels'] = y_tensor
        return tokens_device, y_tensor, w
    
    def _compute_model_loss(self, inputs, labels):
        outputs = self.model.bert(**{k: v for k, v in inputs.items() if k != 'labels'})
        
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
        
        pooled_output = last_hidden_state[:, 0, :]
        logits = self.model.classifier(pooled_output)
        
        if labels is not None:
            if self.task == 'classification' and self.n_tasks == 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.task == 'classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels)
            return loss, logits
        return None, logits
    
    def _predict(self, generator, transformers, uncertainty, other_output_types):
        results = []
        for batch in generator:
            inputs, labels, weights = self._prepare_batch(batch)
            
            with torch.no_grad():
                _, logits = self._compute_model_loss(inputs, labels)
                
                if self.task == 'classification':
                    if self.n_tasks == 1:
                        probs = torch.softmax(logits, dim=1)
                        predictions = probs[:, 1].cpu().numpy()
                    else:
                        predictions = torch.sigmoid(logits).cpu().numpy()
                else:
                    predictions = logits.squeeze().cpu().numpy()
                
                results.append(predictions)
        
        results = np.concatenate(results)
        return results


# Gemini
# from typing import Dict, Any, Tuple, Optional
# import torch
# import numpy as np
# import torch.nn as nn
# from deepchem.models.torch_models.hf_models import HuggingFaceModel
# from transformers import (AutoModelForMaskedLM, AutoTokenizer, 
#                           DataCollatorForLanguageModeling, BertConfig)

# class DNABERTModel(HuggingFaceModel):
#     def __init__(
#             self,
#             task: str,
#             model_name: str = 'zhihan1996/DNABERT-2-117M',
#             n_tasks: int = 1,
#             config: Dict[Any, Any] = {},
#             **kwargs
#     ):
#         self.n_tasks = n_tasks
#         self.model_name = model_name
        
#         # 1. FIX: Load the NATIVE BertConfig to avoid the ValueError mismatch
#         model_config = BertConfig.from_pretrained(model_name, **config)
        
#         # 2. Load Tokenizer (BPE for DNABERT-2)
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, trust_remote_code=True
#         )

#         # 3. Load the Backbone Model
#         # We load MaskedLM because DNABERT-2 is natively a Masked LM
#         # We will add our own head in the _prepare_batch or via a wrapper
#         base_model = AutoModelForMaskedLM.from_pretrained(
#             model_name, config=model_config, trust_remote_code=True
#         )

#         # Attach a classification/regression head if needed
#         if task in ['classification', 'regression']:
#             hidden_size = model_config.hidden_size
#             out_features = 2 if (task == 'classification' and n_tasks == 1) else n_tasks
#             # Attach it directly to the base_model object
#             base_model.classifier = nn.Linear(hidden_size, out_features)
            
#         super(DNABERTModel, self).__init__(
#             model=base_model,
#             task=task,
#             tokenizer=tokenizer,
#             **kwargs
#         )
        
#         if task == 'mlm':
#             self.data_collator = DataCollatorForLanguageModeling(
#                 tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#             )

#     def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
#         sequences_batch, y, w = batch
        
#         # Standardize input to list of strings
#         if isinstance(sequences_batch, np.ndarray):
#             sequences_list = sequences_batch.ravel().tolist()
#         else:
#             sequences_list = list(sequences_batch)
        
#         tokens = self.tokenizer(
#             sequences_list,
#             padding=True,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt"
#         ).to(self.device)

#         if self.task == 'mlm':
#             inputs, labels = self.data_collator.torch_mask_tokens(tokens['input_ids'])
#             return {'input_ids': inputs.to(self.device), 
#                     'labels': labels.to(self.device),
#                     'attention_mask': tokens['attention_mask']}, None, w
        
#         # --- Classification/Regression Logic ---
#         # 1. Get backbone outputs (using .bert to skip the original MLM head)
#         outputs = self.model.bert(**tokens)
        
#         # 2. Pooling: Use the [CLS] token (first token)
#         pooled_output = outputs.last_hidden_state[:, 0, :]
        
#         # 3. Head: Pass through our custom linear layer
#         logits = self.model.classifier(pooled_output)
        
#         loss = None
#         y_tensor = None
#         if y is not None:
#             y_tensor = torch.from_numpy(np.asarray(y)).to(self.device)
#             if self.task == 'classification' and self.n_tasks == 1:
#                 y_tensor = y_tensor.view(-1).long()
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits, y_tensor)
#             else:
#                 y_tensor = y_tensor.float()
#                 loss_fct = nn.MSELoss() if self.task == 'regression' else nn.BCEWithLogitsLoss()
#                 loss = loss_fct(logits, y_tensor)

#         # Return dict for the model, plus labels and weights
#         return {'logits': logits, 'loss': loss}, y_tensor, w


# GPT
# from typing import Dict,Any,Tuple
# from deepchem.models.torch_models.hf_models import HuggingFaceModel
# from transformers import (AutoConfig,AutoModelForMaskedLM,AutoTokenizer,DataCollatorForLanguageModeling,AutoModelForSequenceClassification)
# from transformers.modeling_utils import PreTrainedModel

# try:
#     import torch
#     import numpy as np
#     has_torch=True
# except:
#     has_torch=False

# class DNABERTModel(HuggingFaceModel):
#     """DNABERT Model for DNA Sequenece analysis.
    
#     DNABERT is a transformer-based model pretrained on genomic sequences.
#     It can be used for both pretraining embeddings and fine tuning for downstream genomic applications
#     such as promoter, sequence classification, splice site detection and sequenec rgeression. 
    
#     The model supports multiple task types:
#     -  `mlm`- Masked Language Modeling for pretraining.
#     -  `regression`- Single or multi-task regression.
#     -  `classification`- Single or multi-label classification.
    
#     Parameters
#     -----------
#     task: str
#         The learning task type. Supported tasks:
#         - `mlm`- masked language modeling.
#         -  `regression`- Regression Tasks (e.g- Binding Affinity Prediction)
#         -  `classification`- Classification Tasks(e.g Promoter Detection)           # Reminder- Test all these

#     model_name: str, optional(default "zhihan1996/DNABERT-2-117M")
#         Hugging Face model identifier or local path
    
#     n_tasks: int, default 1
#         Number of prediction targets for a multitask learning model

#     config : Dict[Any, Any], optional (default {})
#         Additional configuration parameters for the model

#     Example
#     --------
#     ### Need to fill- when testing is done

#     Notes
#     --------
#     - DNABERT-2 uses k-mer tokenization optimized for DNA Sequences.
#     - The model expects uppercase DNA Sequences (A,C,G,T).
#     - For best results, sequences should be between 50-512 base pairs.

#     References
#     ----------
#     .. [1] Zhou, Z., et al. "DNABERT-2: Efficient Foundation Model for 
#        Multi-Species Genome." arXiv preprint arXiv:2306.15006 (2023).
#     """

#     def __init__(
#             self,
#             task:str,
#             model_name:str='zhihan1996/DNABERT-2-117M',
#             n_tasks:int=1,
#             config:Dict[Any,Any]={},
#             **kwargs
#     ):
#         self.n_tasks=n_tasks
#         self.model_name=model_name
#         tokenizer=AutoTokenizer.from_pretrained(
#             model_name,trust_remote_code=True
#         )
        
#         model:PreTrainedModel
#         if task == "mlm":
#             model = AutoModelForMaskedLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True
#             )

#         elif task in ["classification", "regression"]:

#             model_config = AutoConfig.from_pretrained(
#                 model_name,
#                 trust_remote_code=True
#             )

#             if task == "classification":
#                 if n_tasks == 1:
#                     model_config.problem_type = "single_label_classification"
#                     model_config.num_labels = 2
#                 else:
#                     model_config.problem_type = "multi_label_classification"
#                     model_config.num_labels = n_tasks

#             elif task == "regression":
#                 model_config.problem_type = "regression"
#                 model_config.num_labels = n_tasks

#             model = AutoModelForSequenceClassification.from_pretrained(
#                 model_name,
#                 config=model_config,
#                 trust_remote_code=True
#             )

#         else:
#             raise ValueError("Invalid task")

        
#         super(DNABERTModel,self).__init__(
#             model=model,
#             task=task,
#             tokenizer=tokenizer,
#             **kwargs
#         )
        
#         if task=='mlm':
#             self.data_collator=DataCollatorForLanguageModeling(
#                 tokenizer=tokenizer,
#                 mlm=True,
#                 mlm_probability=0.15
#             )
    
#     def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
#         sequences_batch, y, w = batch

#         # Convert numpy array to list
#         if isinstance(sequences_batch, np.ndarray):
#             sequences_list = sequences_batch.tolist()
#         else:
#             sequences_list = list(sequences_batch)

#         tokens = self.tokenizer(
#             sequences_list,
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )

#         # Move tokens to device
#         for key in tokens:
#             tokens[key] = tokens[key].to(self.device)

#         if self.task == "mlm":
#             inputs, labels = self.data_collator.torch_mask_tokens(
#                 tokens["input_ids"]
#             )

#             inputs = {
#                 "input_ids": inputs.to(self.device),
#                 "labels": labels.to(self.device),
#                 "attention_mask": tokens["attention_mask"],
#             }

#             return inputs, None, w

#         elif self.task in ["classification", "regression"]:

#             if y is not None:
#                 y_tensor = torch.from_numpy(y)

#                 if self.task == "regression":
#                     y_tensor = y_tensor.float()
#                 elif self.task == "classification":
#                     if self.n_tasks == 1:
#                         y_tensor = y_tensor.long()
#                     else:
#                         y_tensor = y_tensor.float()

#                 y_tensor = y_tensor.to(self.device)
#                 tokens["labels"] = y_tensor

#             return tokens, y, w


# from typing import Dict,Any,Tuple
# from deepchem.models.torch_models.hf_models import HuggingFaceModel
# from transformers import (AutoConfig,AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer)
# from transformers import DataCollatorForLanguageModeling # added this
# from transformers.modeling_utils import PreTrainedModel

# try:
#     import torch
#     import numpy as np # added    
#     has_torch=True
# except:
#     has_torch=False

# class DNABERTModel(HuggingFaceModel):
#     """DNABERT Model for DNA Sequenece analysis.
    
#     DNABERT is a transformer-based model pretrained on genomic sequences.
#     It can be used for both pretraining embeddings and fine tuning for downstream genomic applications
#     such as promoter, sequence classification, splice site detection and sequenec rgeression. 
    
#     The model supports multiple task types:
#     -  `mlm`- Masked Language Modeling for pretraining.
#     -  `regression`- Single or multi-task regression.
#     -  `classification`- Single or multi-label classification.
    
#     Parameters
#     -----------
#     task: str
#         The learning task type. Supported tasks:
#         - `mlm`- masked language modeling.
#         -  `regression`- Regression Tasks (e.g- Binding Affinity Prediction)
#         -  `classification`- Classification Tasks(e.g Promoter Detection)           # Reminder- Test all these

#     model_name: str, optional(default "zhihan1996/DNABERT-2-117M")
#         Hugging Face model identifier or local path
    
#     n_tasks: int, default 1
#         Number of prediction targets for a multitask learning model

#     config : Dict[Any, Any], optional (default {})
#         Additional configuration parameters for the model

#     Example
#     --------
#     ### Need to fill- when testing is done

#     Notes
#     --------
#     - DNABERT-2 uses k-mer tokenization optimized for DNA Sequences.
#     - The model expects uppercase DNA Sequences (A,C,G,T).
#     - For best results, sequences should be between 50-512 base pairs.

#     References
#     ----------
#     .. [1] Zhou, Z., et al. "DNABERT-2: Efficient Foundation Model for 
#        Multi-Species Genome." arXiv preprint arXiv:2306.15006 (2023).
#     """

#     def __init__(
#             self,
#             task:str,
#             model_name:str='zhihan1996/DNABERT-2-117M',
#             n_tasks:int=1,
#             config:Dict[Any,Any]={},
#             **kwargs
#     ):
#         self.n_tasks=n_tasks
#         self.model_name=model_name
#         tokenizer=AutoTokenizer.from_pretrained(
#             model_name,trust_remote_code=True
#         )
#         model_config=AutoConfig.from_pretrained(
#             model_name,**config,trust_remote_code=True
#         )
#         model:PreTrainedModel
#         if task=='mlm':
#             model=AutoModelForMaskedLM.from_pretrained(
#                 model_name,
#                 config=model_config,
#                 trust_remote_code=True
#             )
#         elif task=='regression':
#             model_config.problem_type='regression'
#             model_config.num_labels=n_tasks
#             model=AutoModelForSequenceClassification.from_pretrained(
#                 model_name,
#                 config=model_config,
#                 trust_remote_code=True
#             )
#         elif task=='classification':
#             if n_tasks==1:
#                 model_config.problem_type='single_label_classification'
#                 model_config.num_labels=2
#             else:
#                 model_config.problem_type='multi_label_classification'
#                 model_config.num_labels=n_tasks

#             model=AutoModelForSequenceClassification.from_pretrained(
#                 model_name,
#                 config=model_config,
#                 trust_remote_code=True
#             )
#         else:
#             raise ValueError('invalid task specification')
        
#         super(DNABERTModel,self).__init__(
#             model=model,
#             task=task,
#             tokenizer=tokenizer,
#             **kwargs
#         )
#         # adding data collator here
#         if task=='mlm':
#             self.data_collator=DataCollatorForLanguageModeling(
#                 tokenizer=tokenizer,
#                 mlm=True,
#                 mlm_probability=0.15
#         )
    
#     def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
#         """Prepares a batch of DNA sequences for the model.

#         Handles different label formats based on task type:
#         - Classification (single-task): uses long int for CrossEntropyLoss
#         - Classification (multi-task): uses float for BCEWithLogitsLoss
#         - Regression: uses float
#         - MLM: masks tokens for pretraining
#         """
#         sequences_batch, y, w = batch
        
#         # Add
#         if isinstance(sequences_batch, np.ndarray):
#             sequences_list = sequences_batch.tolist()
#         else:
#             sequences_list = list(sequences_batch)
#         tokens = self.tokenizer(
#             # sequences_batch[0].tolist(),
#             sequences_list, #Added
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )

#         if self.task == 'mlm':
#             inputs, labels = self.data_collator.torch_mask_tokens(
#                 tokens['input_ids']
#             )
#             inputs = {
#                 'input_ids': inputs.to(self.device),
#                 'labels': labels.to(self.device),
#                 'attention_mask': tokens['attention_mask'].to(self.device),
#             }
#             return inputs, None, w
        
#         elif self.task in ['regression', 'classification']:
#             if y is not None:
#                 # y = torch.from_numpy(y[0])
#                 y = torch.from_numpy(y)
#                 if self.task == 'regression':
#                     y = y.float().to(self.device)
#                 elif self.task == 'classification':
#                     if self.n_tasks == 1:
#                         y = y.long().to(self.device)
#                     else:
#                         y = y.float().to(self.device)
            
#             for key, value in tokens.items():
#                 tokens[key] = value.to(self.device)

#             inputs = {**tokens, 'labels': y}
#             return inputs, y, w