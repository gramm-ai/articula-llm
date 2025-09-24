#!/usr/bin/env python3
"""
Modern training script for chat-style QA models using latest Transformers API.

Features:
- Latest transformers API (4.46+)
- Support for both pre-tokenized and raw JSONL data
- Optional 4-bit quantization (bitsandbytes)
- Optional LoRA fine-tuning (PEFT)
- Gradient checkpointing support
- Mixed precision training (bf16/fp16)

Usage:
    # Pre-tokenized data (recommended)
    python train.py --train_dir data/train_tokenized --eval_dir data/val_tokenized

    # Raw JSONL data
    python train.py --train_file data/train.jsonl --val_file data/val.jsonl

    # With LoRA and 4-bit quantization
    python train.py --use_lora --use_4bit --lora_r 16 --lora_alpha 32
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Check minimum version requirements
check_min_version("4.46.0")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "microsoft/phi-4-mini-reasoning"
ASSISTANT_MARKER = "<|im_start|>assistant"
END_MARKER = "<|im_end|>"
SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<scratchpad>",
    "</scratchpad>",
    "<final>",
    "</final>",
    "user",
    "assistant",
]


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = field(
        default=DEFAULT_MODEL,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by tokenizers library)"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch, tag, or commit id)"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading models"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default torch.dtype and load model with another dtype. "
                    "Options: 'auto', 'bfloat16', 'float16', 'float32'"
        }
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit mode using bitsandbytes"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit mode using bitsandbytes"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to the data used for training and evaluation."""

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data file (JSONL format)"}
    )
    val_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation data file (JSONL format)"}
    )
    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized training dataset directory"}
    )
    eval_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized evaluation dataset directory"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of processes to use for data preprocessing"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad all samples to max_seq_length"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of evaluation examples"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA (Low-Rank Adaptation) configuration."""

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA scaling parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of module names to apply LoRA to. "
                    "If None, will automatically select common projection layers."
        }
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Options: 'none', 'all', 'lora_only'"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments with additional options."""

    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "The output directory for model checkpoints and logs"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy: 'no', 'steps', or 'epoch'"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use"}
    )


def find_substring_position(text: List[int], pattern: List[int]) -> int:
    """Find the position of a pattern in text. Returns -1 if not found."""
    if not pattern or len(pattern) > len(text):
        return -1

    for i in range(len(text) - len(pattern) + 1):
        if text[i:i + len(pattern)] == pattern:
            return i
    return -1


def create_labels_with_masking(
        input_ids: List[int],
        boundary_ids: List[int],
        mask_before_boundary: bool = True
) -> List[int]:
    """
    Create labels with masking for causal language modeling.
    Masks tokens before the assistant boundary marker.
    """
    labels = input_ids.copy()

    if mask_before_boundary:
        boundary_pos = find_substring_position(input_ids, boundary_ids)
        if boundary_pos != -1:
            # Mask everything up to and including the boundary
            mask_until = boundary_pos + len(boundary_ids)
            for i in range(mask_until):
                labels[i] = -100

    return labels


def tokenize_function(
        examples: Dict[str, Any],
        tokenizer: AutoTokenizer,
        max_length: int,
        assistant_marker: str = ASSISTANT_MARKER
) -> Dict[str, List]:
    """
    Tokenize examples and create labels with appropriate masking.
    """
    # Build full text from instruction and output
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"{instruction.rstrip()}\n\n{output.rstrip()}\n"
        texts.append(text)

    # Tokenize all texts
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )

    # Get assistant marker token IDs
    boundary_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

    # Create labels with masking
    labels = []
    valid_indices = []

    for idx, input_ids in enumerate(model_inputs["input_ids"]):
        label_ids = create_labels_with_masking(input_ids, boundary_ids)

        # Check if we have enough non-masked tokens
        non_masked = sum(1 for token in label_ids if token != -100)
        if non_masked >= 8:  # Minimum answer tokens
            labels.append(label_ids)
            valid_indices.append(idx)

    # Filter to keep only valid examples
    filtered_inputs = {
        "input_ids": [model_inputs["input_ids"][i] for i in valid_indices],
        "attention_mask": [model_inputs["attention_mask"][i] for i in valid_indices],
        "labels": labels
    }

    return filtered_inputs


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling with label padding.
    """
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels from features
        labels = [f.pop("labels", None) for f in features]

        # Pad input features
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Pad labels with -100
        if labels and labels[0] is not None:
            max_length = batch["input_ids"].shape[1]
            padded_labels = []

            for label_list in labels:
                if len(label_list) < max_length:
                    label_list = label_list + [-100] * (max_length - len(label_list))
                else:
                    label_list = label_list[:max_length]
                padded_labels.append(label_list)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def load_and_prepare_datasets(
        data_args: DataTrainingArguments,
        tokenizer: AutoTokenizer,
        logger: logging.Logger
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare training and evaluation datasets.
    """
    # Try loading pre-tokenized datasets first
    train_dataset = None
    eval_dataset = None

    if data_args.train_dir and os.path.exists(data_args.train_dir):
        logger.info(f"Loading pre-tokenized training data from {data_args.train_dir}")
        train_dataset = load_from_disk(data_args.train_dir)
    elif data_args.train_file and os.path.exists(data_args.train_file):
        logger.info(f"Loading and tokenizing training data from {data_args.train_file}")
        raw_dataset = load_dataset("json", data_files=data_args.train_file)["train"]

        # Tokenize dataset
        train_dataset = raw_dataset.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing training data",
        )
    else:
        logger.warning(
            f"Training data not found at default path ({data_args.train_dir}) "
            f"or no alternative path provided"
        )

    if data_args.eval_dir and os.path.exists(data_args.eval_dir):
        logger.info(f"Loading pre-tokenized evaluation data from {data_args.eval_dir}")
        eval_dataset = load_from_disk(data_args.eval_dir)
    elif data_args.val_file and os.path.exists(data_args.val_file):
        logger.info(f"Loading and tokenizing evaluation data from {data_args.val_file}")
        raw_dataset = load_dataset("json", data_files=data_args.val_file)["train"]

        # Tokenize dataset
        eval_dataset = raw_dataset.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing evaluation data",
        )
    else:
        logger.warning(
            f"Evaluation data not found at default path ({data_args.eval_dir}) "
            f"or no alternative path provided"
        )

    # Apply sample limits if specified
    if data_args.max_train_samples and train_dataset:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if data_args.max_eval_samples and eval_dataset:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    return train_dataset, eval_dataset


def setup_model_and_tokenizer(
        model_args: ModelArguments,
        lora_args: LoraArguments,
        logger: logging.Logger
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Setup model and tokenizer with appropriate configurations.
    """
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Add special tokens
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_tokens} special tokens")

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup model dtype
    torch_dtype = None
    if model_args.torch_dtype:
        if model_args.torch_dtype == "auto":
            torch_dtype = "auto"
        elif model_args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif model_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif model_args.torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid torch_dtype: {model_args.torch_dtype}")

    # Setup quantization config
    quantization_config = None
    if model_args.use_4bit or model_args.use_8bit:
        if not torch.cuda.is_available():
            raise ValueError("Quantization requires CUDA")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit,
            load_in_8bit=model_args.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype in [torch.bfloat16, "auto"] else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto" if (model_args.use_4bit or model_args.use_8bit) else None,
    )

    # Resize token embeddings if we added special tokens
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Setup LoRA if requested
    if lora_args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            # If using k-bit quantization with LoRA, prepare model so grads flow correctly
            if (model_args.use_4bit or model_args.use_8bit):
                try:
                    from peft import prepare_model_for_kbit_training
                    model = prepare_model_for_kbit_training(model)
                except Exception as e:
                    logger.warning(f"Failed to prepare model for k-bit training: {e}")

            # Determine target modules
            if lora_args.lora_target_modules:
                target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
            else:
                # Auto-detect common projection layers
                target_modules = []
                for name, module in model.named_modules():
                    if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj"]):
                        module_name = name.split(".")[-1]
                        if module_name not in target_modules:
                            target_modules.append(module_name)

                if not target_modules:
                    logger.warning("No target modules found for LoRA, using defaults")
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            logger.info(f"Applying LoRA to modules: {target_modules}")

            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                target_modules=target_modules,
                bias=lora_args.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        except ImportError:
            raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    return model, tokenizer


def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        CustomTrainingArguments,
        LoraArguments
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load arguments from JSON file
        model_args, data_args, training_args, lora_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Parse command line arguments
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}, "
        f"bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args, lora_args, logger)

    # Ensure gradient checkpointing works with PEFT/quantized models
    if training_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing support on model")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if lora_args.use_lora or model_args.use_4bit or model_args.use_8bit:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def _make_inputs_require_grad(module, _input, output):
                    if isinstance(output, torch.Tensor):
                        output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

        if getattr(model.config, "use_cache", None):
            model.config.use_cache = False

    # Load and prepare datasets
    train_dataset, eval_dataset = load_and_prepare_datasets(data_args, tokenizer, logger)

    if not train_dataset:
        raise ValueError("No training dataset found. Please provide --train_file or --train_dir")

    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # Setup data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Using processing_class (new API)
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.info("*** Starting training ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Save model
        trainer.save_model()

        # Save training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval and eval_dataset:
        logger.info("*** Running evaluation ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.train_file is not None:
        kwargs["dataset"] = data_args.train_file

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()