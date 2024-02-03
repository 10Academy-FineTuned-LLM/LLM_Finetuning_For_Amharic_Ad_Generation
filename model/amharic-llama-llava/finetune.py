# importing modules
from sqlalchemy import false
import torch
from contextlib import nullcontext
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    TrainerCallback, 
    default_data_collator, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    PeftModel,
    prepare_model_for_kbit_training
)

from pathlib import Path
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import amharic_dataset
from datasets import load_dataset
from functools import partial




def get_max_length(model):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """

    # Pull model configuration
    conf = model.config
    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def preprocess_dataset(tokenizer, max_length, seed, dataset):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["instruction", "input", "output", "text"],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed = seed)

    return dataset

def create_prompt_formats(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """

    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample

def print_trainable_parameters(model):
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")


def finetune():
    LLAMA_DIR = "/model/Llama-2-7b-hf"
    PT_DIR = "/model/llama-2-amharic-3784m"
    OUTPUT_DIR = "./output"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(LLAMA_DIR, quantization_config=bnb_config, device_map='auto', torch_dtype=torch.float16)


    # Load dataset
    dataset_name = "/data/fine_tun_data2.json"
    dataset = load_dataset("json", data_files = dataset_name, split = "train")
    seed = 33

    max_length = get_max_length(model)
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
    #train_dataset = get_preprocessed_dataset(tokenizer, amharic_dataset, 'train')


    model.train()


    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) != embedding_size:
        print("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))


    print('loading the pretrained model from config')

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, PT_DIR)
    model.print_trainable_parameters()
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            #modules_to_save = ["embed_tokens","lm_head"]
        )

    enable_profiler = False


    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 1,
        'per_device_train_batch_size': 1,
        'gradient_checkpointing': False,
    }

    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUTPUT_DIR}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)

        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler

            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()


    # Define training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        warmup_ratio=0.03,
        optim="adamw_torch_fused",
        max_steps= 1000, #total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, mlm_probability=0.15
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

        print_trainable_parameters(model)

        # Start training
        trainer.train()

    model.save_pretrained(OUTPUT_DIR)


finetune()