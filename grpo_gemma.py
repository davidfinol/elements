"""
GRPO Demo - Training Gemma with Group Relative Policy Optimization

This script is adapted from Tunix's example to fine-tune Gemma using GRPO.
It demonstrates training the Gemma 3 1B-IT model on the GSM8K math reasoning benchmark.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs.
It is a variant of PPO that reduces memory usage by eliminating the need for
a separate value function model.
"""

import os
from typing import Optional
import functools
import re
import json
import shutil
import logging
from datetime import datetime

from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from flax import nnx
import humanize
from huggingface_hub import snapshot_download
import jax
import optax
from orbax import checkpoint as ocp
import qwix
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

import nest_asyncio

# ============================================================================
# Logging Setup
# ============================================================================

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
log_filename = f"logs/grpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_filename}")

# ============================================================================
# JAX Device Configuration (GPU/TPU)
# ============================================================================

# Configure JAX to use GPU if available
os.environ['JAX_PLATFORMS'] = 'cuda,tpu,cpu'  # Try CUDA first, then TPU, then CPU

# Log available devices
logger.info("="*60)
logger.info("JAX Device Configuration")
logger.info("="*60)
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"Available devices: {jax.devices()}")
logger.info(f"Default backend: {jax.default_backend()}")

# Log device details
for device in jax.devices():
    logger.info(f"Device: {device} - Type: {device.platform}")

if jax.default_backend() == 'gpu':
    logger.info("✓ GPU detected and will be used for training")
elif jax.default_backend() == 'tpu':
    logger.info("✓ TPU detected and will be used for training")
else:
    logger.warning("⚠ No GPU/TPU detected. Training will run on CPU (this will be slow)")

# ============================================================================
# Environment Setup
# ============================================================================

# Load environment variables
load_dotenv()
logger.info("Loading environment variables")

# Apply nest_asyncio for async operations
nest_asyncio.apply()
logger.info("nest_asyncio applied")

# Login to Hugging Face if token is available
if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
    hf_token = os.environ["HF_TOKEN"]
    logger.info("HF_TOKEN found. Logging in to Hugging Face...")
    os.system(f'hf auth login --token "{hf_token}"')
else:
    logger.warning("HF_TOKEN not found. Skipping Hugging Face login.")

# ============================================================================
# Hyperparameters
# ============================================================================

# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = .9

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
# Adjust mesh based on your device count and model size.
NUM_DEVICES = len(jax.devices())
logger.info(f"Number of devices detected: {NUM_DEVICES}")

if NUM_DEVICES == 8:
    MESH_COUNTS = (1, 4)
    logger.info("Using mesh configuration for 8 devices: (1, 4)")
elif NUM_DEVICES == 1:
    MESH_COUNTS = (1, 1)
    logger.info("Using mesh configuration for 1 device: (1, 1)")
else:
    logger.warning(f"Unusual number of devices: {NUM_DEVICES}. Using default mesh (1, 1)")
    MESH_COUNTS = (1, 1)

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (μ in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (β) in the GRPO loss function.
BETA = 0.08
# Epsilon value for clipping (ε in GRPO loss in paper).
EPSILON = 0.2

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly.
NUM_TEST_BATCHES = 64

EVAL_EVERY_N_STEPS = 64
NUM_EPOCHS = 1

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# Linearly increase learning rate from 0. to LEARNING_RATE in the first 10% training steps
WARMUP_STEPS = 0.1 * MAX_STEPS
# Grad clipping to prevent large gradients
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": None, "top_k": 1, "top_p": None},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ============================================================================
# Utility Functions
# ============================================================================

def show_hbm_usage():
    """Displays memory usage per device."""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)

    logger.info("="*60)
    logger.info("Device Memory Usage")
    logger.info("="*60)
    for d in jax.local_devices():
        stats = d.memory_stats()
        if stats:
            used = stats["bytes_in_use"]
            limit = stats["bytes_limit"]
            logger.info(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")
        else:
            logger.info(f"No stats available")
# ============================================================================
# Data Preprocessing
# ============================================================================

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. First, think about the problem \
and provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""

# Load dataset
logger.info("="*60)
logger.info("Loading GSM8K dataset...")
logger.info("="*60)
dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"].rename_column("question", "prompts")
test_dataset = dataset["test"].rename_column("question", "prompts")

# For testing, use small fractions (comment these lines out for full training)
logger.warning("Using small dataset fractions for testing (5 samples). Comment out for full training!")
train_dataset = train_dataset.select(list(range(0, 5)))
test_dataset = test_dataset.select(list(range(0, 2)))

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Test dataset size: {len(test_dataset)}")

# ============================================================================
# Load Models
# ============================================================================

# Download model from Hugging Face
ignore_patterns = ["*.pth"]
logger.info("="*60)
logger.info(f"Downloading {MODEL_ID} from Hugging Face...")
logger.info("="*60)
local_model_path = snapshot_download(
    repo_id=MODEL_ID, ignore_patterns=ignore_patterns
)
logger.info(f"✓ Model successfully downloaded to: {local_model_path}")

# Load EOS tokens
EOS_TOKENS = []
generation_config_path = os.path.join(local_model_path, "generation_config.json")
if os.path.exists(generation_config_path):
    with open(generation_config_path, "r") as f:
        generation_configs = json.load(f)
    EOS_TOKENS = generation_configs.get("eos_token_id", [])
    logger.info(f"Using EOS token IDs: {EOS_TOKENS}")
else:
    logger.warning("generation_config.json not found. EOS tokens may need to be set manually.")

# Load model configuration
MODEL_CP_PATH = local_model_path

logger.info("="*60)
logger.info("Configuring model...")
logger.info("="*60)

model_config = None
if "gemma-3-270m" in MODEL_ID:
    model_config = gemma_lib.ModelConfig.gemma3_270m()
    logger.info("Using Gemma 3 270M configuration")
elif "gemma-3-1b" in MODEL_ID:
    model_config = gemma_lib.ModelConfig.gemma3_1b()
    logger.info("Using Gemma 3 1B configuration")
else:
    raise ValueError(f"Unknown model id: {MODEL_ID}")

# Create mesh
logger.info(f"Creating mesh with configuration: {MESH}")
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
logger.info(f"✓ Mesh created successfully")

# Load base model
logger.info("="*60)
logger.info("Loading base model...")
logger.info("="*60)
with mesh:
    gemma3 = params_safetensors_lib.create_model_from_safe_tensors(
        MODEL_CP_PATH, (model_config), mesh
    )
    logger.info("✓ Base model loaded successfully")
    # Prints the base model architecture
    # nnx.display(gemma3)

# ============================================================================
# LoRA Application
# ============================================================================

def get_lora_model(base_model, mesh):
    """Apply LoRA layers to the base model."""
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            ".*attn_vec_einsum"
        ),
        rank=RANK,
        alpha=ALPHA,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model

# Create policy model with LoRA
logger.info("="*60)
logger.info(f"Creating policy model with LoRA (rank={RANK}, alpha={ALPHA})...")
logger.info("="*60)
lora_policy = get_lora_model(gemma3, mesh=mesh)
logger.info("✓ Policy model with LoRA created successfully")
# Prints the LoRA policy model architecture
# nnx.display(lora_policy)

# Load tokenizer
logger.info("="*60)
logger.info("Loading tokenizer...")
logger.info("="*60)
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
logger.info("✓ Tokenizer loaded successfully")
if tokenizer.eos_id() not in EOS_TOKENS:
    EOS_TOKENS.append(tokenizer.eos_id())
    logger.info(f"Updated EOS token IDs: {EOS_TOKENS}")

# ============================================================================
# Reward Functions
# ============================================================================

# Regex for checking format match
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def match_format_exactly(prompts, completions, **kwargs):
    """Reward if format matches exactly."""
    return [
        0 if match_format.search(response) is None else 3.0
        for response in completions
    ]

def match_format_approximately(prompts, completions, **kwargs):
    """Reward if format matches partially."""
    scores = []
    for completion in completions:
        score = 0
        response = completion
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.find(reasoning_start) == 0 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """Reward if answer is correct or close."""
    responses = completions
    extracted_responses = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    assert len(extracted_responses) == len(answer), \
        f"{extracted_responses} and {answer} have mismatching length"

    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0
            except:
                score -= 0.5
        scores.append(score)
    return scores

# Regex for extracting numbers
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

def check_numbers(prompts, completions, answer, **kwargs):
    """Extract and check numerical answer."""
    question = prompts
    responses = completions

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    if len(question) > 0 and len(responses) > 0:
        logger.debug("="*60)
        logger.debug(f"Question:\t{question[0]}")
        logger.debug(f"Answer:\t{answer[0]}")
        logger.debug(f"Response:\t{responses[0][:100]}...")  # Truncate long responses
        logger.debug(f"Extracted:\t{extracted_responses[0]}")
        logger.debug("="*60)

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores

# ============================================================================
# Evaluation Functions
# ============================================================================

def generate(
    question: str,
    sampler,
    temperature: float = 0.7,
    top_k: int = 5,
    top_p: int = 0.95,
    seed: Optional[int] = None
):
    """Given prompt, generates text."""
    if isinstance(question, str):
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=question,
            ),
        ]
    else:
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=q,
            )
            for q in question
        ]

    logger.debug(f"Generating with temperature={temperature}, top_k={top_k}, top_p={top_p}")
    logger.debug(f"Input batch size: {len(input_batch)}")

    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=10,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed if seed is not None else None,
        eos_tokens=EOS_TOKENS,
    )

    output = out_data.text
    if isinstance(question, str):
        return output[0]
    return output

def evaluate(
    dataset: Dataset,
    sampler,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    num_passes: int = 1,
    corr_lst=False,
    make_lst=False,
):
    """Computes accuracy and percentage of outputs matching the format."""
    logger.info(f"Starting evaluation with {len(dataset)} samples, {num_passes} pass(es)")
    logger.info(f"Generation params: temperature={temperature}, top_k={top_k}, top_p={top_p}")

    response_lst = []
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    for batch in tqdm(dataset, desc="Evaluating"):
        answers = batch["answer"]
        questions = batch["prompts"]

        multiple_call_responses = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(
                questions, sampler, temperature, top_k, top_p, seed=p
            )
            logger.debug(f"Pass {p+1}/{num_passes} - Generated {len(responses)} responses")
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)
                logger.debug(f"Question:\t{questions[idx][:80]}...")
                logger.debug(f"Correct Answer:\t{answers[idx][:50]}...")
                logger.debug(f"Response:\t{response[:100]}...")
                logger.debug("-" * 50)

        for question, multiple_call_response, answer in zip(
            questions, multiple_call_responses, answers
        ):
            corr_ctr_per_question = 0
            partially_corr_per_question = 0
            corr_format_per_question = 0

            for response in multiple_call_response:
                extracted_response = (
                    guess.group(1)
                    if (guess := match_numbers.search(response)) is not None
                    else "-1000000"
                )
                try:
                    if float(extracted_response.strip()) == float(answer.strip()):
                        corr_ctr_per_question += 1

                    ratio = float(extracted_response.strip()) / float(answer.strip())
                    if ratio >= 0.9 and ratio <= 1.1:
                        partially_corr_per_question += 1
                except:
                    logger.debug("Skipped comparison due to parse error")

                if match_format.search(response) is not None:
                    corr_format_per_question += 1

                if (
                    corr_ctr_per_question > 0
                    and partially_corr_per_question > 0
                    and corr_format_per_question > 0
                ):
                    break

            if corr_ctr_per_question > 0:
                corr += 1
                if corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            else:
                if not corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            if partially_corr_per_question > 0:
                partially_corr += 1
            if corr_format_per_question > 0:
                corr_format += 1

            total += 1
            if total % 10 == 0:
                logger.info(
                    f"Progress: correct={corr}, total={total}, "
                    f"accuracy={corr / total * 100:.2f}%, "
                    f"partial_accuracy={partially_corr / total * 100:.2f}%, "
                    f"format_accuracy={corr_format / total * 100:.2f}%"
                )

    to_return = (
        corr,
        total,
        corr / total * 100,
        partially_corr / total * 100,
        corr_format / total * 100,
    )
    if make_lst:
        return to_return, response_lst
    return to_return

# ============================================================================
# Pre-training Evaluation
# ============================================================================

logger.info("\n" + "="*60)
logger.info("Evaluating model before training...")
logger.info("="*60)

# Show initial memory usage
show_hbm_usage()

sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)
logger.info("✓ Sampler created successfully")

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
logger.info("="*60)
logger.info("PRE-TRAINING EVALUATION RESULTS")
logger.info("="*60)
logger.info(f"Correct answers: {num_correct}/{total}")
logger.info(f"Accuracy: {accuracy:.2f}%")
logger.info(f"Partial accuracy: {partial_accuracy:.2f}%")
logger.info(f"Format accuracy: {format_accuracy:.2f}%")
logger.info("="*60)

# ============================================================================
# Training Setup
# ============================================================================

# Checkpoint saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/content/tmp/tensorboard/grpo", flush_every_n_steps=20
)

# Note: For Tensorboard visualization, run in a separate terminal:
# tensorboard --logdir /tmp/content/tmp/tensorboard/grpo --port=6006

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)

if MAX_GRAD_NORM is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
        optimizer,
    )

# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=EOS_TOKENS,
    ),
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)

# ============================================================================
# Initialize GRPO Trainer
# ============================================================================

logger.info("\n" + "="*60)
logger.info("Initializing GRPO trainer...")
logger.info("="*60)

# RL cluster
logger.info("Creating RL cluster with actor, reference, and tokenizer...")
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=gemma3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
logger.info("✓ RL cluster created successfully")

# GRPO Trainer
logger.info("Creating GRPO learner with reward functions...")
logger.info(f"GRPO Config: num_generations={NUM_GENERATIONS}, num_iterations={NUM_ITERATIONS}, "
           f"beta={BETA}, epsilon={EPSILON}")
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    algo_config=grpo_config,
)
logger.info("✓ GRPO trainer created successfully")

# ============================================================================
# Training
# ============================================================================

logger.info("\n" + "="*60)
logger.info("Starting training...")
logger.info("="*60)
logger.info(f"Training hyperparameters:")
logger.info(f"  - Max steps: {MAX_STEPS}")
logger.info(f"  - Learning rate: {LEARNING_RATE}")
logger.info(f"  - Batch size: {TRAIN_MICRO_BATCH_SIZE}")
logger.info(f"  - Warmup steps: {WARMUP_STEPS}")
logger.info(f"  - Max grad norm: {MAX_GRAD_NORM}")
logger.info(f"  - Checkpoint dir: {CKPT_DIR}")
logger.info(f"  - Save interval: {SAVE_INTERVAL_STEPS} steps")
logger.info("="*60)

# Show memory usage before training
show_hbm_usage()

# Split dataset into train and validation
val_dataset = test_dataset  # Using test as validation for this example

with mesh:
    logger.info("Starting GRPO training loop...")
    grpo_trainer.train(train_dataset, val_dataset)

logger.info("\n" + "="*60)
logger.info("✓ Training completed successfully!")
logger.info("="*60)

# Show memory usage after training
show_hbm_usage()

# ============================================================================
# Post-training Evaluation
# ============================================================================

print("\n" + "="*60)
print("Evaluating model after training...")
print("="*60)

# Load trained checkpoint
trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(1), "model_params"
)

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_policy, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_policy, nnx.LoRAParam),
        trained_lora_params,
    ),
)

# Re-create sampler with trained model
sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"Post-training results: {num_correct=}, {total=}, {accuracy=}%, "
    f"{partial_accuracy=}%, {format_accuracy=}%"
)

# ============================================================================
# Export Merged LoRA Weights
# ============================================================================

print("\n" + "="*60)
print("Exporting merged LoRA weights...")
print("="*60)

output_dir = f"./{MODEL_ID}-lora"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print(f"Saving merged LoRA model to {output_dir}")

# Save the merged model
gemma_params.save_lora_merged_model_as_safetensors(
    local_model_path=local_model_path,
    output_dir=output_dir,
    lora_model=lora_policy,
    rank=RANK,
    alpha=ALPHA,
)

print("\n" + "="*60)
print("Model saved successfully!")
print(f"Output directory: {output_dir}")
print("="*60)

print("\nSaved files:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
    print(f"  {f:<30} {size:>10.2f} MB")

print("\nScript completed successfully!")
