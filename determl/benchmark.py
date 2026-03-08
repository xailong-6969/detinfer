"""
determl.benchmark -- Auto-scaling Determinism Benchmark

Provides a diverse set of test prompts across multiple categories,
auto-scales based on model size to keep runtime practical:
  - Small models (<3B):   20 prompts × 5 runs
  - Medium models (3-13B): 10 prompts × 3 runs
  - Large models (13B+):    5 prompts × 2 runs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt Bank -- diverse prompts across categories
# ---------------------------------------------------------------------------

PROMPTS = {
    # --- Tier 1: Sanity checks (short output, high confidence) ---
    "sanity": [
        "What is 2 + 2? Answer with just the number.",
        "Is the sky blue? Answer yes or no.",
        "What is the capital of France? One word.",
    ],

    # --- Tier 2: Long output (100+ tokens, many CUDA ops) ---
    "long_output": [
        "Write a detailed explanation of how neural networks learn, covering forward pass, loss calculation, and backpropagation. Be thorough.",
        "Explain the difference between TCP and UDP protocols. Cover reliability, speed, use cases, and handshake mechanisms in detail.",
        "Write a complete Python class for a binary search tree with insert, search, delete, and in-order traversal methods. Include docstrings.",
        "Describe the water cycle in detail, covering evaporation, condensation, precipitation, and collection. Write at least 150 words.",
        "Explain how a compiler works, covering lexing, parsing, AST generation, optimization, and code generation. Be detailed.",
        "Write a detailed comparison of Python, JavaScript, and Rust. Cover type systems, performance, memory management, and use cases.",
    ],

    # --- Tier 3: Creative/uncertain (model has low confidence) ---
    "uncertain": [
        "Continue this story creatively: 'The astronaut opened the airlock and saw something impossible—'",
        "Write a short poem about a machine that dreams of being human.",
        "Invent a new word and define it. Then use it in three sentences.",
        "Describe an alien planet with purple oceans and glass mountains. Be creative and vivid.",
        "Write a dialogue between a philosopher and a robot about consciousness.",
        "Complete this sentence in an unexpected way: 'The last thing anyone expected at the funeral was'",
    ],

    # --- Tier 4: Complex code (deep computation paths) ---
    "complex_code": [
        "Write a Python function that implements merge sort. Include the merge helper function. Add type hints and comments.",
        "Write a Python implementation of a simple linked list with append, prepend, delete, find, and __str__ methods.",
        "Write a Python function to solve the N-Queens problem using backtracking. Include example usage for N=4.",
        "Implement a Python LRU cache class from scratch using a dictionary and a doubly linked list. Include get and put methods.",
        "Write a Python function that evaluates a mathematical expression string supporting +, -, *, / and parentheses. No eval().",
    ],

    # --- Tier 5: Multi-step reasoning (deep attention) ---
    "reasoning": [
        "A farmer has 17 sheep. All but 9 die. How many are left? Explain your reasoning step by step.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Show your work.",
        "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You pick one fruit from the 'Mixed' box and it's an apple. What's in each box? Explain step by step.",
        "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully and show your reasoning.",
        "You have 8 identical-looking balls. One is heavier. You have a balance scale. What's the minimum number of weighings to find the heavy ball? Explain the strategy.",
    ],

    # --- Tier 6: Long input context (stress attention mechanism) ---
    "deep_context": [
        "Analyze this Fibonacci and prime number code, explain what prime_fibs does, its time complexity, and suggest optimizations:\n```python\ndef fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b\n\ndef is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\n\ndef prime_fibs(limit):\n    return [fibonacci(i) for i in range(limit) if is_prime(fibonacci(i))]\n```",
        "Summarize this passage about machine learning in exactly 3 bullet points: 'Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, ML systems identify patterns in data to make decisions. There are three main types: supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based). Deep learning, a subset of ML, uses neural networks with many layers. Key challenges include overfitting, underfitting, data quality, and computational cost.'",
    ],

    # --- Tier 7: Adversarial (designed to trigger non-determinism) ---
    "adversarial": [
        "Generate a list of exactly 20 words that sound random but aren't.",
        "Write 10 sentences, each exactly 7 words long, on different topics.",
        "List the numbers from 1 to 50, but replace every multiple of 3 with 'fizz', every multiple of 5 with 'buzz', and every multiple of both with 'fizzbuzz'.",
        "Write a paragraph using every letter of the alphabet at least once. Make it coherent.",
        "Count backwards from 100 to 1, but skip every prime number. List the remaining numbers.",
    ],

    # --- Tier 8: Edge cases ---
    "edge_cases": [
        "Respond with exactly one character: the letter 'A'.",
        "What is 0 * 999999999? Answer with just the number.",
        "Repeat this string exactly without any changes: 'Hello! @#$% 世界 🌍 café naïve résumé'",
        "Output nothing but whitespace (spaces and newlines).",
    ],
}


def get_all_prompts() -> list[tuple[str, str]]:
    """Return all prompts as (category, prompt) tuples."""
    result = []
    for category, prompts in PROMPTS.items():
        for prompt in prompts:
            result.append((category, prompt))
    return result


def _short_prompt(prompt: str, max_len: int = 0) -> str:
    """Get a clean single-line summary of a prompt for display."""
    # Take first line only (multiline prompts like the Fibonacci one)
    first_line = prompt.split("\n")[0].strip()
    return first_line


# ---------------------------------------------------------------------------
# Auto-scaling logic
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark, auto-scales based on model size."""
    num_prompts: int = 20
    runs_per_prompt: int = 5
    depth: str = "auto"  # "auto", "light", "standard", "deep"

    @classmethod
    def from_depth(cls, depth: str, param_count_b: float | None = None) -> "BenchmarkConfig":
        """Create config based on depth level and optional model size."""
        if depth == "light":
            return cls(num_prompts=10, runs_per_prompt=3, depth=depth)
        elif depth == "standard":
            return cls(num_prompts=25, runs_per_prompt=5, depth=depth)
        elif depth == "deep":
            return cls(num_prompts=40, runs_per_prompt=10, depth=depth)
        elif depth == "auto" and param_count_b is not None:
            if param_count_b < 3.0:
                return cls(num_prompts=20, runs_per_prompt=5, depth=f"auto ({param_count_b:.1f}B)")
            elif param_count_b < 13.0:
                return cls(num_prompts=10, runs_per_prompt=3, depth=f"auto ({param_count_b:.1f}B)")
            else:
                return cls(num_prompts=5, runs_per_prompt=2, depth=f"auto ({param_count_b:.1f}B)")
        else:
            return cls(num_prompts=15, runs_per_prompt=3, depth="auto (unknown)")

    @property
    def total_runs(self) -> int:
        return self.num_prompts * self.runs_per_prompt


def estimate_param_count(model) -> float | None:
    """Estimate model parameter count in billions."""
    try:
        total = sum(p.numel() for p in model.parameters())
        return total / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    """Result for a single prompt across multiple runs."""
    category: str
    prompt: str
    hashes: list[str] = field(default_factory=list)

    @property
    def is_deterministic(self) -> bool:
        return len(set(self.hashes)) == 1

    @property
    def num_runs(self) -> int:
        return len(self.hashes)


@dataclass
class BenchmarkResult:
    """Result for the full benchmark run."""
    config: BenchmarkConfig
    prompt_results: list[PromptResult] = field(default_factory=list)
    model_name: str = ""
    gpu_name: str = ""
    seed: int = 42
    elapsed_seconds: float = 0.0

    @property
    def total_prompts(self) -> int:
        return len(self.prompt_results)

    @property
    def deterministic_count(self) -> int:
        return sum(1 for r in self.prompt_results if r.is_deterministic)

    @property
    def total_runs(self) -> int:
        return sum(r.num_runs for r in self.prompt_results)

    @property
    def matching_runs(self) -> int:
        return sum(r.num_runs for r in self.prompt_results if r.is_deterministic)

    @property
    def all_deterministic(self) -> bool:
        return self.deterministic_count == self.total_prompts

    def __str__(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  DETERML BENCHMARK RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Model:  {self.model_name}")
        lines.append(f"  GPU:    {self.gpu_name}")
        lines.append(f"  Seed:   {self.seed}")
        lines.append(f"  Depth:  {self.config.depth}")
        lines.append(f"  Time:   {self.elapsed_seconds:.1f}s")
        lines.append("-" * 70)
        lines.append(f"  {'Category':<14} {'Prompt':<38} {'Runs':<6} {'Result'}")
        lines.append("-" * 70)

        for r in self.prompt_results:
            cat = r.category
            prompt_short = _short_prompt(r.prompt, 35)
            runs = f"{r.num_runs}/{r.num_runs}"
            status = "\u2713 PASS" if r.is_deterministic else "\u2717 FAIL"
            lines.append(f"  {cat:<14} {prompt_short:<38} {runs:<6} {status}")

        lines.append("-" * 70)

        if self.all_deterministic:
            lines.append(f"  RESULT: {self.deterministic_count}/{self.total_prompts} DETERMINISTIC")
            lines.append(f"          {self.matching_runs}/{self.total_runs} total runs matched")
        else:
            failed = self.total_prompts - self.deterministic_count
            lines.append(f"  RESULT: {failed} FAILED out of {self.total_prompts} prompts")
            lines.append(f"          {self.matching_runs}/{self.total_runs} total runs matched")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    engine,
    config: BenchmarkConfig | None = None,
    max_new_tokens: int = 50,
) -> BenchmarkResult:
    """
    Run the determinism benchmark.

    Args:
        engine: A loaded DeterministicEngine
        config: Benchmark config (auto-detected if None)
        max_new_tokens: Max tokens per generation

    Returns:
        BenchmarkResult with all results
    """
    import torch

    # Auto-detect config if needed
    if config is None:
        param_b = estimate_param_count(engine.model) if engine.model else None
        config = BenchmarkConfig.from_depth("auto", param_b)

    # Get prompts and select subset
    all_prompts = get_all_prompts()
    # Distribute evenly across categories
    selected: list[tuple[str, str]] = []
    categories = list(PROMPTS.keys())
    per_cat = max(1, config.num_prompts // len(categories))
    for cat in categories:
        cat_prompts = [(c, p) for c, p in all_prompts if c == cat]
        selected.extend(cat_prompts[:per_cat])
    selected = selected[:config.num_prompts]

    # Get GPU name
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    result = BenchmarkResult(
        config=config,
        model_name=engine.model_name if hasattr(engine, 'model_name') else "unknown",
        gpu_name=gpu_name,
        seed=engine.seed if hasattr(engine, 'seed') else 42,
    )

    start = time.time()

    for idx, (category, prompt) in enumerate(selected):
        pr = PromptResult(category=category, prompt=prompt)

        for run in range(config.runs_per_prompt):
            run_result = engine.run(prompt, max_new_tokens=max_new_tokens)
            pr.hashes.append(run_result.canonical_hash)

        result.prompt_results.append(pr)

        # Progress
        status = "\u2713" if pr.is_deterministic else "\u2717"
        pct = (idx + 1) / len(selected) * 100
        prompt_display = _short_prompt(prompt, 45)
        print(f"  [{pct:5.1f}%] {status} {category:<14} {prompt_display}")

    result.elapsed_seconds = time.time() - start

    return result
