# Continual Learning with vLLM

## Vision

Enable LLMs to learn continuously during deployment - adapting to new data, user feedback, and domain shifts without catastrophic forgetting.

## Why This Matters

1. **Knowledge Currency**: Models become outdated; continual learning keeps them current
2. **Personalization**: Adapt to specific users/domains over time
3. **Feedback Integration**: Learn from corrections and preferences in real-time
4. **Efficient Updates**: No need for expensive full retraining cycles

## Key Challenges

1. **Catastrophic Forgetting**: Learning new things destroys old knowledge
2. **Compute Efficiency**: Updates must be fast (during/between inference)
3. **Memory Constraints**: Can't store unlimited replay buffers
4. **Serving Latency**: Learning shouldn't block inference
5. **Stability**: Model shouldn't degrade over time

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    vLLM Continual Learning Engine                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Unified Model Instance                        │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │   │
│  │  │  Base Model    │  │  LoRA Bank     │  │  Adapter Router    │  │   │
│  │  │  (frozen)      │  │  (trainable)   │  │  (selects adapter) │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│              ┌───────────────┼───────────────┐                          │
│              │               │               │                          │
│              ▼               ▼               ▼                          │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐        │
│  │ Inference Path   │ │ Learning     │ │ Anti-Forgetting      │        │
│  │                  │ │ Scheduler    │ │ Module               │        │
│  │ • Generate       │ │              │ │                      │        │
│  │ • Score          │ │ • When       │ │ • EWC                │        │
│  │ • Embed          │ │ • What       │ │ • Replay Buffer      │        │
│  │                  │ │ • How much   │ │ • Regularization     │        │
│  └──────────────────┘ └──────────────┘ └──────────────────────┘        │
│              │               │               │                          │
│              └───────────────┼───────────────┘                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Experience Store                               │   │
│  │  • Recent experiences (FIFO)                                      │   │
│  │  • Important experiences (priority queue)                         │   │
│  │  • Feedback signals (rewards, corrections)                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. LoRA Bank (Multi-Adapter System)

Instead of a single LoRA, maintain a bank of adapters:

```python
class LoRABank:
    """Bank of LoRA adapters for different knowledge domains/tasks."""

    def __init__(self, base_model, num_adapters: int = 8, rank: int = 16):
        self.adapters = nn.ModuleDict({
            f"adapter_{i}": LoRAAdapter(rank) for i in range(num_adapters)
        })
        self.router = AdapterRouter(num_adapters)  # Learns which adapter to use

    def forward(self, x, task_embedding=None):
        # Route to appropriate adapter(s)
        weights = self.router(x, task_embedding)  # [batch, num_adapters]

        # Weighted combination of adapters
        outputs = []
        for name, adapter in self.adapters.items():
            outputs.append(adapter(x))

        return sum(w * o for w, o in zip(weights.T, outputs))
```

**Benefits:**
- Different adapters for different domains/tasks
- Can add new adapters without affecting old ones
- Router learns task similarity automatically

### 2. Learning Scheduler

Decides WHEN and HOW to learn:

```python
class LearningScheduler:
    """Decides when to trigger learning updates."""

    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        self.update_counter = 0

    def should_learn(self) -> bool:
        """Check if we should trigger a learning update."""
        triggers = [
            len(self.experience_buffer) >= self.config.min_batch_size,
            self.update_counter >= self.config.update_frequency,
            self.experience_buffer.has_high_priority_samples(),
        ]
        return any(triggers)

    def get_learning_batch(self) -> LearningBatch:
        """Get a batch for learning, mixing new and replay samples."""
        new_samples = self.experience_buffer.get_recent(n=self.config.new_sample_ratio)
        replay_samples = self.experience_buffer.get_important(n=self.config.replay_ratio)
        return LearningBatch(new=new_samples, replay=replay_samples)
```

**Scheduling Strategies:**
- **Time-based**: Learn every N seconds
- **Sample-based**: Learn after N new samples
- **Uncertainty-based**: Learn when model is uncertain
- **Feedback-based**: Learn immediately on corrections

### 3. Anti-Forgetting Module

Prevents catastrophic forgetting:

```python
class AntiForgettingModule:
    """Prevents catastrophic forgetting during continual learning."""

    def __init__(self, method: str = "ewc"):
        self.method = method
        self.fisher_information = {}  # For EWC
        self.reference_params = {}    # For L2 regularization

    def compute_ewc_loss(self, model) -> torch.Tensor:
        """Elastic Weight Consolidation loss."""
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                ref_param = self.reference_params[name]
                ewc_loss += (fisher * (param - ref_param) ** 2).sum()
        return ewc_loss

    def update_fisher(self, model, dataloader):
        """Update Fisher information after learning a task."""
        # Compute diagonal Fisher information
        ...
```

**Anti-Forgetting Strategies:**
- **EWC (Elastic Weight Consolidation)**: Penalize changes to important weights
- **Replay**: Mix old samples with new ones
- **Progressive Nets**: Add new capacity, freeze old
- **PackNet**: Prune and reuse weights
- **LoRA Composition**: Separate adapters per domain

### 4. Experience Store

Efficient storage for continual learning:

```python
class ExperienceStore:
    """Manages experiences for continual learning."""

    def __init__(self, max_size: int = 100000):
        self.recent_buffer = deque(maxlen=max_size // 2)  # FIFO
        self.priority_buffer = PriorityQueue(maxsize=max_size // 2)

    def add(self, experience: Experience):
        """Add an experience with computed priority."""
        self.recent_buffer.append(experience)

        # Compute priority (based on loss, uncertainty, reward, etc.)
        priority = self.compute_priority(experience)
        if priority > self.priority_threshold:
            self.priority_buffer.put((-priority, experience))

    def sample_replay_batch(self, n: int) -> list[Experience]:
        """Sample a batch for replay."""
        recent = random.sample(list(self.recent_buffer), n // 2)
        important = [self.priority_buffer.get()[1] for _ in range(n // 2)]
        return recent + important
```

---

## Learning Modes

### Mode 1: Online Learning (During Inference)

Learn from every interaction in real-time:

```python
class OnlineLearner:
    """Learn from each inference request."""

    async def generate_and_learn(self, prompt: str, feedback: Feedback = None):
        # Generate response
        response = await self.generate(prompt)

        # Create experience
        experience = Experience(
            prompt=prompt,
            response=response,
            feedback=feedback,
            timestamp=time.time(),
        )

        # Store for learning
        self.experience_store.add(experience)

        # Maybe trigger learning (non-blocking)
        if self.scheduler.should_learn():
            asyncio.create_task(self.background_learn())

        return response
```

### Mode 2: Batch Periodic Learning

Learn in periodic batches (lower overhead):

```python
class PeriodicLearner:
    """Learn periodically from accumulated experiences."""

    def __init__(self, learn_interval: int = 3600):  # Every hour
        self.learn_interval = learn_interval
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.learn, 'interval', seconds=learn_interval)

    def learn(self):
        """Periodic learning job."""
        batch = self.experience_store.get_learning_batch()
        if len(batch) < self.min_batch_size:
            return

        # Compute loss with anti-forgetting
        loss = self.compute_loss(batch)
        loss += self.anti_forgetting.compute_ewc_loss(self.model)

        # Update
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### Mode 3: Feedback-Driven Learning

Learn from explicit feedback (corrections, preferences):

```python
class FeedbackLearner:
    """Learn from user feedback signals."""

    def learn_from_correction(self, prompt: str, bad_response: str, good_response: str):
        """Learn from a correction (like DPO)."""
        # Create preference pair
        experience = PreferenceExperience(
            prompt=prompt,
            chosen=good_response,
            rejected=bad_response,
            priority=HIGH,  # Corrections are high priority
        )

        # Immediate learning for corrections
        self.priority_learn([experience])

    def learn_from_rating(self, prompt: str, response: str, rating: float):
        """Learn from a rating signal."""
        experience = RatedExperience(
            prompt=prompt,
            response=response,
            reward=rating,
        )
        self.experience_store.add(experience)
```

---

## Research Experiments Enabled

### Experiment 1: Forgetting Dynamics

```python
# Measure how quickly the model forgets old knowledge
experiment = ForgettingExperiment(
    model=continual_model,
    task_sequence=["math", "code", "writing", "science"],
    eval_after_each_task=True,
)
results = experiment.run()
# Returns: forgetting_matrix[task_i][task_j] = performance on task_i after learning task_j
```

### Experiment 2: Replay Buffer Strategies

```python
# Compare different replay strategies
strategies = [
    RandomReplay(buffer_size=1000),
    PriorityReplay(buffer_size=1000, priority_fn=loss_based_priority),
    GradientReplay(buffer_size=1000),  # Maximize gradient diversity
    HerdingReplay(buffer_size=1000),   # Maintain class balance
]

for strategy in strategies:
    model = ContinualModel(replay_strategy=strategy)
    metrics = evaluate_continual_learning(model, task_stream)
```

### Experiment 3: Adapter Growth Strategies

```python
# When should we add a new adapter vs. update existing?
growth_strategies = [
    FixedAdapters(num_adapters=4),
    GrowOnDrift(drift_threshold=0.1),
    GrowOnSaturation(capacity_threshold=0.9),
    PackNet(pruning_ratio=0.5),
]
```

### Experiment 4: Learning Rate Schedules for CL

```python
# Optimal learning rate strategies for continual learning
lr_strategies = [
    ConstantLR(lr=1e-4),
    WarmRestarts(lr_max=1e-3, lr_min=1e-5, cycle=1000),
    AdaptiveLR(based_on="gradient_magnitude"),
    TaskAwareLR(base_lr=1e-4, task_lr_multipliers=learned),
]
```

---

## API for Researchers

```python
from vllm_train.continual import (
    ContinualLLM,
    ContinualConfig,
    ExperienceStore,
    ReplayStrategy,
    AntiForgettingMethod,
)

# Initialize continual learning model
config = ContinualConfig(
    # Base model
    model="meta-llama/Llama-3.1-8B",

    # LoRA settings
    lora_rank=16,
    num_adapters=4,
    adapter_routing="learned",  # or "task_id", "embedding_similarity"

    # Learning settings
    learning_mode="periodic",  # or "online", "feedback"
    update_frequency=100,      # Learn every 100 samples
    batch_size=8,
    learning_rate=1e-4,

    # Anti-forgetting
    anti_forgetting="ewc",     # or "replay", "packnet", "none"
    ewc_lambda=0.5,
    replay_ratio=0.3,

    # Experience store
    buffer_size=10000,
    priority_function="loss",  # or "uncertainty", "recency", "reward"
)

model = ContinualLLM(config)

# Use for inference
response = model.generate("What is 2+2?")

# Add experiences
model.add_experience(prompt="...", response="...", reward=1.0)

# Learn from feedback
model.learn_from_correction(
    prompt="What is the capital of France?",
    bad_response="London",
    good_response="Paris",
)

# Evaluate forgetting
metrics = model.evaluate_retention(test_sets=["math", "code", "writing"])

# Checkpoint (saves adapters + experience store + Fisher info)
model.save_checkpoint("./checkpoint")
```

---

## Implementation Priority

### Phase 1: Foundation (Enable Training in vLLM)
- [ ] Enable gradients for LoRA parameters in vLLM
- [ ] Basic training loop that works with vLLM model
- [ ] Simple experience store

### Phase 2: Anti-Forgetting
- [ ] EWC implementation
- [ ] Replay buffer with priority sampling
- [ ] Metrics for measuring forgetting

### Phase 3: Multi-Adapter System
- [ ] LoRA bank with multiple adapters
- [ ] Adapter routing (task-based and learned)
- [ ] Adapter merging/composition

### Phase 4: Online Learning
- [ ] Async learning during inference
- [ ] Feedback integration
- [ ] Learning scheduler

### Phase 5: Research Tools
- [ ] Experiment framework
- [ ] Visualization tools
- [ ] Benchmark suites
