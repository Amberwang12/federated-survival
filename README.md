# Federated Survival Analysis

A federated learning framework for survival analysis, enabling privacy-preserving collaborative learning across multiple institutions while maintaining data confidentiality.

## Features

- **Data Generation and Loading**: Support for both simulated data generation and real-world data loading
- **Data Partitioning**: Tools for splitting data into training and test sets, with training data distributed across federated learning clients
- **Federated Learning**: Implementation of Federated Averaging (FedAvg) algorithm for survival analysis
- **Multiple Models**: Support for various survival analysis models:
  - PC-Hazard
  - LogisticHazard
  - DeepHit
  - DeepSurv
  - CoxPH
  - CoxTime
  - CoxCC
- **Data Augmentation**: Support for client-side data augmentation using MVAEC and MVAES methods
- **Differential Privacy**: Optional differential privacy protection for enhanced privacy preservation
- **Evaluation Metrics**: Comprehensive evaluation including C-index and IBS metrics
- **Training History**: Support for tracking and returning training history

## Installation

You can install the package using pip:

```bash
pip install federated-survival
```

## Usage

### Data Generation and Loading

The framework provides comprehensive tools for generating simulated survival data with various characteristics:

```python
from federated_survival.data.generator import DataGenerator, SimulationConfig

# Configure data generation
sim_config = SimulationConfig(
    n_samples=100,      # Number of samples
    n_features=10,      # Number of features
    censoring_rate=0.2, # Rate of censored observations
    random_state=42     # Random seed for reproducibility
)


# Load real-world data
loader = DataLoader()
data = loader.load("path/to/your/data")


# Generate simulated data
generator = DataGenerator(config=sim_config)

# Generate data with different simulation types:
# 1. Accelerated Failure Time (AFT) Models:
data_weibull = generator.generate('weibull')    # Weibull AFT model
data_lognormal = generator.generate('lognormal') # Lognormal AFT model

# 2. Proportional Hazards Models:
data_sdgm1 = generator.generate('SDGM1')  # Standard proportional hazards
data_sdgm4 = generator.generate('SDGM4')  # Proportional hazards with log-normal errors

# 3. Non-Proportional Hazards Models:
data_sdgm2 = generator.generate('SDGM2')  # Mild violations of proportional hazards
data_sdgm3 = generator.generate('SDGM3')  # Strong violations of proportional hazards
```

The generated data includes:
- Features (x1, x2, ..., xp): Generated with AR(1) covariance structure
- Time: Observed survival/censoring time
- Status: Event indicator (1 = event, 0 = censored)

Each simulation type has different characteristics:
- `weibull`: Weibull AFT model with second half of features relevant
- `lognormal`: Lognormal AFT model with first and last 20% of features relevant
- `SDGM1`: Standard proportional hazards model
- `SDGM2`: Mild violations of proportional hazards with non-linear effects
- `SDGM3`: Strong violations of proportional hazards with shape parameter dependency
- `SDGM4`: Proportional hazards with log-normal errors and covariate-dependent censoring

### Data Partitioning

The framework provides flexible data partitioning methods to simulate various federated learning scenarios:

```python
from federated_survival.data.splitter import DataSplitter

# Initialize splitter with specific configuration
splitter = DataSplitter(
    n_clients=3,           # Number of federated learning clients
    split_type='iid',      # Partition type: 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'
    alpha=0.5,             # Dirichlet distribution parameter for non-IID splitting
    test_size=0.2,         # Proportion of test set
    random_state=42        # Random seed for reproducibility
)

# Split and distribute data to clients
client_data = splitter.split(data)
```

The `split` method returns a `DataSet` object containing:
- `clients_set`: Dictionary of client data, where each client's data is a tuple of (features, labels)
- `test_data`: Test set features
- `test_label`: Test set labels (time and status)
- `raw_aug_clients_set`: Placeholder for augmented client data

#### Available Partition Types

1. **IID (Independent and Identically Distributed)**
   - Ensures each client has the same censoring rate
   - Data is stratified by censoring status before splitting
   - Suitable for simulating ideal federated learning scenarios

2. **Non-IID (Non-Independent and Identically Distributed)**
   - Randomly splits data without maintaining censoring rate balance
   - Simulates scenarios where clients have different data distributions
   - Useful for testing model robustness

3. **Time-Non-IID**
   - Splits data based on survival time ranges
   - Maintains censoring status balance within each time range
   - Simulates scenarios where clients have different time distributions
   - Useful for testing temporal distribution shifts

4. **Dirichlet (Experimental)**
   - Uses Dirichlet distribution to create non-IID splits
   - Considers feature values when assigning samples to clients
   - Allows control over the degree of non-IID through alpha parameter
   - Useful for creating complex non-IID scenarios

#### Data Structure

The partitioned data follows this structure:
- Features (X): numpy array of shape (n_samples, n_features)
- Labels (y): numpy array of shape (n_samples, 2)
  - First column: survival/censoring time
  - Second column: event indicator (1 = event, 0 = censored)

### Federated Learning

The framework implements Federated Averaging (FedAvg) algorithm for survival analysis with support for multiple survival models.

#### FedAvg Algorithm

The Federated Averaging algorithm enables collaborative model training across multiple clients without sharing raw data. Here's the detailed algorithm:

**Algorithm: Federated Averaging for Survival Analysis**

```
Input:
  - K: Number of clients
  - E: Number of local epochs
  - T: Number of global communication rounds
  - η: Learning rate
  - C: Client sampling ratio (0 < C ≤ 1)
  - {D_k}: Local datasets at each client k

Initialization:
  - Initialize global model w_0 at server
  - Set random seed for reproducibility

For each global round t = 1, 2, ..., T:
  1. Server: Sample m = max(C·K, 1) clients randomly
     S_t ← random_sample(K, m)
  
  2. Server: Broadcast global model w_t to selected clients
  
  3. For each selected client k ∈ S_t (in parallel):
     a) Initialize local model: w_k^0 ← w_t
     
     b) For each local epoch e = 1, 2, ..., E:
        - Sample batch B from local dataset D_k
        - Compute loss: L_k(w_k^{e-1}, B)
        - Compute gradients: g_k ← ∇L_k(w_k^{e-1}, B)
        
        [If differential privacy enabled:]
          - Clip gradients: g_k ← clip(g_k, C_clip)
          - Add noise: g_k ← g_k + N(0, σ²I)
        
        - Update weights: w_k^e ← w_k^{e-1} - η·g_k
     
     c) Send local model w_k^E to server
  
  4. Server: Aggregate client models using weighted average
     w_{t+1} ← Σ_{k∈S_t} (n_k / n) · w_k^E
     
     where:
     - n_k: Number of samples at client k
     - n: Total samples across selected clients (n = Σ_{k∈S_t} n_k)
  
  5. Server: Evaluate aggregated model on test set
     - Compute C-index and IBS metrics
  
  6. [If early stopping enabled:]
     - Check if performance has not improved for p rounds
     - If true, stop training and return w_{t+1}

Output: Final global model w_T
```

**Key Features:**

1. **Client Sampling**: In each round, a subset of clients is randomly selected to participate in training, controlled by the client sampling ratio `C`.

2. **Local Training**: Each selected client trains the model locally for `E` epochs using its own data, without sharing raw data with the server or other clients.

3. **Weighted Aggregation**: The server aggregates client models by computing a weighted average, where weights are proportional to the number of samples at each client. This ensures that clients with more data have proportionally more influence on the global model.

4. **Privacy Preservation**: Raw data never leaves the client. Only model parameters (weights) are communicated between clients and server.

5. **Differential Privacy (Optional)**: When enabled, gradient clipping and Gaussian noise addition provide formal privacy guarantees:
   - Gradient clipping: `g_clipped = g · min(1, C_clip / ||g||_2)`
   - Noise addition: `g_noisy = g_clipped + N(0, σ²I)` where `σ = (sensitivity × noise_multiplier) / √K`

6. **Convergence**: The algorithm converges when:
   - Maximum number of global rounds `T` is reached, or
   - Early stopping criterion is met (no improvement for `p` consecutive rounds)

**Mathematical Formulation:**

The objective is to minimize the global loss function:

$$F(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w)$$

where:
- $F_k(w)$ is the local loss at client $k$
- $n_k$ is the number of samples at client $k$
- $n = \sum_{k=1}^K n_k$ is the total number of samples

The local loss for survival analysis is model-dependent:
- **PC-Hazard/LogisticHazard**: Negative log-likelihood of discrete hazard
- **DeepHit**: Deep learning loss with competing risks
- **CoxPH/DeepSurv/CoxCC**: Cox partial likelihood
- **CoxTime**: Time-dependent Cox loss

**Communication Efficiency:**

The algorithm requires:
- **Downlink communication** (server → clients): `T × m × |w|` where `|w|` is model size
- **Uplink communication** (clients → server): `T × m × |w|`
- **Total communication**: `2 × T × m × |w|`

Communication can be reduced by:
- Decreasing client sampling ratio `C`
- Increasing local epochs `E` (more local work per round)
- Using model compression techniques (not currently implemented)

**Convergence Guarantees:**

Under standard assumptions (convexity, smoothness, bounded gradients), FedAvg converges at rate:

$$\mathbb{E}[F(w_T) - F(w^*)] \leq O\left(\frac{1}{T}\right)$$

where $w^*$ is the optimal solution. In practice, convergence depends on:
- Data heterogeneity across clients (IID vs non-IID)
- Number of local epochs `E`
- Learning rate `η`
- Client sampling ratio `C`

#### Usage Example

```python
from federated_survival.core.runner import FSARunner
from federated_survival.core.config import FSAConfig

# Configure the federated learning process
config = FSAConfig(
    num_clients=3,           # Number of federated learning clients
    n_features=10,           # Number of features
    n_samples=100,           # Number of samples
    censor_rate=0.3,         # Censoring rate
    model_type='PC-Hazard',  # Survival model type
    local_epochs=2,          # Number of local training epochs
    global_epochs=2,         # Number of global communication rounds
    learning_rate=0.01,      # Learning rate
    batch_size=32,           # Batch size
    random_seed=42,          # Random seed
    client_sample_ratio=0.5, # Ratio of clients selected in each round
    early_stopping=True,     # Enable early stopping
    early_stopping_patience=5 # Number of epochs to wait before early stopping
)

# Initialize and run the federated learning process
runner = FSARunner(config)
results = runner.run(client_data)

# Access evaluation metrics
train_cindex = results['train_Cindex']
train_ibs = results['train_IBS']
test_cindex = results['test_Cindex']
test_ibs = results['test_IBS']
```

#### Available Survival Models

1. **PC-Hazard**
   - Piecewise constant hazard model
   - Discretizes time into intervals
   - Suitable for general survival analysis tasks
   - Uses quantile-based time discretization

2. **LogisticHazard**
   - Logistic regression-based hazard model
   - Similar to PC-Hazard but with logistic activation
   - Better for modeling smooth hazard functions
   - Uses quantile-based time discretization

3. **DeepHit**
   - Deep learning-based survival model
   - Can capture complex non-linear relationships
   - Handles competing risks
   - Uses quantile-based time discretization

4. **DeepSurv**
   - Deep learning-based survival model
   - Can capture complex non-linear relationships
   - Assumes proportional hazards
   - No time discretization needed
   - Good baseline model

5. **CoxPH**
   - Traditional Cox proportional hazards model
   - Assumes proportional hazards
   - No time discretization needed
   - Good baseline model

6. **CoxTime**
   - Time-dependent Cox model
   - Allows time-varying effects
   - More flexible than CoxPH
   - No time discretization needed

7. **CoxCC**
   - Case-control Cox model
   - Efficient for large datasets
   - Suitable for matched case-control studies
   - No time discretization needed

#### Training Process

1. **Initialization**
   - Set random seeds for reproducibility
   - Initialize global model at server
   - Create client models with local data

2. **Federated Training**
   - For each global epoch:
     - Select a subset of clients (controlled by client_sample_ratio)
     - Each selected client performs local training
     - Clients send model updates to server
     - Server aggregates updates using FedAvg
     - Update global model

3. **Model Evaluation**
   - Track training metrics (C-index and IBS)
   - Evaluate on test set
   - Support for early stopping
   - Visualization of training progress

#### Evaluation Metrics

1. **C-index (Concordance Index)**
   - Measures model's ability to correctly rank survival times
   - Range: 0.5 (random) to 1.0 (perfect)
   - Higher values indicate better performance

2. **IBS (Integrated Brier Score)**
   - Measures accuracy of predicted survival probabilities
   - Range: 0.0 (perfect) to 0.25 (worst)
   - Lower values indicate better performance

#### Visualization

```python
# Plot training results
runner.plot_results(
    raw_results=results,          # Results from raw data
)
```

The plot shows:
- C-index over training epochs
- IBS over training epochs
- C-index over test epochs
- IBS over test epochs

### Data Augmentation

The framework provides two advanced data augmentation methods for survival analysis in federated learning:

```python
from federated_survival.core.runner import FSARunner
from federated_survival.core.config import FSAConfig

# Configure the federated learning process with augmentation
config = FSAConfig(
    num_clients=3,
    n_features=10,
    n_samples=100,
    censor_rate=0.3,
    model_type='PC-Hazard',
    local_epochs=2,
    global_epochs=2,
    learning_rate=0.01,
    batch_size=32,
    random_seed=42,
    # Augmentation parameters
    latent_num=10,    # Dimension of latent space
    hidden_num=30,    # Dimension of hidden layer
    alpha=1.0,        # Weight for KL divergence
    beta=1.0,         # Weight for conditional loss
    k=0.5            # Augmentation ratio (0 < k <= 1)
)

# Initialize and run with data augmentation
runner = FSARunner(config)
results = runner.run(
    client_data,
    type='raw_aug',
    aug_method='MVAEC'  # or 'MVAES'
)
```

#### Available Augmentation Methods

1. **MVAEC (Multi-task Variational Autoencoder at Each Client)**
   - Each client generates augmented data using its own data
   - Uses a variational autoencoder trained on uncensored samples
   - Maintains data privacy as no data is shared between clients
   - Suitable for scenarios with strong privacy requirements
   - Augmentation ratio (k) controls the amount of generated data

2. **MVAES (Multi-task Variational Autoencoder at the Server)**
   - Collects augmented data from all clients at the server
   - Redistributes the augmented data to clients
   - May improve data diversity across clients
   - Requires more communication overhead
   - Useful when clients have limited local data

#### Augmentation Process

1. **Data Preparation**
   - Only uncensored samples are used for training the VAE
   - Each client must have at least 10 samples
   - Each client must have at least one uncensored sample

2. **Model Training**
   - Uses a variational autoencoder with configurable architecture
   - Latent space dimension can be adjusted (default: 10)
   - Hidden layer dimension can be configured (default: 30)
   - KL divergence weight (alpha) controls the trade-off between reconstruction and regularization
   - Conditional loss weight (beta) controls the importance of survival time prediction

3. **Data Generation**
   - Generates new samples in the latent space
   - Maintains the statistical properties of the original data
   - Preserves the relationship between features and survival time
   - Augmentation ratio (k) determines the number of generated samples

#### Usage Considerations

- Choose MVAEC when:
  - Privacy is a primary concern
  - Clients have sufficient local data
  - Communication overhead should be minimized

- Choose MVAES when:
  - Data diversity is important
  - Clients have limited local data
  - Communication overhead is acceptable

- Parameter Tuning:
  - Adjust latent_num and hidden_num based on data complexity
  - Modify alpha and beta to control the balance between reconstruction and regularization
  - Set k according to the desired amount of augmented data

### Differential Privacy

The framework supports **three differential privacy mechanisms** to enhance privacy preservation in federated learning. Each mechanism provides different privacy-utility trade-offs suitable for various scenarios.

#### Overview of Three Mechanisms

| Mechanism | Privacy Guarantee | Noise Type | Best Use Case |
|-----------|------------------|------------|---------------|
| **Gaussian** | (ε, δ)-DP | Normal Distribution | Deep learning gradients |
| **Laplace** | ε-DP | Laplace Distribution | Counting/sum queries |
| **Exponential** | ε-DP | Probability Sampling | Model selection |

#### 1. Gaussian Mechanism (Default)

The Gaussian mechanism provides (ε, δ)-differential privacy and is ideal for deep learning scenarios.

```python
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

# Configure with Gaussian mechanism
config = FSAConfig(
    num_clients=5,
    n_features=20,
    n_samples=1000,
    model_type='PC-Hazard',
    global_epochs=50,
    
    # Gaussian mechanism parameters
    use_differential_privacy=True,
    dp_mechanism='gaussian',       # Gaussian mechanism (default)
    dp_epsilon=1.0,               # Privacy budget (ε)
    dp_delta=1e-5,                # Failure probability (δ) - required for Gaussian
    dp_sensitivity=1.0,           # Sensitivity
    dp_noise_multiplier=1.0,      # Noise multiplier - Gaussian specific
    dp_clip_norm=1.0,             # Gradient clipping norm
)

runner = FSARunner(config)
results = runner.run(client_data)

# Get privacy information
privacy_info = runner.get_privacy_info()
print(f"Mechanism: {privacy_info['mechanism']}")
print(f"Privacy budget (ε): {privacy_info['epsilon']}")
print(f"Failure probability (δ): {privacy_info['delta']}")
print(f"Noise scale: {privacy_info['noise_scale']}")
```

**When to use Gaussian mechanism:**
- Federated learning with gradient-based optimization
- Deep learning model training
- Scenarios requiring multiple rounds of training
- When (ε, δ)-DP is acceptable

**Mathematical Foundation:**

For a function $f$ with sensitivity $\Delta f$, the Gaussian mechanism adds noise:

$$\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 I)$$

where the noise scale is:

$$\sigma = \frac{\Delta f \sqrt{2\ln(1.25/\delta)}}{\epsilon}$$

#### 2. Laplace Mechanism

The Laplace mechanism provides pure ε-differential privacy without requiring δ.

```python
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

# Configure with Laplace mechanism
config = FSAConfig(
    num_clients=5,
    n_features=20,
    n_samples=1000,
    model_type='PC-Hazard',
    global_epochs=50,
    
    # Laplace mechanism parameters
    use_differential_privacy=True,
    dp_mechanism='laplace',       # Laplace mechanism
    dp_epsilon=1.0,               # Privacy budget (ε)
    dp_sensitivity=1.0,           # Sensitivity
    dp_clip_norm=1.0,             # Gradient clipping norm
    # Note: dp_delta and dp_noise_multiplier are not needed for Laplace
)

runner = FSARunner(config)
results = runner.run(client_data)

# Get privacy information
privacy_info = runner.get_privacy_info()
print(f"Mechanism: {privacy_info['mechanism']}")
print(f"Privacy budget (ε): {privacy_info['epsilon']}")
print(f"Clip norm: {privacy_info['clip_norm']}")
# Note: No 'delta' or 'noise_multiplier' in Laplace mechanism
```

**When to use Laplace mechanism:**
- Counting queries or sum queries
- Scenarios requiring pure ε-DP (no δ)
- Low-dimensional numerical queries
- When stricter privacy guarantees are needed

**Mathematical Foundation:**

For a function $f$ with sensitivity $\Delta f$, the Laplace mechanism adds noise:

$$\mathcal{M}(D) = f(D) + \text{Lap}(b)$$

where the scale parameter is:

$$b = \frac{\Delta f}{\epsilon}$$

The Laplace distribution has probability density:

$$p(x|b) = \frac{1}{2b}\exp\left(-\frac{|x|}{b}\right)$$

#### 3. Exponential Mechanism

The Exponential mechanism is designed for discrete selection problems where adding noise directly is not appropriate.

```python
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy
import torch

# Configure with Exponential mechanism
config = FSAConfig(
    use_differential_privacy=True,
    dp_mechanism='exponential',   # Exponential mechanism
    dp_epsilon=1.0,               # Privacy budget (ε)
    dp_sensitivity=1.0,           # Quality function sensitivity
    # Note: No gradient-related parameters needed
)

dp_tool = DifferentialPrivacy(config)

# Example: Select best model configuration
candidate_configs = torch.randn(5, 100)  # 5 candidate configurations
quality_scores = torch.tensor([0.75, 0.80, 0.85, 0.78, 0.82])  # Validation scores

# Use exponential mechanism to select
selected_idx = dp_tool.exponential_mechanism(
    candidates=candidate_configs,
    quality_scores=quality_scores,
    epsilon=1.0
)

print(f"Selected configuration: {selected_idx}")
print(f"Quality score: {quality_scores[selected_idx]:.4f}")

# Or get the selected configuration directly
selected_config = dp_tool.exponential_mechanism_tensor(
    candidates=candidate_configs,
    quality_scores=quality_scores
)
```

**When to use Exponential mechanism:**
- Model selection among discrete candidates
- Hyperparameter tuning
- Selecting best client for aggregation
- Any discrete choice problem

**Mathematical Foundation:**

For a quality function $q: D \times R \rightarrow \mathbb{R}$ with sensitivity $\Delta q$, the exponential mechanism selects output $r \in R$ with probability:

$$P(r) \propto \exp\left(\frac{\epsilon \cdot q(D, r)}{2\Delta q}\right)$$

#### Mechanism Comparison

**Privacy Guarantees:**

| Mechanism | Privacy Type | Parameters Required | Noise Characteristics |
|-----------|-------------|--------------------|-----------------------|
| Gaussian | (ε, δ)-DP | ε, δ, sensitivity, noise_multiplier, clip_norm | Normal distribution, symmetric |
| Laplace | ε-DP | ε, sensitivity, clip_norm | Laplace distribution, heavier tails |
| Exponential | ε-DP | ε, sensitivity | Probability sampling, no noise |

**Performance Characteristics:**

```python
# Example: Compare three mechanisms
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

mechanisms = ['gaussian', 'laplace']
results_dict = {}

for mechanism in mechanisms:
    config = FSAConfig(
        num_clients=5,
        n_features=20,
        model_type='PC-Hazard',
        global_epochs=30,
        use_differential_privacy=True,
        dp_mechanism=mechanism,
        dp_epsilon=1.0,
        dp_delta=1e-5 if mechanism == 'gaussian' else None,
        dp_sensitivity=1.0,
    )
    
    runner = FSARunner(config)
    results = runner.run(client_data)
    results_dict[mechanism] = results
    
    print(f"\n{mechanism.upper()} Mechanism:")
    print(f"  Final C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  Final IBS: {results['test_IBS'][-1]:.4f}")
```

#### Differential Privacy Parameters

**Common Parameters (All Mechanisms):**

1. **dp_mechanism** (string)
   - Specifies which DP mechanism to use
   - Options: `'gaussian'`, `'laplace'`, `'exponential'`
   - Default: `'gaussian'`

2. **dp_epsilon** (float)
   - Privacy budget (ε)
   - Lower values = stronger privacy, potentially lower utility
   - Typical range: 0.1 to 10.0
   - Default: 1.0

3. **dp_sensitivity** (float)
   - Maximum change in output when one sample is added/removed
   - Affects the amount of noise/probability distribution
   - Default: 1.0

**Gaussian-Specific Parameters:**

4. **dp_delta** (float)
   - Failure probability (δ) for (ε, δ)-DP
   - Should be much smaller than 1/n (n = dataset size)
   - Typical range: 1e-6 to 1e-3
   - Default: 1e-5
   - **Note**: Only required for Gaussian mechanism

5. **dp_noise_multiplier** (float)
   - Controls the scale of added Gaussian noise
   - Higher values = more privacy, lower utility
   - Default: 1.0
   - **Note**: Only used by Gaussian mechanism

**Gradient-Based Parameters (Gaussian and Laplace):**

6. **dp_clip_norm** (float)
   - Maximum L2 norm for gradient clipping
   - Helps control sensitivity in gradient-based methods
   - Default: 1.0
   - **Note**: Used by Gaussian and Laplace mechanisms

#### Privacy Protection Mechanisms

**1. Gradient Clipping** (Gaussian and Laplace)
   - Clips gradients to control sensitivity
   - Applied during local training at each client
   - Prevents gradients from becoming too large
   - Formula: $\text{clip\_coef} = \min(1.0, \frac{C}{\|\nabla f\|_2 + \epsilon})$

**2. Noise Addition**

   **Gaussian Mechanism:**
   - Adds calibrated Gaussian noise to gradients
   - Noise scale depends on privacy parameters
   - Applied during local training only
   - Formula: $\text{noise} \sim \mathcal{N}(0, \sigma^2 I)$ 
   - Where: $\sigma = \frac{\text{sensitivity} \times \text{noise\_multiplier}}{\sqrt{\text{num\_clients}}}$

   **Laplace Mechanism:**
   - Adds Laplace noise to gradients
   - Simpler than Gaussian, pure ε-DP
   - Formula: $\text{noise} \sim \text{Lap}(b)$
   - Where: $b = \frac{\text{sensitivity}}{\epsilon}$

**3. Probability Sampling** (Exponential Mechanism)
   - Selects outputs based on quality scores
   - No noise added, uses probability distribution
   - Maintains output format and semantic meaning
   - Selection probability: $P(r) \propto \exp\left(\frac{\epsilon \cdot q(r)}{2\Delta q}\right)$

#### Mathematical Foundations

**Differential Privacy Definition:**

A mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-differential privacy if for any two adjacent datasets $D$ and $D'$ differing in at most one record, and any subset $S$ of outputs:

$$P[\mathcal{M}(D) \in S] \leq e^{\epsilon} \cdot P[\mathcal{M}(D') \in S] + \delta$$

For pure ε-DP (Laplace and Exponential), δ = 0.

**1. Gaussian Mechanism:**

For a function $f$ with sensitivity $\Delta f$, the Gaussian mechanism:

$$\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 I)$$

Noise scale for (ε, δ)-DP:

$$\sigma = \frac{\Delta f \sqrt{2\ln(1.25/\delta)}}{\epsilon}$$

**2. Laplace Mechanism:**

For a function $f$ with sensitivity $\Delta f$, the Laplace mechanism:

$$\mathcal{M}(D) = f(D) + \text{Lap}(b)$$

Scale parameter for ε-DP:

$$b = \frac{\Delta f}{\epsilon}$$

Laplace distribution PDF:

$$p(x|b) = \frac{1}{2b}\exp\left(-\frac{|x|}{b}\right)$$

**3. Exponential Mechanism:**

For a quality function $q: D \times R \rightarrow \mathbb{R}$ with sensitivity $\Delta q$:

$$P[\mathcal{M}(D) = r] = \frac{\exp(\epsilon q(D,r)/(2\Delta q))}{\sum_{r' \in R}\exp(\epsilon q(D,r')/(2\Delta q))}$$

This provides ε-DP without adding noise to outputs.

**Composition Theorem:**
For $k$ mechanisms each satisfying $(\epsilon_i, \delta_i)$-differential privacy, the composition satisfies:

$$\left(\sum_{i=1}^k \epsilon_i, \sum_{i=1}^k \delta_i\right)\text{-differential privacy}$$

**Renyi Differential Privacy:**
For order $\alpha > 1$, the Renyi divergence is:

$$D_\alpha(P\|Q) = \frac{1}{\alpha-1}\log\mathbb{E}_{x \sim Q}\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

The mechanism satisfies $(\alpha, \epsilon)$-RDP if:

$$D_\alpha(\mathcal{M}(D)\|\mathcal{M}(D')) \leq \epsilon$$

**Privacy Budget Calculation:**
For federated learning with $T$ rounds and $K$ clients per round:

- Per-round privacy: $\epsilon_{\text{round}} = \frac{\epsilon_{\text{total}}}{T}$
- Noise scale: $\sigma = \frac{\sqrt{2\ln(1.25/\delta)} \cdot \text{sensitivity}}{\epsilon_{\text{round}}}$
- Effective noise: $\sigma_{\text{effective}} = \frac{\sigma}{\sqrt{K}}$ (due to averaging over $K$ clients)

#### Privacy-Utility Trade-off

- **Higher Privacy (lower ε)**: More noise, potentially lower model performance
- **Lower Privacy (higher ε)**: Less noise, better model performance
- **Balanced Approach**: Choose ε based on privacy requirements and acceptable utility loss

#### Usage Guidelines

**Choosing the Right Mechanism:**

```python
# Decision flowchart
if task == "federated_learning_gradients":
    mechanism = 'gaussian'      # Best for deep learning
elif task == "counting_queries":
    mechanism = 'laplace'       # Best for numerical queries
elif task == "model_selection":
    mechanism = 'exponential'   # Best for discrete choices
```

**Mechanism Selection Criteria:**

| Scenario | Recommended Mechanism | Reason |
|----------|----------------------|--------|
| Deep learning training | Gaussian | Good composition properties, works well with SGD |
| Counting queries | Laplace | Pure ε-DP, simpler, no δ needed |
| Sum aggregation | Laplace | Direct noise addition, easier to analyze |
| Hyperparameter tuning | Exponential | Maintains output format, discrete selection |
| Model selection | Exponential | Probability-based, no noise distortion |
| Multiple training rounds | Gaussian | Better privacy budget accounting |

**When to Enable Differential Privacy:**

- Working with sensitive data (medical, financial, personal)
- Privacy regulations require protection (GDPR, HIPAA)
- Clients are concerned about data leakage
- Multi-party collaboration requires trust
- Public release of model updates

**Privacy-Utility Trade-off:**

| ε Value | Privacy Level | Noise Impact | Recommended For |
|---------|--------------|--------------|----------------|
| 0.1 | Very High | Very High | Extremely sensitive data |
| 0.5 | High | High | Medical/financial data |
| 1.0 | Medium | Medium | **General recommendation** |
| 2.0 | Moderate | Moderate | Business data |
| 5.0 | Low | Low | Public datasets |
| 10.0+ | Very Low | Minimal | Testing/debugging |

**Parameter Selection Guide:**

```python
# High privacy scenario (medical data)
config_high_privacy = FSAConfig(
    use_differential_privacy=True,
    dp_mechanism='gaussian',
    dp_epsilon=0.5,      # Strong privacy
    dp_delta=1e-6,       # Very small failure probability
    dp_clip_norm=0.5,    # Conservative clipping
)

# Balanced scenario (general use)
config_balanced = FSAConfig(
    use_differential_privacy=True,
    dp_mechanism='gaussian',
    dp_epsilon=1.0,      # Moderate privacy
    dp_delta=1e-5,       # Standard failure probability
    dp_clip_norm=1.0,    # Standard clipping
)

# Utility-focused scenario (less sensitive data)
config_utility = FSAConfig(
    use_differential_privacy=True,
    dp_mechanism='laplace',  # Pure ε-DP
    dp_epsilon=5.0,          # Weaker privacy, better utility
    dp_clip_norm=2.0,        # More generous clipping
)
```

#### Implementation Examples

Here's how the three mechanisms are implemented in the code:

**1. Gaussian Mechanism Implementation:**

```python
# Gradient clipping (common to Gaussian and Laplace)
def clip_gradients(self, model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    # Apply clipping: clip_coef = min(1.0, C / ||grad||_2)
    clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(clip_coef)
    return total_norm

# Gaussian noise addition
def add_gaussian_noise(self, tensor, sensitivity=None):
    if sensitivity is None:
        sensitivity = self.sensitivity
    
    # Calculate noise scale: σ = sensitivity × noise_multiplier
    sigma = sensitivity * self.noise_multiplier
    
    # Generate Gaussian noise: noise ~ N(0, σ²I)
    noise = torch.normal(0, sigma, size=tensor.shape, 
                        device=tensor.device, dtype=tensor.dtype)
    return tensor + noise
```

**2. Laplace Mechanism Implementation:**

```python
def add_laplace_noise(self, tensor, sensitivity=None, epsilon=None):
    if sensitivity is None:
        sensitivity = self.sensitivity
    if epsilon is None:
        epsilon = self.epsilon
    
    # Calculate Laplace scale: b = Δf / ε
    scale = sensitivity / epsilon
    
    # Generate Laplace noise
    noise_np = np.random.laplace(loc=0.0, scale=scale, size=tensor.shape)
    noise = torch.from_numpy(noise_np).to(device=tensor.device, dtype=tensor.dtype)
    
    return tensor + noise
```

**3. Exponential Mechanism Implementation:**

```python
def exponential_mechanism(self, candidates, quality_scores, 
                          sensitivity=None, epsilon=None):
    if sensitivity is None:
        sensitivity = self.sensitivity
    if epsilon is None:
        epsilon = self.epsilon
    
    # Calculate selection probability: P(r) ∝ exp(ε·q(r) / (2·Δq))
    scores = quality_scores.cpu().numpy()
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)
    
    # Sample based on probabilities
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    
    return selected_idx
```

**Applying DP in Federated Learning:**

```python
# In client.py - local training with DP
def local_train(self, global_model, epoch):
    # ... training code ...
    
    # Apply differential privacy to gradients
    if self.dp_tool is not None:
        # Get mechanism from config (default: 'gaussian')
        mechanism = self.config.dp_mechanism 
        
        # Apply DP based on mechanism type
        self.dp_tool.apply_dp_to_gradients(
            model=local_model.net,
            optimizer=optimizer,
            mechanism=mechanism  # 'gaussian' or 'laplace'
        )
    
    return local_model.net
```
    return tensor + noise

# Privacy budget calculation
def compute_privacy_budget(self, num_rounds, num_clients):
    # Per-round privacy: ε_round = ε_total / T
    per_round_epsilon = self.epsilon / num_rounds
    
    # Noise scale: σ = √(2ln(1.25/δ)) × sensitivity / ε
    sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / per_round_epsilon
    
    # Effective noise due to client averaging: σ_effective = σ / √K
    effective_sigma = sigma / math.sqrt(num_clients)
    
    return per_round_epsilon, effective_sigma
```

#### Parameter Selection Guidelines

**Privacy Budget (ε) Selection:**
- **ε = 0.1**: Very strong privacy, significant utility loss
- **ε = 1.0**: Good balance between privacy and utility
- **ε = 5.0**: Weak privacy, minimal utility loss
- **ε = 10.0**: Very weak privacy, almost no protection

**Failure Probability (δ) Selection:**
- **δ = 1e-6**: Very conservative, suitable for small datasets
- **δ = 1e-5**: Standard choice, good for most applications
- **δ = 1e-4**: Less conservative, suitable for large datasets

**Sensitivity Selection:**
- **Sensitivity = 1.0**: Standard choice for normalized gradients
- **Sensitivity = 0.5**: More conservative, stronger privacy
- **Sensitivity = 2.0**: Less conservative, weaker privacy

**Noise Multiplier Selection:**
- **Multiplier = 0.5**: Less noise, weaker privacy
- **Multiplier = 1.0**: Standard choice
- **Multiplier = 2.0**: More noise, stronger privacy

## Project Structure

- `core/`: Core components including runner implementation and model definitions
  - `runner.py`: Main federated learning runner implementation
  - `config.py`: Configuration management
- `data/`: Data processing components:
  - `generator.py`: Simulated data generation
  - `loader.py`: Real-world data loading
  - `splitter.py`: Data partitioning utilities
- `utils/`: Helper functions and utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details. 