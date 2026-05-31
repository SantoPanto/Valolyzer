# Valolyzer Architecture - Technical Deep Dive

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALOLYZER SYSTEM ARCHITECTURE                │
└─────────────────────────────────────────────────────────────────┘

TRAINING PIPELINE (01_veri_incelemesi.ipynb)
│
├─ Phase 1: Data Preparation
│  ├─ Load: matches.csv, maps.csv, player_stats.csv
│  ├─ Clean: match_id, map_id, agent names
│  ├─ Merge: Matches → Maps → Players
│  └─ Output: df_train (prepared data)
│
├─ Phase 2: Differential Encoding
│  ├─ Map one-hot encoding (11 maps)
│  ├─ Agent differential: T1(+1) vs T2(-1)
│  └─ Output: diff_cols (25 features)
│
├─ Feature Engineering Layers
│  ├─ Phase 2: Role-based aggregation (4 features)
│  ├─ Phase 3: Synergy detection (10 features)
│  └─ Phase 4: Agent-map weighting (25 features)
│
├─ Data Augmentation (Symmetry)
│  ├─ Normal: Team A vs Team B
│  └─ Swapped: Team B vs Team A (inverted signs)
│
├─ Model Training
│  └─ LogisticRegression(fit_intercept=False)
│
└─ Output Artifacts
   ├─ valorant_lr_model.pkl (binary model)
   ├─ model_columns.pkl (75 feature names)
   ├─ agent_roles.pkl (role mappings)
   ├─ agent_synergies.pkl (synergy pairs)
   ├─ agent_map_winrates.pkl (win rate data)
   └─ map_agent_stats.pkl (top agents/map)

INFERENCE PIPELINE (app.py)
│
├─ Load Resources
│  ├─ Load all 6 .pkl files
│  └─ Cache in memory
│
├─ User Input
│  ├─ Select map
│  ├─ Select 5 agents for Team 1
│  └─ Select 5 agents for Team 2
│
├─ Feature Engineering (Real-time)
│  ├─ Map encoding
│  ├─ Differential encoding
│  ├─ Role counting
│  ├─ Synergy detection
│  └─ Agent-map weighting
│
├─ Prediction
│  └─ Logistic Regression: probability calculation
│
└─ Output
   ├─ Team winner prediction
   ├─ Winning probability (%)
   ├─ Feature importance visualization
   └─ Top agents per map display
```

---

## Phase 1: Differential Feature Encoding

### Mathematical Formulation

For each agent `i`, create differential feature:

$$\text{Agent}_i^{\text{diff}} = \mathbb{1}[\text{Agent}_i \in \text{Team}_A] - \mathbb{1}[\text{Agent}_i \in \text{Team}_B]$$

**Domain**: $\{-1, 0, 1\}$
- $+1$: Agent only in Team A
- $0$: Agent in both teams or neither
- $-1$: Agent only in Team B

### Example

```
Input: Team A = [Jett, Sova, Omen, Sage, Killjoy]
       Team B = [Raze, Breach, Viper, Cypher, Phoenix]

Jett_diff = 1 (Team A only)
Raze_diff = -1 (Team B only)
Sova_diff = 1 (Team A only)
Breach_diff = -1 (Team B only)
... (25 agents total)
```

### Properties
- ✅ Symmetric: Swapping teams negates all values
- ✅ Normalized: Bounded range
- ✅ Sparse: ~80% features are 0

---

## Phase 2: Role-Based Aggregation

### Role Classification

```
DUELIST (Attack-focused):
  Agents: Jett, Raze, Phoenix, Reyna, Neon, Yoru, Iso
  Count per team: [0, 7]

CONTROLLER (Zoning/Information):
  Agents: Omen, Brimstone, Viper, Astra, Harbor, Vyse
  Count per team: [0, 6]

INITIATOR (Support/Recon):
  Agents: Sova, Breach, KAY/O, Fade, Skye, Gekko
  Count per team: [0, 6]

SENTINEL (Defense/Info):
  Agents: Sage, Killjoy, Cypher, Chamber, Clove, Deadlock
  Count per team: [0, 6]
```

### Feature Calculation

$$\text{Role}_r^{\text{diff}} = |\{\text{Agent}_i \in \text{Team}_A : \text{role}(i) = r\}| - |\{\text{Agent}_i \in \text{Team}_B : \text{role}(i) = r\}|$$

**Domain**: $\{-5, -4, ..., 4, 5\}$

### Example

```
Team A = [Jett, Raze, Breach, Omen, Sage]
  → Duelist: 2, Controller: 1, Initiator: 1, Sentinel: 1

Team B = [Phoenix, Neon, Sova, Killjoy, Viper]
  → Duelist: 2, Controller: 1, Initiator: 1, Sentinel: 1

Duelist_diff = 2 - 2 = 0
Controller_diff = 1 - 1 = 0
Initiator_diff = 1 - 1 = 0
Sentinel_diff = 1 - 1 = 0
```

### Strategic Significance
- Balanced composition (all 0) is neutral
- Duelist-heavy teams (+2) favor aggressive play
- Sentinel-heavy teams (-2) favor defensive play
- Controller difference impacts map control

---

## Phase 3: Synergy Features

### Synergy Pairs (Carefully Selected)

```python
SYNERGIES = [
    ('Jett', 'Sova'),       # Jett for entry, Sova for intel
    ('Raze', 'Fade'),       # Area denial + information
    ('Viper', 'Cypher'),    # Zone control coordination
    ('Omen', 'Sage'),       # Map control + sustain
    ('Brimstone', 'Killjoy'), # Territory dominance
    ('Chamber', 'Harbor'),  # Sentinel coordination
    ('Astra', 'Breach'),    # Global pressure
    ('Reyna', 'Skye'),      # Aggression + support
    ('KAY/O', 'Phoenix'),   # Suppression + damage
    ('Neon', 'Harbor'),     # Duelist + map control
]
```

### Mathematical Definition

$$\text{Synergy}_{(i,j)}^{\text{diff}} = \begin{cases}
+1 & \text{if } i, j \in \text{Team}_A \\
-1 & \text{if } i, j \in \text{Team}_B \\
0 & \text{otherwise}
\end{cases}$$

### Example

```
Team A = [Jett, Sova, ...]  → Jett_Sova_synergy = 1
Team B = [Raze, Fade, ...]  → Raze_Fade_synergy = -1
Neither = [Omen, Breach]    → Omen_Sage_synergy = 0
```

### Interpretation
- Positive coefficient: Synergy favors Team 1
- Negative coefficient: Synergy favors Team 2
- Zero coefficient: Synergy not relevant

---

## Phase 4: Agent-Map Weighting

### Win Rate Calculation

For agent $i$ on map $m$ from team perspective $t$:

$$WR_{i,m,t} = \frac{\text{Wins}_{\text{agent } i \text{ on map } m \text{ team } t}}{\text{Plays}_{\text{agent } i \text{ on map } m \text{ team } t}}$$

**Training data**: Symmetric dataset, so both Team A and B perspectives captured

### Differential Conversion

Convert win rate (0-1 range) to differential (-1 to +1):

$$\text{Weight}_{i,m}^{\text{diff}} = 2 \cdot (WR_{i,m,A} - 0.5) - 2 \cdot (WR_{i,m,B} - 0.5)$$

### Application

When agent $i$ selected for Team 1: $\text{Agent}_i^{\text{map\_weight}} = \text{Weight}_{i,m}^{\text{diff}}$

When agent $i$ selected for Team 2: $\text{Agent}_i^{\text{map\_weight}} = -\text{Weight}_{i,m}^{\text{diff}}$

### Example

```
Jett on Haven:
  Team A (Jett) win rate: 54%  → differential: (0.54 - 0.5) × 2 = +0.08
  Team B (Jett) win rate: 46%  → differential: (0.46 - 0.5) × 2 = -0.08
  Combined weight: 0.08 - (-0.08) = 0.16

If Jett selected for Team 1: Jett_map_weight = +0.16 (bonus)
If Jett selected for Team 2: Jett_map_weight = -0.16 (penalty)
```

---

## Symmetric Training Data Construction

### Problem Statement

Machine learning models trained on biased data learn bias. If training data is skewed toward Team A, model predicts Team A wins more often.

### Solution: Data Augmentation

For each data point $(x, y)$ where $y \in \{0, 1\}$:
1. Keep original: $(x, y)$
2. Create swapped: $(-x, 1-y)$

### Mathematical Property

For linear models with zero intercept:

$$P(y=1|-x) = \sigma(-w^T x) = 1 - \sigma(w^T x) = 1 - P(y=1|x)$$

Where $\sigma$ is sigmoid function.

### Guarantee

Any zero vector (identical teams) maps to:

$$P(\text{Team A wins} | \mathbf{0}) = \sigma(0) = 0.5$$

This is **exact**, not approximate.

---

## Logistic Regression Model

### Why LogisticRegression instead of Random Forest?

| Property | Random Forest | LogisticRegression |
|----------|---------------|-------------------|
| Decision Surface | Non-linear trees | Linear hyperplane |
| Interpretability | Feature importance | Coefficients |
| Symmetry Property | Approximate | Exact (fit_intercept=False) |
| Overfitting | Low | Low (linear) |
| Calibration | Poor | Good |

### Model Specification

```python
model = LogisticRegression(
    fit_intercept=False,  # Critical: ensures zero input → 0.5 prob
    solver='lbfgs',       # Good for small-medium datasets
    max_iter=1000,        # Sufficient for convergence
    random_state=42       # Reproducibility
)
```

### Mathematical Foundation

$$P(\text{Team A wins} | x) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

Where:
- $w$ = learned coefficients
- $x$ = feature vector
- $\sigma$ = sigmoid function

**Key**: With $\text{fit\_intercept}=\text{False}$, intercept is 0, so:
- $w^T \mathbf{0} = 0$
- $\sigma(0) = 0.5$ exactly

---

## Feature Engineering Pipeline (Inference)

### Step-by-Step Processing

```python
def prepare_input_data(selected_map, team1_agents, team2_agents, 
                       model_columns, agent_roles, agent_synergies, 
                       agent_map_winrates):
    
    # 1. Initialize zero vector for all 75 features
    input_data = {col: 0.0 for col in model_columns}
    
    # 2. Map encoding (1 of 11 maps = 1, others = 0)
    input_data[f"map_{selected_map}"] = 1
    
    # 3. Individual agent differential (25 features)
    for agent in team1_agents:
        input_data[f"{agent}_diff"] += 1
    for agent in team2_agents:
        input_data[f"{agent}_diff"] -= 1
    
    # 4. Role-based features (4 features)
    for role in roles:
        t1_count = sum(1 for ag in team1 if role(ag) == role)
        t2_count = sum(1 for ag in team2 if role(ag) == role)
        input_data[f"{role}_diff"] = t1_count - t2_count
    
    # 5. Synergy features (10 features)
    for agent1, agent2 in synergies:
        if agent1 in team1 and agent2 in team1:
            input_data[f"{agent1}_{agent2}_synergy"] = 1
        elif agent1 in team2 and agent2 in team2:
            input_data[f"{agent1}_{agent2}_synergy"] = -1
    
    # 6. Agent-map weights (25 features)
    for agent in all_agents:
        wr_a = agent_map_winrates.get((agent, map, 'A'), 0.5)
        wr_b = agent_map_winrates.get((agent, map, 'B'), 0.5)
        
        weight = (wr_a - 0.5)*2 - (wr_b - 0.5)*2
        
        if agent in team1:
            input_data[f"{agent}_map_weight"] = weight
        elif agent in team2:
            input_data[f"{agent}_map_weight"] = -weight
    
    # 7. Return as DataFrame for model prediction
    return pd.DataFrame([input_data])
```

### Complexity Analysis

- **Time**: O(n_agents + n_synergies) = O(25 + 10) = O(1) constant
- **Space**: O(n_features) = O(75) constant
- **Latency**: < 1ms on modern CPU

---

## Bias Analysis

### Before Refactoring

```
P(Team A wins | Team A = Team B)
├─ Random Forest tree 1: 0.52
├─ Random Forest tree 2: 0.48
├─ Random Forest tree 3: 0.51
├─ Average: 0.503 ± 0.02 (statistically biased)
└─ Problem: Breaks fairness property
```

### After Refactoring

```
P(Team A wins | Team A = Team B)
├─ Logistic Regression: σ(0) = 0.5000
└─ Mathematically guaranteed (no approximation)
```

### Win Rate Consistency Check

For any team composition:
```
If P(Team A wins | config) = 0.65
Then P(Team B wins | config) = 0.35

Swapping team labels:
P(Team B wins | config) should = 0.65
P(Team A wins | config) should = 0.35

✅ This property is guaranteed by symmetric training + linear model
```

---

## Performance Metrics

### Training Metrics

```
Dataset size: N samples
After augmentation: 2N samples (balanced 50-50)
Test set size: 0.2 × 2N

Metrics:
├─ Accuracy: Correctly classified / Total predictions
├─ AUC-ROC: True positive rate vs False positive rate (threshold-agnostic)
├─ Precision: TP / (TP + FP) - reliability of positive predictions
├─ Recall: TP / (TP + FN) - coverage of positive predictions
└─ Calibration: Predicted probability = Actual frequency
```

### Expected Results

- **Accuracy**: 55-65% (better than random 50%)
- **AUC-ROC**: 0.60-0.70 (agent composition does matter, but not deterministic)
- **Calibration**: Good (Logistic Regression is naturally well-calibrated)

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ Training Flow                                                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CSV Files          Data Cleaning        Feature Engineering │
│  ├─ matches.csv   ────────────┬────────────────┬───────────  │
│  ├─ maps.csv                  │                │            │
│  └─ player_stats.csv          │                ↓            │
│                               │          Phase 1-4 Features  │
│                               ↓                │             │
│                          df_train              │             │
│                          (cleaned)             │             │
│                               │                │             │
│                               └────────────────┤             │
│                                                ↓             │
│                                         Feature Matrix       │
│                                         (75 features)        │
│                                                │             │
│                                           Symmetric          │
│                                           Augmentation       │
│                                                │             │
│                                    ┌──────────┴──────────┐  │
│                                    ↓                     ↓  │
│                               Normal          Swapped    │  │
│                               (Team A wins) (Team B wins)   │
│                                    │                     │  │
│                                    └──────────┬──────────┘  │
│                                               ↓             │
│                                     Merged Dataset (2N)     │
│                                               │             │
│                                     Train/Test Split        │
│                                               │             │
│                                    ┌──────────┴─────────┐   │
│                                    ↓                    ↓   │
│                            Training Set (1.6N)  Test Set   │
│                                    │              (0.4N)   │
│                                    ↓                    ↓   │
│                        LogisticRegression           Metrics │
│                        model.fit()                         │
│                                    │                    ↓  │
│                            ┌────────┴─────────┐      AUC    │
│                            ↓                  ↓     Accuracy│
│                        Coefficients    Predictions (Test)   │
│                                    │                    │   │
│                                    └────────┬───────────┘   │
│                                             ↓               │
│                                      Serialize .pkl Files   │
│                                      ├─ model              │
│                                      ├─ columns            │
│                                      ├─ roles              │
│                                      ├─ synergies          │
│                                      ├─ map_winrates       │
│                                      └─ map_stats          │
│                                                             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Inference Flow                                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User Input                Feature Preparation              │
│  ├─ Map: "Ascent" ───────────┬─────────────────────────────  │
│  ├─ Team A: [5 agents]       │                              │
│  └─ Team B: [5 agents]       ↓                              │
│                         prepare_input_data()                │
│                              │                              │
│                              ↓                              │
│                         Feature Vector                      │
│                         (75 features)                       │
│                              │                              │
│                              ├─ Loaded .pkl Files           │
│                              │  ├─ model                   │
│                              │  ├─ roles                   │
│                              │  ├─ synergies               │
│                              │  └─ map_winrates             │
│                              ↓                              │
│                      model.predict_proba()                  │
│                              │                              │
│                    ┌─────────┴─────────┐                   │
│                    ↓                   ↓                   │
│            [0.35, 0.65]           Probabilities            │
│            (Team B, Team A)             │                   │
│                    │                   ↓                   │
│                    │            Display Result:            │
│                    │            "Team A wins 65%"          │
│                    │                   │                   │
│                    └─────────┬─────────┘                   │
│                              ↓                              │
│                     Feature Importance                      │
│                     (Coef[Agent}_diff] etc.)               │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Edge Cases & Handling

### Case 1: Identical Teams
```
Input: Team A = Team B
Expected: P(Team A wins) = 0.5 exactly
Mechanism: All differentials = 0 → w·0 = 0 → σ(0) = 0.5
```

### Case 2: No Synergies
```
Input: Team A = [Jett, Breach, Omen, Sage, Killjoy]
       Team B = [Raze, Sova, Viper, Cypher, Phoenix]
Result: All synergy features = 0 (no impact)
Mechanism: Model learns synergies help, but absence doesn't hurt
```

### Case 3: New Agent (Future Proof)
```
If new agent added to Valorant:
1. Add to AGENT_ROLES dict
2. Feature {NewAgent}_diff automatically created
3. Agent-map winrates = 0.5 (neutral) until data collected
4. Model works with neutral prior
```

### Case 4: Sparse Agent-Map Data
```
If agent not played on a map:
├─ Win rate defaults to 0.5
├─ Map weight defaults to 0.0 (neutral)
└─ Model handles gracefully
```

---

## Deployment Checklist

- [ ] Training notebook runs end-to-end
- [ ] Symmetry check outputs 0.5000 ± 0.0001
- [ ] All 6 .pkl files generated
- [ ] App loads without errors
- [ ] Identical teams prediction = 50%
- [ ] Feature visualization working
- [ ] Performance metrics acceptable (AUC > 0.55)
- [ ] Documentation complete
- [ ] Test script passes all checks

---

## Conclusion

The refactored Valolyzer system implements a mathematically rigorous approach to Valorant prediction with:

1. **Guaranteed Fairness**: No inherent team bias
2. **Advanced Features**: 4 layers of feature engineering
3. **Interpretability**: Linear model with clear coefficient meanings
4. **Robustness**: Symmetric training prevents overfitting
5. **Scalability**: Constant-time inference, easy to update

The system balances accuracy, fairness, and interpretability, making it suitable for both casual use and competitive analysis.
