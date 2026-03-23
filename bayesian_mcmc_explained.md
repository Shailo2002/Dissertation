# Bayesian MCMC Inversion — Explained Simply

## The Goal

Your friend has MT (Magnetotelluric) data — measurements taken at the surface. They want to find out: **what layers are underneath?** — how many layers, how deep each one goes, and what resistivity each layer has.

The problem: **many different subsurface models can produce similar data**. So instead of finding ONE answer, they want to find ALL plausible answers and see how much they agree or disagree.

---

## The Analogy

Imagine you're blindfolded in a hilly landscape and you want to map out all the **valleys** (good-fitting models).

- **Deterministic inversion** (like standard gravity inversion): You roll a ball downhill. It finds ONE valley and stops. You get one answer, no idea if there are other valleys.

- **PSO** (your approach): You throw 50 balls at once. They communicate and mostly end up in the deepest valley. You get a good answer, and the spread of balls gives some idea of uncertainty.

- **MCMC** (your friend's approach): You walk around the landscape yourself, step by step. You tend to walk downhill (toward better models), but sometimes you deliberately walk uphill (toward worse models). Over thousands of steps, the places you visit most often ARE the valleys. Your **footprint map** = the posterior probability.

---

## How MCMC Works, Step by Step

### Step 0: Start Somewhere Random

Pick any random model. Say: 3 layers, with random depths and random resistivities. It probably fits the data terribly. That's fine.

```
Current model: [Layer 1: 0-500m, rho=100] [Layer 2: 500-2000m, rho=10] [Layer 3: 2000m+, rho=1000]
Misfit: 85 (terrible)
```

### Step 1: Propose a Small Change

Randomly pick ONE thing to change. The code picks from 5 options:

| Proposal Type | Probability | What it does |
|---|---|---|
| **Birth** | 10% | Add a new layer at a random depth |
| **Death** | 40% | Remove a random layer |
| **Move** | 16% | Shift a layer boundary up/down |
| **Change rho** | 34% | Tweak resistivity of a layer |
| **Change noise** | ~0% | Adjust data noise estimate |

Say it picks **Change rho** and proposes: Layer 2 resistivity changes from 10 to 15.

```
Proposed model: [Layer 1: 0-500m, rho=100] [Layer 2: 500-2000m, rho=15] [Layer 3: 2000m+, rho=1000]
```

### Step 2: Compute Forward Response

Run the proposed model through the **physics equations** (MT forward modeling). This gives you: "if the Earth looked like this, what MT data would I measure at the surface?"

### Step 3: Compare With Real Data

Compute the **likelihood** — how well does the predicted data match the observed data?

```
Current model misfit:  85
Proposed model misfit: 72  <-- better!
```

### Step 4: Accept or Reject (The Key Rule)

This is the **Metropolis criterion**:

- If the proposed model fits **better** --> **always accept** it
- If the proposed model fits **worse** --> **sometimes accept** it, with a probability that decreases the worse it is

**Why accept worse models?** Because this is what makes MCMC work. If you only accepted better models, you'd just be an optimizer (find one best answer and stop). By sometimes accepting worse models, the chain **wanders around** and explores different possibilities.

The math:

```
alpha = exp(-(misfit_proposed - misfit_current))

if alpha > random number between 0 and 1 --> ACCEPT
otherwise --> REJECT (stay where you are)
```

- If proposed is much better: alpha ~ 1 --> almost always accept
- If proposed is slightly worse: alpha ~ 0.7 --> accept 70% of the time
- If proposed is much worse: alpha ~ 0.01 --> accept only 1% of the time

---

## Step 5: Repeat 100,000 Times

Now just **repeat steps 1-4 again and again**, 100,000 times.

Think of it like a **dark room with furniture**. You want to make a map of where all the furniture is.

- You start at a random spot
- Each step, you take a small step in a random direction
- If you bump into furniture (bad fit), you mostly step back
- If the path is clear (good fit), you keep going

After walking for hours, look at your **footprints on the floor**:
- Lots of footprints in open areas (good models)
- Few footprints near furniture (bad models)
- The footprint density IS the map

**That's exactly what MCMC does.** Each "step" is one proposal + accept/reject. The collection of all visited models IS the answer.

---

## What "100 steps x 1000 samples" Actually Means

Think of it as simply **100,000 accepted models total** per chain. The two loops are just for bookkeeping:

```
Outer loop (100 steps) --> just for saving progress to disk
  Inner loop (1000 samples) --> collect 1000 accepted models
```

So after everything finishes, you have a **list** like:

```
Model #1:      3 layers, depths=[0, 500, 2000],        rho=[100, 10, 1000]         misfit=85
Model #2:      3 layers, depths=[0, 500, 2000],        rho=[100, 15, 1000]         misfit=72
Model #3:      3 layers, depths=[0, 520, 2000],        rho=[100, 15, 1000]         misfit=70
Model #4:      4 layers, depths=[0, 520, 1200, 2000],  rho=[100, 15, 50, 1000]     misfit=65
...
Model #100000: 5 layers, depths=[0, 480, 1100, 2500, 5000], rho=[95, 12, 55, 800, 1200]  misfit=3.2
```

Notice:
- Early models (#1, #2, #3...) have **high misfit** — still exploring randomly
- Later models have **low misfit** — found good regions
- Models change only slightly between consecutive steps (one small change at a time)
- Sometimes layers are added (birth) or removed (death), so the number of layers changes

---

## Processing: Why Throw Away the First Half? (Burn-in)

Look at the misfit values over time:

```
Model #1:      misfit = 85    <-- garbage, still random
Model #100:    misfit = 45    <-- getting better
Model #1000:   misfit = 12    <-- okay-ish
Model #5000:   misfit = 4     <-- now in a good region
...
Model #50000:  misfit = 3.1   <-- sampling the good region
Model #100000: misfit = 3.5   <-- still in the good region
```

The first ~50,000 models were just the chain **walking from its random start toward good-fitting regions**. This journey is called **burn-in**. These models don't represent the real answer — they're just the path to get there.

So you **throw them away**. Keep only model #50,001 to #100,000.

---

## Thinning: Why Keep Every 10th Model?

Model #50,001 and Model #50,002 differ by only ONE tiny change (remember, each step only tweaks one thing). They're almost identical. So are #50,002 and #50,003.

Keeping all of them is redundant — like asking 100 people the same question but 99 of them just copied from the person next to them.

So you **keep every 10th model**: #50,001, #50,011, #50,021, ...

Now you have ~5,000 models that are **reasonably independent** from each other.

---

## Building the Final Answer (The Posterior)

Now you have ~5,000 good, independent models. Each one is a complete description of the Earth (layers + resistivities).

**The question**: at depth = 1000m, what is the resistivity?

Just **look at all 5,000 models and check what they say at 1000m depth**:

```
Model #50001:  at 1000m, rho = 95
Model #50011:  at 1000m, rho = 102
Model #50021:  at 1000m, rho = 88
Model #50031:  at 1000m, rho = 110
Model #50041:  at 1000m, rho = 97
...
```

Now make a **histogram** of these 5,000 values:

```
rho = 50-70:    ##  (200 models)
rho = 70-90:    ########  (1500 models)
rho = 90-110:   ################  (2800 models)    <-- MOST LIKELY
rho = 110-130:  ###  (400 models)
rho = 130-150:  #  (100 models)
```

This histogram IS the **posterior probability at depth 1000m**. It tells you:
- **Best estimate**: rho ~ 100 (most models agree)
- **Uncertainty**: roughly 70 to 130 (the spread)
- **Confidence**: high (narrow histogram)

**Repeat this for EVERY depth** (0m, 2000m, 4000m, ..., 350,000m).

- At some depths the histogram will be **narrow** --> data constrains that depth well --> low uncertainty
- At other depths the histogram will be **wide** --> data doesn't constrain it --> high uncertainty

---

## The Final Plot (3 Panels)

### Panel 1: Posterior PDF (The Main Result)
- x-axis = resistivity
- y-axis = depth
- color = how many models agree (probability)
- Red lines = 5th and 95th percentile (90% confidence bounds)

Where the color is concentrated in a narrow band --> **we know the answer well**.
Where the color is spread out --> **we're uncertain**.

### Panel 2: Interface Probability
- At each depth, count how many of the 5,000 models place a layer boundary there
- A peak means: "most models agree there's a boundary here"
- This tells you where the distinct rock types change

### Panel 3: Number of Layers Histogram
- How many of the 5,000 models had 3 layers vs 4 vs 5 vs 6, etc.
- Tells you the most likely number of distinct subsurface units
- Example: if most models have 4-5 layers, you're confident the Earth has about that many distinct units

---

## The Transdimensional Part (Birth/Death)

This is what makes this code special. Normal MCMC fixes the number of layers (say 5) and only changes their properties. This code lets the **number of layers itself change**:

```
Step 100: model has 4 layers --> propose BIRTH --> now 5 layers --> accepted
Step 101: model has 5 layers --> propose DEATH --> now 4 layers --> rejected, stay at 5
Step 102: model has 5 layers --> propose BIRTH --> now 6 layers --> accepted
...
```

The algorithm figures out on its own how many layers are needed. If the data requires 5 layers to fit well, most accepted models will have ~5 layers.

---

## Parallel Tempering (Multiple Chains)

One chain can get "stuck" in a valley — it found a good region but can't climb over the hill to find other good regions.

Solution: run **multiple chains at different "temperatures"**:
- **Cold chain** (temperature = 1): strict, stays in good regions, gives the real answer
- **Hot chain** (temperature > 1): loose, accepts bad models more easily, explores freely

Periodically, hot and cold chains **swap their models**. This lets the cold chain "teleport" to new regions discovered by the hot chain.

In the code: `CData.temperature = [1 1]` means two cold chains (no tempering active right now).

---

## One-Line Summary

**Run 100,000 models. Throw away the bad early ones. Look at what the remaining good ones agree and disagree on. Agreement = answer. Disagreement = uncertainty.**

---

## How This Relates to Your Gravity PSO Work

| Aspect | Friend's MT Bayesian | Your Gravity PSO |
|---|---|---|
| Data type | MT impedance | Gravity anomaly |
| Unknown | Resistivity + depths + number of layers | Basement depth at each point |
| Dimensionality | Transdimensional (layers born/killed) | Fixed dimension |
| Method | MCMC (sequential random walk) | PSO (swarm optimization) |
| UQ | Built-in — posterior PDF from chain | Post-hoc — from swarm spread |
| Noise handling | Hierarchical — noise is also inverted | Typically fixed |

### Inspiration You Can Take

1. **Posterior visualization**: Create a 2D plot of basement depth vs position with color = probability from your PSO ensemble
2. **KL divergence**: Measure where your data actually constrains the solution vs where it doesn't
3. **Hierarchical noise**: Let the algorithm estimate the noise level too, instead of fixing it
