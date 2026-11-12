# DPO vs. GRPO vs. PPO: A Comparative Guide

The three methods—**DPO (Direct Preference Optimization)**, **GRPO (Group Relative Policy Optimization)**, and **PPO (Proximal Policy Optimization)**—are all ways to improve a language model's behavior beyond ordinary supervised fine-tuning (SFT). However, they approach the problem quite differently.

The most important distinction is:

- **DPO learns from preferences that have already been collected**. It does not need to generate new responses during the optimization phase or calculate an explicit reward for each response.
- **GRPO generates responses online and uses a reward function to compare multiple responses to the same prompt**. It avoids a separate learned value model by using group statistics as a baseline.
- **PPO generates responses online and uses a reward function plus a learned value model to estimate how good the generated responses are**. It uses that estimate to calculate advantages and update the policy.

Let's compare them systematically, then connect their mathematical objectives to the reinforcement-learning goal of maximizing cumulative reward.

## 1. At-a-glance comparison

```text
Dimension               DPO                   GRPO                         PPO
-----------------       ------------------    ---------                    ---------
Optimization type       Offline preference    Online RL                    Online RL
                        optimization

Training signal         Chosen vs. rejected   Scalar reward per            Scalar reward per generated 
                        responses             generated response           response

Explicit reward         No                    No, if using a hand-         No, if using a hand-designed
model required?                               designed reward; optional    reward; optional

Value/critic model      No                    No                           Yes, typically

Multiple responses      Not required          Yes, typically a group       Not required
per prompt

Generates responses     No                    Yes                          Yes
during training?

Reference policy        Typically yes         Usually yes when KL          Usually yes when KL
                                              regularization is used       regularization is used

Main optimization       Increase relative     Increase probability of      Increase probability of
idea                    likelihood of         above-average responses      responses with positive
                        preferred responses   within a group               estimated advantage

Main challenge          Quality of            Reward quality and           Reward quality, value estimation,
                        preference pairs      group-relative signal        and training stability

Typical complexity      Lowest                Medium                       Highest
```




A terminology clarification: DPO is commonly grouped with preference-based post-training and sometimes loosely called an RL method. Mathematically, it derives from a particular RL objective, but its training procedure is not an online RL loop like PPO or GRPO.

## 2. Models involved

A useful way to understand the model architecture is to distinguish five roles. Not every algorithm needs all five.
```
Model/component     Purpose                             DPO         GRPO        PPO
---------------     --------------------------------    ---         ----        ---
Base/SFT model      Starting point for optimization;    Yes         Yes         Yes
                    becomes the policy

Policy model	    Model being trained to produce      Yes         Yes         Yes
                    better responses

Reference model	    Frozen copy used to constrain       Typically   Typically   Typically
                    policy changes

Reward model/	    Evaluates the quality of generated	Not needed  Yes         Yes
reward function     responses                           during DPO
                                                        optimization

Value model/	    Estimates expected future reward	No          No          Yes
critic              from a state

Dataset-generation  Generates candidate responses for	Often       Not         Not
model               constructing training data                      necessarily necessarily
```
Let's examine the roles in more detail.

### 2.1 Policy model

This is the model we want to improve.

In our experiments:

- R1 is the SFT-trained GPT-2 model.
- R2, R3, and R4 each start from R1.
- Each algorithm produces a separately optimized policy.

The policy is the model that generates text and whose parameters are updated.

### 2.2 Reference model

The reference model is a frozen copy of the initial policy, commonly the SFT model.

Its purpose is to prevent optimization from moving too far away from the model's original language-generation behavior.

For example, we can penalize the policy when its generated token distribution diverges too much from the reference distribution using KL divergence:

```math
D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})
```

This is particularly important when the reward function is imperfect. Without a constraint, the policy may discover ways to obtain high reward while producing unnatural or undesirable language.

In our experiments, R1 serves as the reference model for R2, R3, and R4.

### 2.3 Reward model or reward function

This evaluates how good a generated response is.

There are two broad possibilities:

- Learned reward model: A neural network trained to score responses, often from human preference labels.
- Hand-designed reward function: A program that calculates a score using rules, heuristics, external evaluators, or other models.

For our experiments, E1 is a learned sentiment classifier. We use its probability for the requested sentiment as the reward.

For example:

$$
r(x,y,\text{positive}) =
P_{\mathrm{E1}}(\text{positive}\mid x+y)
$$

Here, $x$ is the review prefix and $y$ is the generated continuation.

Importantly, the reward model is not necessarily the model being optimized. It is an evaluator that supplies a training signal to the policy.

### 2.4 Value model or critic

This is specific to PPO in our comparison.

The value model estimates the expected future reward from a particular state:

```math
V_\phi(s_t) \approx
\mathbb{E}\left[
\sum_{k=t}^{T}\gamma^{k-t}r_k
\mid s_t
\right]
```

For an LLM, a state $s_t$ is essentially the prompt together with the tokens generated so far.

The value model helps answer:

"Given what the model has generated so far, how much reward should it expect to receive by the end of this response?"

It is used to calculate the advantage of an action—whether generating a particular token was better or worse than expected.

Our R4 implementation has a GPT-2 backbone with an additional scalar value head. The value head predicts a value for each token position.

### 2.5 Dataset-generation model

This is an optional but important component, especially for offline preference optimization.

For DPO, we need preference pairs:

```math
(x,y_{\mathrm{chosen}},y_{\mathrm{rejected}})
```

A model can generate multiple candidate responses for each prompt. A human, reward model, or another evaluator can then identify the preferred response.

In our R2 experiment:

1. R1 generates four candidate continuations for each prompt.
2. E1 scores the candidates.
3. We select a chosen and rejected response.
4. We train R2 using those pairs.

The candidate-generation model and the policy being trained do not have to be the same model. In our experiment, R1 generates the candidates, while R2 is the policy optimized by DPO.

For GRPO and PPO, the policy itself normally generates new responses during training. Therefore, a separate dataset-generation model is not required.

## 3. What is used as the reward or training signal?

This is one of the most important differences.

**DPO: preference pairs**

DPO does not directly consume a numerical reward during its optimization loop.

Instead, it consumes pairs:

- A chosen response: preferred.
- A rejected response: less preferred.

For example:

```
Prompt                                                  Chosen                      Rejected
Continue with positive sentiment: "The movie is..."     "wonderful and engaging"    "utterly disappointing"
```

The preference pair tells the model which response should be more likely relative to the other.

The preference can originate from:

- Human annotations.
- A reward model.
- Rules or heuristics.
- Another model acting as a judge.

In our experiment, E1 creates the preference signal indirectly by scoring R1's candidate continuations.

**Key point**: E1 is used to construct the R2 training dataset, but it is not called during DPO training itself.

**GRPO: reward for each generated response**

GRPO generates multiple responses for the same prompt and calculates a reward for each.

For example, suppose the policy generates four responses:
```
Response    E1 positive-sentiment reward
A           0.90
B           0.75
C           0.40
D           0.25
```

GRPO uses the relative reward of each response within this group to calculate an advantage.

A response with a reward above the group average receives a positive advantage; one below the average receives a negative advantage.

In our R3 experiment, E1 provides these scalar rewards.

**PPO: reward for each generated response**

PPO also generates responses and calculates rewards.

The reward can come from the same sources as GRPO:

- A learned reward model.
- A hand-designed reward function.
- A combination of multiple reward components.

In our R4 experiment, E1 provides a scalar reward for the generated continuation, based on the requested sentiment.

The main difference between PPO and GRPO is not the reward itself. It is how they estimate the advantage used to update the policy.

## 4. Policy optimization and learning objectives

Now let's look at the central mathematical differences.

### 4.1 DPO: increase the relative probability of preferred responses

DPO optimizes the policy so that a chosen response becomes more likely than its rejected counterpart.

A common DPO loss is:

```math
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-\mathbb{E}_{(x,y_w,y_l)}
\left[
\log \sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}
{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}
{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right)
\right]
```

Where:

- $y_w$: chosen response.
- $y_l$: rejected response.
- $\pi_\theta$: trainable policy.
- $\pi_{\mathrm{ref}}$: frozen reference policy.
- $\sigma$: sigmoid function.
- $\beta$: controls the strength of the reference-policy constraint.

Let's unpack the formula.

The first log-ratio measures how much the policy has changed the likelihood of the chosen response relative to the reference model.

The second measures the corresponding change for the rejected response.

DPO wants the first quantity to be greater than the second.

In simpler terms:

    Increase the relative likelihood of the chosen response while decreasing the relative likelihood of the rejected response.

The loss is a binary logistic loss over the preference margin.

**DPO optimization intuition**
```
Chosen response         Increase relative probability
Rejected response       Decrease relative probability
```

The objective is to increase the chosen-versus-rejected margin, measured relative to the reference policy.

### 4.2 GRPO: increase the probability of above-average responses

GRPO's central idea is to generate a group of responses for each prompt and use their relative rewards to estimate advantages. A convenient way to understand it is that GRPO samples/generates multiple preference pairs in an online fashion while optimizing the policy in the same way as PPO, except that it gets rid of the critic used in PPO for the purpose of advantage calculation.

For a group of $G$ responses, calculate:

```math
A_i = 
\frac{r_i-\text{mean}(r)}
{\text{std}(r)+\epsilon}
```


Here:

- $r_i$ is the reward for response $i$.
- The mean and standard deviation are calculated across the group.
- $A_i$ is the normalized relative advantage.

For our example:
```
Response    Reward  Relative advantage (approx.)
A           0.90    +1.16
B           0.75    +0.62
C           0.40    -0.62
D           0.25    -1.16
```

The policy should increase the probability of responses A and B and decrease the probability of C and D.

A simplified GRPO objective is:

```math
\mathcal{L}_{\mathrm{GRPO}}
=
-\mathbb{E}
\left[
\min
\left(
\rho_{i,t}A_i,
\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
\right]
+
\beta D_{\mathrm{KL}}
```

Where:

```math
\rho_{i,t} =
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,\text{<}t})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,\text{<}t})}
```

This is a simplified representation of the clipped GRPO objective. Actual implementations can differ in how they aggregate token-level objectives, apply KL penalties, and normalize rewards.

The important point is that GRPO uses a group-relative advantage rather than a learned value function.

### 4.3 PPO: increase the probability of actions with positive advantage

PPO also uses a clipped policy objective, but it obtains advantages using a value model.

The core objective is:

```math
\mathcal{L}_{\mathrm{PPO}}^{\mathrm{policy}}
=
-\mathbb{E}_t
\left[
\min
\left(
\rho_t A_t,
\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
```

Where:

- $\rho_t$ is the ratio between the new policy's probability and the old policy's probability for the sampled token.
- $A_t$ is the estimated advantage.
- $\epsilon$ controls how much the probability ratio can change in one update.

The value model is trained separately using a value prediction loss, commonly a squared-error loss:

```math
\mathcal{L}_{\mathrm{value}}
=
\mathbb{E}_t
\left[
(V_\phi(s_t)-R_t)^2
\right]
```

Here, $R_t$ is a return or return target, often estimated using Generalized Advantage Estimation (GAE).

The overall PPO loss also commonly includes an entropy term to encourage exploration:

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{policy}}
+c_v\mathcal{L}_{\mathrm{value}}
-c_e\mathcal{H}(\pi_\theta)
```

A KL penalty against a reference policy may be incorporated into the reward or objective as an additional constraint.

### 4.4 The key distinction

How each algorithm obtains its training signal

**DPO**\
Chosen response vs. rejected response\
Preference margin

**GRPO**\
Multiple responses + their rewards\
Group-relative advantage

**PPO**\
Generated response + reward + value estimates\
GAE advantage

All three ultimately use gradient-based optimization to update the policy parameters. Their differences lie primarily in how they construct the training signal and constrain policy updates.

## 5. How do we turn maximizing long-term cumulative reward into a loss function?

This is the most conceptually important part of the comparison.

In reinforcement learning, the general objective is to find a policy that maximizes expected cumulative reward:

```math
J(\theta)=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\gamma^t r_t
\right]
```

Where:

- $\pi_\theta$ is the policy.
- $\tau$ is a trajectory of actions and states.
- $r_t$ is the reward at time $t$.
- $\gamma$ discounts future rewards.

For an LLM:

- State: prompt plus the tokens generated so far.
- Action: generate the next token.
- Trajectory: the entire generated response.
- Reward: the quality score assigned to the response, potentially including intermediate token-level rewards.

The goal is to make the policy generate responses that receive higher expected cumulative reward.

But how do we do that with gradient descent?

## 5.1 The policy-gradient intuition

Suppose our model generates a response consisting of five tokens:

```
The movie is absolutely wonderful.
```

The reward model evaluates the completed response and gives it a score of 0.9.

We want the policy to become more likely to generate this kind of response in the future.

The policy-gradient theorem gives us a way to estimate the gradient of expected reward:

```math
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta \log\pi_\theta(a_t|s_t)
Q^\pi(s_t,a_t)
\right]
```

The important term is:

```math
\nabla_\theta\log\pi_\theta(a_t|s_t)
```

It tells us how to change the model parameters to increase the probability of an action.

If the action leads to a good outcome, we increase its probability. If it leads to a bad outcome, we decrease its probability.

This is the foundation of PPO and GRPO.

### 5.2 Why do we need an advantage rather than just a reward?

Suppose a prompt produces four responses, and all four receive high rewards.

Simply increasing the probability of all four responses may not be very informative. We want to know which responses are better than the alternatives.

This motivates the concept of advantage:

```math
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
```

- $Q(s_t,a_t)$: expected return after taking action $a_t$.
- $V(s_t)$: expected return from the current state, regardless of which action is taken.

The advantage tells us whether an action is better or worse than the baseline expectation.

This reduces the variance of policy-gradient estimates and gives a more useful training signal.

### 5.3 How PPO estimates the advantage: GAE

PPO uses a value model to estimate the value of each state.

A common method is Generalized Advantage Estimation (GAE):

```math
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
```

```math
A_t^{\mathrm{GAE}}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}
```

Here:

- $\delta_t$ is the temporal-difference error.
- $\gamma$ controls reward discounting.
- $\lambda$ controls the bias-variance trade-off.

Intuitively:

1. The value model predicts how much reward we expect.
2. The actual reward and the next state's estimated value tell us whether that prediction was too optimistic or pessimistic.
3. GAE combines these errors over future time steps.
4. The resulting advantage tells us which actions deserve increased or decreased probability.

For our R4 implementation, the sentiment reward is primarily assigned at the end of the generated continuation. We also distribute a KL penalty over the response tokens. GAE then propagates the terminal reward information backward across the generated tokens.

This is how a reward that evaluates the entire response can influence the probability of individual tokens.

### 5.4 How GRPO estimates the advantage without a value model

GRPO uses a different baseline.

Instead of training a value model to estimate expected future reward, it generates multiple responses for the same prompt and calculates the group mean reward.

```math
A_i =
\frac{r_i-\bar r}{\sigma_r+\epsilon}
```

The group mean acts as a baseline.

This is a particularly clever idea for LLMs: multiple candidate completions can be generated for the same prompt, and their rewards provide a relative measure of quality.

For example, if four responses receive rewards of 0.9, 0.7, 0.4, and 0.2, the model receives a clear signal about which responses performed better than the group.

The trade-off is that GRPO's advantage estimate depends on the quality and diversity of the generated group.

### 5.5 How DPO gets around explicit reward estimation

DPO takes a different route.

It starts from a KL-regularized reward-maximization objective:

```math
\max_\pi
\mathbb{E}_{y\sim\pi(\cdot|x)}
[r(x,y)]
-
\beta D_{\mathrm{KL}}
\left(
\pi(\cdot|x)\Vert\pi_{\mathrm{ref}}(\cdot|x)
\right)
```

Under this objective, the optimal policy has a relationship to the reward:

```math
r(x,y)
=
\beta\log
\frac{\pi^*(y|x)}
{\pi_{\mathrm{ref}}(y|x)}
+
\beta\log Z(x)
```

where $Z(x)$ is a normalization term dependent on the prompt.

DPO substitutes this relationship into a preference-modeling objective, commonly based on the Bradley-Terry preference model.

The result is the DPO loss we saw earlier.

In other words:

    DPO derives an implicit reward from the relationship between the optimized policy and the reference policy. It learns from preference comparisons without explicitly training or querying a reward model during optimization.

This is why DPO can optimize preferences without running a traditional online RL loop.

### 5.6 A useful mental model

```
Algorithm   How it answers "Was this response good?"
---------   ----------------------------------------
DPO         Was it preferred over another response?
GRPO        Did it score better than the other responses generated for the same prompt?
PPO         Did it perform better than the value model predicted?
```

This is the conceptual heart of the three algorithms.

## 6. Training loop / pipeline

Let's compare the actual workflows.

### 6.1 DPO pipeline
```
                    SFT model
        Initial policy and reference model
                      ↓
        Generate candidate responses
    Often using the SFT model or another generator
                      ↓
            Preference annotation
Human, reward model, or heuristic selects chosen and rejected responses
                      ↓
            Preference dataset
    Prompt, chosen response, rejected response
                      ↓
                DPO training
Policy and reference log probabilities → DPO loss → gradient update
                      ↓
                Optimized Policy
```


The important characteristic is that the preference dataset is prepared before optimization. During DPO training, the policy does not need to generate new responses or query a reward model (thus, it is called "offline" in RL, i.e., where an agent not interacting with the environment, as a result of which, there is no new experience to learn from).

The reference model is used to calculate log probabilities, but it is not updated.

### 6.2 GRPO pipeline

```
                Prompt dataset
Prompts only; no chosen/rejected pairs required
                      ↓
      Policy generates G responses per prompt
                      ↓
      Reward function evaluates each response
                      ↓
        Calculate group-relative advantages
                      ↓
            GRPO loss + KL constraint
    Compare new and old policy probabilities
                      ⟳
            Update policy and repeat
```

The key is that the policy generates new training examples as training proceeds. This is called online learning or online RL.

### 6.3 PPO pipeline
```
                Prompt dataset
                      ↓
        Policy generates responses
                      ↓
    Reward function scores responses
                      ↓
   Value model estimates state values
                      ↓
  Calculate returns and GAE advantages
                      ↓
  PPO clipped policy loss + value loss
                      ↓
    Update policy and value model
                      ⟳
Generate a fresh batch of rollouts and repeat
```

The PPO pipeline is more involved because it has to maintain a value model and calculate advantages.

A typical PPO training iteration involves collecting rollouts, computing rewards and advantages, then performing several minibatch optimization epochs on the collected data.

**An important distinction**: PPO's multiple optimization epochs over the same rollout batch are not the same as generating new rollouts. The policy is updated several times using that batch, then fresh responses are generated for the next iteration.

## 7. Data used in training
```
Data dimension	    DPO	                    GRPO                PPO
--------------      ----------------        -------             -------
Main training	    Preference pairs	    Prompts             Prompts
data

Required labels     Chosen/rejected         No explicit	        No explicit
                                            response labels     response labels

Training            Pre-generated           Generated online    Generated online
responses           and fixed

Reward scores       Not needed              Needed              Needed
                    during optimization

Validation data     Held-out preference	    Held-out prompts	Held-out prompts
                    pairs

Validation          Evaluate preference     Generate responses	Generate responses
procedure           objective and/or        and evaluate        and evaluate
                    preference accuracy     rewards             rewards

```

Some important qualifications:

**DPO**: The dataset is fixed during a training run, although it can be regenerated or expanded between runs. The policy may eventually move far from the model that generated the preference pairs, creating an offline-distribution problem.

**GRPO**: The prompt dataset can be relatively small because the policy generates new responses. However, this does not mean the prompts are unimportant. The model cannot learn to solve tasks that are not represented in its prompt distribution.

**PPO**: Similar to GRPO, it uses prompts to collect online rollouts. The rollout data is typically temporary and is reused for a limited number of PPO optimization epochs.

For all three methods, train/validation separation matters. For example, in our SST-2 experiments, splitting the original dataset before creating review prefixes helps prevent leakage between training and validation.

## 8. Metrics

The metrics should be divided into three categories: training/optimization metrics, reward-related metrics, and language-quality/task-specific metrics.

### 8.1 Training/Optimization metrics — is the network learning healthily?

This category materializes the RL objective into neural network optimization. The value of this category is that it can detect problems early—for example, a sustained high clip ratio or a rapidly dropping entropy both indicate that training is about to go wrong, even before the reward curve reflects it. The risk is that the absolute value of the loss is often not intuitive; it needs to be judged in conjunction with reward and qualitative analysis.

```
Metric                  DPO     GRPO        PPO         What it tells us
-------------           ---     ----        ---         ------------------------------------
Training loss           Yes     Yes         Yes         Whether the optimization
                                                        objective is changing

Validation loss         Yes     Sometimes   Sometimes   Whether the training
                                                        objective generalizes

Preference accuracy     Yes     No          No          Whether chosen responses
                                                        are preferred over rejected responses

KL divergence       Common      Common      Common      How far policy moves from reference

Policy entropy      Possible    Common      Common      Diversity / uncertainty
                                                        of policy

Value loss              No      No          Yes         How well the critic 
                                                        estimates returns

Approximate KL to       No      Common      Common      How far a policy update  
old policy                                              moves from its previous 
                                                        version
                                                        

Clip fraction           No      Common      Common      How often probability 
                                                        ratios hit clipping limits
```

A few distinctions are worth emphasizing:

- DPO preference accuracy is not the same as sentiment accuracy.
- KL divergence is a measure of policy divergence, not a direct quality metric.
- Value loss is a measure of critic prediction error, not policy quality.
- Training loss alone cannot establish that the model is getting better at the target task.

### 8.2 Reward-related metrics — is the goal being achieved?

This category directly reflects the optimization objective of RL. The core is mean reward (whether it trends upward), along with reward distribution (whether it is polarized or has collapsed), etc. Note though, GRPO/PPO reward is not necessarily a measure of language quality.

```
Metric                  DPO     GRPO        PPO         What it tells us
-------------           ---     ----        ---         ------------------------------------
Reward mean             Not     Yes         Yes         Average reward of     
                    intrinsic                           generated responses
                    

Reward standard         Not     Yes         Yes         Variation in reward
deviation           intrinsic
```

### 8.3 Qualitative/Task-specific metrics — is what is learned actually what we want?

This category requires human review, or at least sampling review, and is the most easily neglected. It includes completeness (whether the continuation has jumps or omissions), naturalness (whether the language is fluent and whether there is repetition or templating), reward hacking signals (excessive length, stacking keywords, exploiting evaluation loopholes), diversity (whether the output style is singular), and factual correctness (whether the content is actually correct). The value of this category is that it can capture problems that numerical metrics cannot—for example, the reward may be rising while the model learns to "look like it is positive" rather than actually writing something meaningful. The cost is that it requires human resources and cannot be fully automated.


For our sentiment-conditioned generation experiment, we would use:
```
Metric                          Purpose
----------------------------    ----------------------------------------------------------
Target sentiment probability    Average E1 probability assigned to the requested sentiment

Sentiment match rate            Percentage of responses whose E1-predicted sentiment matches the requested sentiment

Average response length         Detects overly short or long responses

Distinct-1 / Distinct-2         Measures lexical diversity

Repetition rate                 Detects repetitive or degenerate generation

Human fluency score             Evaluates naturalness

Human relevance score           Evaluates whether the continuation fits the prefix

Human sentiment score           Evaluates whether the intended sentiment is expressed appropriately
```

### 8.4 The relationship between the three

These three categories form a chain of "goal → means → result": reward metrics tell you whether the direction is correct, optimization metrics tell you whether the process is healthy, and qualitative metrics tell you whether the result is credible. A typical failure pattern is: reward rises, loss drops, but qualitative review reveals that the model has learned reward hacking—at this point, the first two categories of metrics look "healthy," and only the third category can expose the problem. Conversely, if optimization metrics show abnormal KL or entropy, even if the reward has not yet deteriorated, it is a warning that qualitative degradation is about to occur.

Therefore, the three categories are not parallel, they have a progressive relationship of "early warning → confirmation → final judgment." Optimization metrics are the earliest warning, reward metrics are process confirmation, and qualitative metrics are the final judgment. A complete monitoring system should cover all three, and establish a workflow for regularly sampling high-reward samples for manual review.

For a proper comparison, R0–R4 should use the same held-out prompts, the same decoding parameters, and the same number of generated responses per prompt.

E1 should be treated as one evaluation instrument, not the final authority. Since E1 also supplies the reward for R3 and R4, evaluating exclusively with E1 risks mistaking reward-model exploitation for genuine improvement.

## 9. Stability

Stability is influenced by the algorithm, but also by the reward function, dataset quality, learning rate, batch size, model size, and training configuration.
```
Aspect              DPO                         GRPO                            PPO
--------------      --------------------------  -----------------------------   ------------------------------------------
Main source of      Noisy or biased preference	Noisy rewards; group sampling	Noisy rewards; inaccurate value estimates; 
instability         pairs; excessive policy     variance; weak reward           policy/value interaction
                    movement                    differences

Critic-related      None                        None                            Yes
instability

Sensitivity to      Indirect, through           Direct                          Direct
reward model        preference construction

Reference-policy    Common                      Common                          Common
constraint

Update constraint   Preference objective and    Clipped policy objective and    Clipped policy objective and often KL 
                    reference regularization    often KL regularization         regularization

Training complexity Relatively low              Medium                          High
```

**DPO**

DPO is often relatively straightforward to train because it uses a fixed dataset and a supervised-learning-style loss.

However, it is not automatically stable.

For example:

- Incorrect preference labels can teach undesirable behavior.
- A poorly constructed preference dataset can contain weak or contradictory signals.
- Excessive training can overfit the preference pairs.
- The policy can drift away from the reference model.

**GRPO**

GRPO avoids the need to train a value model, eliminating one important source of instability.

However, it has its own challenges:

- If all responses receive nearly identical rewards, the advantage signal becomes weak.
- If the generated group is too small, relative comparisons can be noisy.
- If the reward function is unreliable, the policy may optimize the wrong behavior.
- If generation produces very similar responses, group comparisons become less informative.

The group-relative baseline is especially useful when there is no convenient way to train a separate critic.

**PPO**

PPO has a more complicated training system because it must optimize both the policy and the value model.

The value model can be inaccurate, causing poor advantage estimates. Policy updates can then amplify those errors.

PPO uses clipping and often KL regularization to limit policy movement, which helps stability, but does not eliminate the problem.

Its additional complexity can be justified when value estimation provides useful information about how rewards should be assigned across a sequence of actions.

## 10. Convergence speed and computational cost

There is no universal ranking of convergence speed. It depends heavily on the task, model, reward function, data, and implementation.

It is useful to distinguish three different concepts:

1. Optimization speed: How quickly the training loss or objective improves.
2. Sample efficiency: How many training examples or generated responses are needed to achieve a given result.
3. Wall-clock efficiency: How much actual computation and time are required.

```
Dimension               DPO                     GRPO                            PPO
-------------           ---------------------   ----------------                ----------------
Training data           Potentially expensive   Relatively light                Relatively light
preparation

Online generation   	No                      Yes                             Yes
during training

Extra model forward     Policy + reference      Policy + reward +reference      Policy + reward +
passes                                                                          reference + value

Value model training    No                      No                              Yes

Training                Relatively simple       Medium                          Complex
implementation

Potential sample        Good with high-quality	Depends on group quality        Depends on rollout and critic quality
efficiency              preference pairs        and reward signal

Wall-clock cost         Often lower             Potentially high due to         Potentially high due to generation and critic 
                                                multiple generations            computation
                                                per prompt
```

A few practical observations:

**DPO** can be computationally efficient after its preference dataset has been prepared. However, generating and evaluating enough preference pairs can itself be expensive.

**GRPO** spends computation generating multiple responses per prompt. If the group size is four, it generates four responses to obtain one group-relative training signal.

**PPO** also spends substantial computation on generation and evaluation, and additionally trains a value model. However, the value model may make better use of information about expected future rewards.

For an LLM, generation is often a major computational bottleneck. Therefore, the number of tokens generated and the number of candidate responses per prompt can matter more than the nominal number of optimizer steps.

## 11. How our R2, R3, and R4 experiments fit into this picture

Let's connect the theory directly to what we implemented.

Common starting point
**R1 — SFT GPT-2**
The initial policy and reference model for all three experiments.

**R2 — DPO**
Offline, 1178 training / 215 validation preference pairs
R1 generated candidates; E1 selected preference pairs; DPO optimized the policy using chosen/rejected responses.


**R3 — GRPO**
Online, 1000 training / 100 validation prompts
R1 generated four responses per prompt; E1 provided rewards; GRPO calculated group-relative advantages without a value model.

**R4 — PPO**
Online, 1000 training / 100 validation prompts
R1 generated responses; E1 provided rewards; the trainable value head estimated values; PPO used GAE and clipped policy updates.


**Our observed results**
```
Experiment  Relevant result                         Interpretation
----------  ------------------------------          --------------------------------
R2 DPO      Validation preference accuracy:         Improved the preference objective
            64.04%                                  over training

R3 GRPO     Validation reward: 0.6304               Stable training, but no convincing
                                                    upward reward trend

R4 PPO      Validation reward: 0.6274               Successful PPO implementation, but no
                                                    demonstrated reward improvement over R3
```

These results are not a direct algorithmic competition yet. The metrics and evaluation procedures were not fully controlled across the three experiments.

In particular:

- R2's preference accuracy is not comparable to R3/R4's reward.
- R3 generated groups of four responses, while R4 generated one response per prompt in its evaluation.
- The E1 reward function is an imperfect proxy for human judgment.

Our experiments demonstrate that we successfully implemented three different post-training approaches. They do not yet establish that one algorithm is superior.

## 12. Final summary: which algorithm should we use?
```
Situation                                       Natural     Why
                                                choice
------------------------------------            -------     -----------------------------------------------------------------
We have high-quality chosen/rejected        	DPO         Directly learns from preferences without online reward evaluation
response pairs

We have a reward function but do not want to    GRPO        Group comparisons provide a baseline
train a critic

We have a reward function and can afford a      PPO         Learns a value estimate and uses it to calculate token-level 
value model                                                 advantages

We have human preference data but no reward     DPO         Preference pairs are sufficient
model

We can generate many candidates and score       GRPO        Can exploit relative comparisons between candidates
them

We need a more traditional actor-critic RL      PPO         Explicit policy and value-function optimization
approach
```

One final conceptual point: **DPO, GRPO, and PPO are not three progressively more advanced versions of the same algorithm.** They are different approaches to obtaining a useful learning signal for policy optimization. DPO avoids online RL; PPO explicitly learns a critic; GRPO replaces a learned critic with group-relative reward statistics.

Understanding that distinction will help us recognize the same design choices in newer post-training methods, even when their names and implementations differ.