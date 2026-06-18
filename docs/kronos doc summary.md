# Kronos: A Foundation Model for the Language of Financial Markets[cite: 1]

**Authors:** Yu Shi$^{1,\dagger}$, Zongliang Fu$^{2,\dagger}$, Shuo Chen$^1$, Bohan Zhao$^1$, Wei Xu$^1$, Changshui Zhang$^2$, Jian Li$^1$[cite: 1]  
**Affiliations:** $^1$Institute for Interdisciplinary Information Sciences, $^2$Department of Automation, Tsinghua University[cite: 1]  
**Contact:** {shi-y23, fzl22, zhaobh23}@mails.tsinghua.edu.cn, ChenSh2003@outlook.com, weixu@tsinghua.edu.cn, zcs@mail.tsinghua.edu.cn, lapordge@gmail.com[cite: 1]  
**Links:** [https://github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)[cite: 1]  
*$\dagger$ Equal contribution*[cite: 1]

---

## Abstract[cite: 1]
The success of large-scale pre-training paradigm, exemplified by Large Language Models (LLMs), has inspired the development of Time Series Foundation Models (TSFMs)[cite: 1]. However, their application to financial candlestick (K-line) data remains limited, often underperforming non-pre-trained architectures[cite: 1]. Moreover, existing TSFMs often overlook crucial downstream tasks such as volatility prediction and synthetic data generation[cite: 1]. To address these limitations, we propose Kronos, a unified, scalable pre-training framework tailored to financial K-line modeling[cite: 1]. Kronos introduces a specialized tokenizer that discretizes continuous market information into token sequences, preserving both price dynamics and trade activity patterns[cite: 1]. We pre-train Kronos using an autoregressive objective on a massive, multi-market corpus of over 12 billion K-line records from 45 global exchanges, enabling it to learn nuanced temporal and cross-asset representations[cite: 1]. Kronos excels in a zero-shot setting across a diverse set of financial tasks[cite: 1]. On benchmark datasets, Kronos boosts price series forecasting RankIC by 93% over the leading TSFM and 87% over the best non-pre-trained baseline[cite: 1]. It also achieves a 9% lower MAE in volatility forecasting and a 22% improvement in generative fidelity for synthetic K-line sequences[cite: 1]. These results establish Kronos as a robust, versatile foundation model for end-to-end financial time series analysis[cite: 1]. Our pre-trained model is publicly available at https://github.com/shiyu-coder/Kronos[cite: 1].

---

## 1 Introduction[cite: 1]
The emergence of Foundation Models (FMs) has initiated a paradigm shift across artificial intelligence, reshaping the methodologies of representation learning and downstream task adaptation[cite: 1]. This shift is exemplified by the success of Large Language Models (LLMs) for natural language processing, with parallel breakthroughs in computer vision[cite: 1]. Inspired by these advances, the FM paradigm has recently been extended to temporal data, giving rise to Time Series Foundation Models (TSFMs)[cite: 1]. The central aim is to build pre-trained, task-agnostic architectures that serve as universal backbones for diverse time series analytical tasks—from forecasting and anomaly detection to causal inference—thereby substantially reducing the need for bespoke model design in each application domain[cite: 1].

Within this expanding research landscape, financial markets stand out as a critical and challenging application area for TSFMs, given their inherent data richness, high-frequency observations, and complex, non-stationary temporal dynamics[cite: 1]. At the core of this domain are K-line sequences, multivariate time series derived from candlestick charts that record Open, High, Low, and Close prices, along with trading Volume and Amount (Turnover) over fixed intervals (OHLCVA)[cite: 1]. These sequences constitute a highly compact, information-dense "language" through which market participants interpret price movements, volatility regimes, liquidity shifts, and collective sentiment[cite: 1]. Consequently, K-line data forms the bedrock of numerous algorithmic trading strategies, portfolio optimization schemes, and risk management systems[cite: 1].

However, applying general-purpose TSFMs to financial K-line data presents significant challenges, due to two principal factors[cite: 1]. First, K-line sequences exhibit unique statistical properties—such as low signal-to-noise ratios, strong non-stationarities, and intricate, high-order dependencies among OHLCVA attributes—that are often misaligned with the inductive biases of generic TSFMs[cite: 1]. Second, the financial domain has largely been underserved by mainstream TSFM research; financial sequences constitute a minor fraction of pre-training corpora for most existing TSFMs, and the spectrum of downstream tasks critical to quantitative finance—spanning volatility estimation, synthetic sequence generation, and risk management—remains largely unaddressed[cite: 1]. These factors lead to an important observation, which we empirically validate in this work: general-purpose TSFMs often underperform specialized, non-pre-trained models (e.g., iTransformer) on financial tasks and fail to generalize across the broader landscape of quantitative finance[cite: 1].

To address these shortcomings, we introduce Kronos, a unified, scalable pre-training framework designed specifically for financial K-line data[cite: 1]. Kronos employs a specialized tokenizer to discretize continuous, multivariate K-line inputs into a sequence of compact tokens, preserving critical price–volume interactions[cite: 1]. It then undergoes autoregressive pre-training on an expansive, heterogeneous corpus of over 12 billion K-line records drawn from over 45 global markets and 7 temporal granularities[cite: 1].

We validate the efficacy of Kronos through comprehensive experiments across a range of quantitative finance tasks, with a high-level summary presented in Figure 1[cite: 1]. On the core task of price series forecasting, Kronos establishes a new state-of-the-art, boosting the RankIC by 93% over the leading TSFM and by 87% over the best-performing non-pre-trained baseline[cite: 1]. Furthermore, it demonstrates strong versatility by achieving a 9% lower MAE in volatility forecasting and a 22% improvement in generative fidelity for synthetic K-line generation[cite: 1]. These findings highlight the broad effectiveness of our approach and underscore Kronos’s potential as a robust foundation model for interpreting the complex "language" of financial markets[cite: 1].

Our main contributions can be summarized as follows[cite: 1]:
* We propose a novel modeling framework for financial K-line data that learns hierarchical representations[cite: 1]. It features a specialized tokenizer that quantizes each multivariate K-line record into structured, dual-component (coarse and fine) tokens, coupled with a tailored autoregressive objective that predicts these subtokens sequentially[cite: 1]. This coarse-to-fine prediction scheme allows Kronos to explicitly model multi-scale market dynamics[cite: 1].
* We conduct large-scale pre-training for a family of Kronos models with varying capacities[cite: 1]. This is performed on a massive, diverse financial corpus of over 12 billion K-line records from over 45 global exchanges, which is fundamental to learning the robust and generalizable market representations that underpin the models’ effectiveness[cite: 1].
* We conduct comprehensive empirical evaluations across a set of quantitative finance tasks[cite: 1]. Our results show that Kronos establishes a new state-of-the-art in price series forecasting, significantly outperforming both TSFMs and specialized baselines[cite: 1]. The model’s versatility is further demonstrated by its strong performance across a broader spectrum of quantitative tasks, including volatility forecasting and synthetic K-line generation[cite: 1].

> **Figure 1: Comprehensive performance of Kronos across several quantitative finance tasks.**[cite: 1] 
> *Context & Meaning:* This figure displays a multi-axis radar chart showing Kronos's performance across Volatility Forecasting (MAE and $R^2$), Return Forecasting (IC and RankIC), Price Forecasting (IC and RankIC), Kline Generation (Disc. Score and IC), and Investment Simulation (AER and IR). The chart benchmarks Kronos (blue lines) against specialized baselines. A greater distance from the center signifies superior performance, visually confirming that Kronos dominates all metrics[cite: 1].

---

## 2 Preliminary[cite: 1]
Let $D$-dimensional vector $x_t \in \mathbb{R}^D$ denote the K-line observation at discrete time $t$, comprising $D$ key financial indicators[cite: 1]. In this work, we fix the dimension $D=6$ to represent OHLCVA attributes (Open, High, Low, Close prices, trading Volume, and Amount)[cite: 1]. The rationale for this input choice is detailed in Appendix H (Q1)[cite: 1]. Given a historical sequence $x_{1:T} = (x_1, x_2, ..., x_T)$, our objective is to predict the following $H$ observations $\hat{x}_{T+1:T+H} = (\hat{x}_{T+1}, \hat{x}_{T+2}, ..., \hat{x}_{T+H})$[cite: 1].

Rather than operating on raw continuous inputs, Kronos first quantizes each multivariate observation $x_t$ into a discrete token $b_t$ via a learnable codebook $\mathcal{C}$[cite: 1]. Consequently, the original sequence $x_{1:T} = (x_1, ..., x_T)$ is mapped to $b_{1:T} = (b_1, ..., b_T)$[cite: 1]. The forecasting task then reduces to an autoregressive token-sequence modeling problem[cite: 1]:

$$p(b_{T+1:T+H} | b_{1:T}) = \prod_{h=1}^H p(b_{T+h} | b_{1:T+h-1}) \quad (1)$$[cite: 1]

Such a discrete formulation is inherently scalable and naturally extends to other tasks that can be framed generatively, such as synthetic data generation and volatility forecasting[cite: 1].

---

## 3 Methodology[cite: 1]
Kronos abstracts financial K-line sequences as a discrete language and implements this via a two-phase framework illustrated in Figure 2: (1) K-line Tokenization and (2) Autoregressive Pre-training[cite: 1]. In the first phase, we design a specialized Transformer-based tokenizer to quantize a continuous, multivariate K-line sequence into a corresponding sequence of discrete tokens, via a learnable codebook[cite: 1]. Each K-line item (OHLCVA) is treated as an individual instance and quantized into a discrete token[cite: 1]. Each token is composed of a coarse-grained subtoken and a fine-grained subtoken[cite: 1]. This property is enforced via a hierarchical reconstruction loss, which explicitly compels the subtokens to model distinct levels of information, thereby creating a coarse-to-fine informational hierarchy[cite: 1]. In the second phase, an autoregressive decoder-only Transformer is pre-trained on these tokenized sequences, using the standard next-token prediction objective to sequentially forecast both subtoken levels at each future time step conditioned on the given historical context[cite: 1]. This unified discretize-and-generate paradigm enables Kronos to construct a high-fidelity, hierarchical representation of market dynamics, providing a robust foundation for downstream quantitative analysis[cite: 1].

> **Figure 2: The two-stage framework of Kronos.**[cite: 1]
> *Context & Meaning:* Illustrates (1) Instance-based K-line Tokenization where a Transformer autoencoder quantizes continuous K-lines into hierarchical discrete tokens (coarse + fine) via Binary Spherical Quantization (BSQ), and (2) Autoregressive Pre-training where a decoder-only causal Transformer models temporal dynamics by sequentially predicting the hierarchical subtokens conditioned on the past[cite: 1].

> **Figure 3: Architecture of the K-line Tokenizer.**[cite: 1]
> *Context & Meaning:* Details the tokenizer's internal flow. $T \times D$ inputs pass through a Causal Transformer, linear projections, and BSQ to create a bit code of length $k = k_c + k_f$. These split into coarse ($k_c$) and fine ($k_f$) subtokens[cite: 1].

### K-line Tokenization[cite: 1]
The first stage of Kronos transforms a continuous, $D$-dimensional K-line sequence $x = (x_1, ..., x_T)$, where $x_t \in \mathbb{R}^D$ encodes OHLCVA indicators, into a corresponding series of discrete tokens[cite: 1]. This is achieved using a Transformer-based autoencoder composed of an encoder $E_{enc}$, a quantizer $Q$, and a decoder $E_{dec}$[cite: 1]. Drawing inspiration from video quantization methods in generative modeling, we adapt Binary Spherical Quantization (BSQ), a variant of Look-up Free Quantization (LFQ), for this task[cite: 1]. BSQ quantizes a continuous latent vector $\xi_t$ into a $k$-bit binary code $b_t \in \{-1, 1\}^k$ by projecting it onto a set of learnable hyperplanes[cite: 1]. 

While a large number of bits $k$ (e.g., $k=20$) is desirable for capturing rich financial patterns, it results in an exponentially large vocabulary of size $2^k$, which introduces significant challenges for the subsequent autoregressive model[cite: 1]. To mitigate this, we factorize the $k$-bit code into $n$ subspaces. We set $n=2$[cite: 1]. We partition the code into a coarse subtoken $b_t^c$ and a fine subtoken $b_t^f$ of equal bit length, $k_c = k_f = k/2$, where $k = k_c + k_f$[cite: 1]. The resulting code $b_t$ is a concatenation of these two subtokens: $b_t = [b_t^c, b_t^f]$, with $b_t^c, b_t^f \in \{-1, 1\}^{k/2}$[cite: 1]. This decomposition transforms a single prediction over a large vocabulary of size $2^k$ into two sequential predictions over $2^{k/2}$ entries, substantially reducing computational and parameter complexity[cite: 1].

To enforce a coarse-to-fine structure within each token, we train the tokenizer with a composite objective[cite: 1]:

$$\mathcal{L}_{tokenizer} = \mathcal{L}_{coarse} + \mathcal{L}_{fine} + \lambda\mathcal{L}_{quant} \quad (2)$$[cite: 1]

where $\lambda$ is a balancing hyperparameter[cite: 1]. The components are defined as[cite: 1]:
* $\mathcal{L}_{coarse} = \mathbb{E}[||x - E_{dec}(b^c)||^2]$, which trains the coarse subtoken $b^c$ to form a low-fidelity reconstruction[cite: 1].
* $\mathcal{L}_{fine} = \mathbb{E}[||x - E_{dec}(b)||^2]$, which evaluates the high-fidelity reconstruction using the complete token $b$[cite: 1].
* $\mathcal{L}_{quant}$ is the quantization loss from BSQ that regularizes the learning process by penalizing the L2 distance between continuous latent vectors $\xi$ and their binary codes $b$[cite: 1].

This hierarchical reconstruction objective explicitly imposes a coarse-to-fine hierarchy into the tokens during quantization[cite: 1].

### Hierarchical Autoregressive Modeling[cite: 1]
Following the tokenization stage, the resulting discrete sequences are modeled using a decoder-only Transformer, denoted as $E_{ar}$, which employs causal-attention[cite: 1]. The primary objective is to estimate the joint distribution over the token sequence $b = \{b_1, ..., b_T\}$[cite: 1]:

$$p(b) = \prod_{t=1}^T p(b_t | b_{<t}) \quad (3)$$[cite: 1]

Given the hierarchical token design, we further decompose the conditional probability using the chain rule[cite: 1]:

$$p(b_t | b_{<t}) = p(b_t^c | b_{<t}) \cdot p(b_t^f | b_{<t}, b_t^c) \quad (4)$$[cite: 1]

The sequence of fused inputs is concatenated and linearly projected to produce a fused input vector[cite: 1]:

$$v_i = W_{fuse}([e_c(b_i^c); e_f(b_i^f)]) \quad (5)$$[cite: 1]

**Coarse Subtoken Prediction.** The history vector $h_t$ is projected by a linear head $W_c$ to produce logits for the first subtoken’s distribution[cite: 1]:

$$p(b_t^c | b_{<t}) = \text{softmax}(W_c h_t) \quad (6)$$[cite: 1]

**Fine Subtoken Prediction.** The context is updated with the predicted coarse subtoken using a cross-attention mechanism[cite: 1]:

$$h_t^{update} = \text{CrossAttn}(q= e_c(\hat{b}_t^c), k= v= h_t)$$[cite: 1]
$$p(b_t^f | b_{<t}, b_t^c) = \text{softmax}(W_f h_t^{update}) \quad (7)$$[cite: 1]

The overall training objective $\mathcal{L}_{ar}$ is the negative log-likelihood of the data[cite: 1]:

$$\mathcal{L}_{ar} = -\mathbb{E}_{b\sim\mathcal{D}} \sum_{t=1}^T [\log p(b_t^c | b_{<t}) + \log p(b_t^f | b_{<t}, b_t^c)] \quad (8)$$[cite: 1]

### Model Pre-training[cite: 1]
* **Dataset:** We curate a large-scale, high-quality financial K-line dataset spanning over 12 billion observations across 7 sampling frequencies from 45 global exchanges[cite: 1].
* **Model Training:** We trained three variants of Kronos with increasing parameter counts, up to nearly 0.5 billion. The configurations are presented in Table 1[cite: 1].
* **Inference:** At inference time, we generate future token sequences autoregressively. Stochasticity is controlled via temperature scaling and top-p sampling[cite: 1].

**Table 1: Model configurations for the Kronos family**[cite: 1]

| Model | Layers | $d_{model}$ | $d_{ff}$ | Heads | Vocab. ($2^k$) | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Kronos$_{small}$ | 8 | 512 | 1024 | 8 | 20 | 24.7M |
| Kronos$_{base}$ | 12 | 832 | 2048 | 16 | 20 | 102.3M |
| Kronos$_{large}$ | 18 | 1664 | 3072 | 32 | 20 | 499.2M |

---

## 4 Experiments[cite: 1]

To comprehensively evaluate the capabilities of Kronos as a foundation model for financial K-line data, we design a suite of experiments spanning 5 representative tasks: Price series forecasting, return forecasting, realized volatility forecasting, synthetic K-line generation, and investment simulation[cite: 1]. We benchmark Kronos against a comprehensive suite of 25 baseline models spanning four paradigms (non-pre-trained full-shot, zero-shot TSFMs, econometric, and generative)[cite: 1].

> **Figure 4: Main experimental results across five representative financial tasks.**[cite: 1]
> *Context & Meaning:* Features Bar and Scatter plots. (a-c) show Kronos achieving superior forecasting performance on price series, returns, and volatility over TimeXer, iTransformer, etc. (d) Shows generative model performance, plotting Discriminative Score against IC/RankIC; Kronos achieves the highest fidelity. (e) Presents backtesting results where Kronos achieves the highest Annualized Excess Return (AER) and Information Ratio (IR) against all competitors[cite: 1].

### Main Results[cite: 1]
* **Prediction Tasks:** Kronos achieves consistent state-of-the-art performance across all of them[cite: 1]. For price series forecasting, Kronos achieves a remarkable 93% improvement in RankIC compared to the strongest TSFM baseline, and an 87% gain over the best non-pre-trained model[cite: 1]. Performance scales up with model size, validating scaling laws[cite: 1].
* **Generative Tasks:** Evaluated on diversity, fidelity (Discriminative Score), and usefulness (Train-on-Synthetic, Test-on-Real). Kronos achieves the best performance in both fidelity and usefulness[cite: 1]. 
* **Investment Simulation:** Simulating a long-only investment strategy on Chinese A-shares, Kronos outperforms all other baselines, achieving the highest Annualized Excess Return (AER) and Information Ratio (IR)[cite: 1].

> **Figure 5: Visual comparison of generative models on the dataset of Shanghai Stock Exchange, 15-minute frequency.**[cite: 1]
> *Context & Meaning:* Displays t-SNE embeddings and Kernel Density Estimates (KDE) comparing original data (red) to synthetic data (blue) from Kronos, DiffusionTS, TimeVAE, and TimeGAN. Kronos's synthetic data visually maps and overlaps perfectly with the real data distribution compared to the fragmented outputs of the baseline models[cite: 1].

### Ablation Study[cite: 1]
We compare Kronos against variants that differ in their prediction spaces (continuous vs. discrete) and objectives (Table 2)[cite: 1]. 

**Table 2: Ablation study dissecting the architectural choices of Kronos**[cite: 1]

| Model | Prediction Space | Training Objective | Price IC ($\uparrow$) | Price RankIC ($\uparrow$) | Return IC ($\uparrow$) | Return RankIC ($\uparrow$) | Volatility $R^2$ ($\uparrow$) | Volatility MAE ($\downarrow$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Direct-AR | Continuous | MSE | 0.0149 | 0.0212 | 0.0179 | 0.0102 | 0.0416 | 0.1608 |
| Prob-AR | Continuous | NLL | 0.0399 | 0.0329 | 0.0356 | 0.0565 | 0.1383 | 0.0464 |
| Kronos-Parallel | Discrete | Cross-Entropy | 0.0345 | 0.0226 | 0.0431 | 0.0254 | 0.0529 | 0.0461 |
| **Kronos$_{small}$** | **Discrete** | **Cross-Entropy** | **0.0505** | **0.0665** | **0.0622** | **0.1784** | **0.0384** | **0.2490** |

*Analysis of Paradigms:* Our discrete-space models markedly outperform continuous alternatives[cite: 1]. Kronos-Parallel performs worse than our sequential approach, demonstrating the importance of modeling subtoken dependencies[cite: 1].

> **Figure 6: Impact of vocabulary size on model performance.**[cite: 1]
> *Context & Meaning:* Plots showing that as vocabulary size ($2^k$) increases from $2^{14}$ to $2^{20}$, reconstruction error (MAE, MSE) decreases, and forecasting capabilities (IC, RankIC) increase[cite: 1].

### Test-Time Scaling[cite: 1]
By leveraging stochastic sampling, Kronos can generate multiple distinct future trajectories. Averaging across multiple paths (increasing the number of samples, N) mitigates the stochasticity inherent in the generation process and reduces prediction variance, yielding a more robust and stable estimate[cite: 1].

> **Figure 7: Impact of the number of inference samples (N) on forecasting performance.**[cite: 1]
> *Context & Meaning:* A line graph showing that performance heavily increases as the number of inference samples scales from 1 to 20, solidifying the test-time scaling advantages of Kronos[cite: 1].

---

## 5 Conclusion[cite: 1]
In this work, we introduce Kronos, a foundation model specifically designed for financial K-line sequences[cite: 1]. Kronos employs a novel two-stage framework, where an instance-based tokenizer first discretizes continuous market data into hierarchical coarse-to-fine tokens, which are then modeled by a large autoregressive Transformer[cite: 1]. Comprehensive empirical evaluations demonstrate that Kronos establishes new state-of-the-art benchmarks in price series forecasting, as well as in other relevant applications such as synthetic K-line generation and volatility forecasting, significantly outperforming existing TSFMs and other baselines[cite: 1]. These results position Kronos as a robust and versatile foundation for a range of applications in quantitative finance[cite: 1].

---

## Appendix A: Related Work[cite: 1]
Discusses Time Series Tokenization (Chronos, TOTEM, VQ-VAE, LFQ, BSQ, IBQ) and General-Purpose Time Series Foundation Models (Lag-Llama, TimesFM, Timer, Time-MoE, Sundial, MOMENT, Moirai, TimeGPT, UniTS)[cite: 1]. Table 3 details that most models dedicate less than 1% of their pre-training data to the financial domain, whereas Kronos is 100% focused on financial K-lines[cite: 1].

## Appendix B: Dataset Details[cite: 1]
Our dataset spans over 12 billion observations across 7 sampling frequencies from 45 global exchanges[cite: 1]. 

**Algorithm 1: Low-Quality Segment Filtering Pipeline**[cite: 1]
1. `PartitionByPriceJumps` splits the sequence by structural breaks (e.g., price jump threshold > 0.10)[cite: 1].
2. `FlagConsecutiveIlliquid` screens for sustained illiquidity[cite: 1].
3. `FlagConsecutiveStagnant` screens for extended price stagnation[cite: 1].
4. Subsequences meeting the minimum length criteria are retained for the final pre-training dataset[cite: 1].

## Appendix C: Implementation Details[cite: 1]
* **Input Preprocessing:** Inputs undergo z-score normalization and are clipped to the range [-5, 5][cite: 1].
* **Temporal Embeddings:** Five time-related features are extracted (minute, hour, day-of-week, day-of-month, month-of-year)[cite: 1].
* **Transformer Architecture:** Employs causal self-attention with Rotary Position Embeddings (RoPE) and Pre-LN (RMSNorm)[cite: 1]:
  $$\text{Attention}(Q, K, V) = \text{CausalMask} \left( \frac{Q'(K')^T}{\sqrt{d_k}} \right) V \quad (9)$$[cite: 1]

## Appendix D: Experimental Design and Implementation[cite: 1]
For full-shot baseline models, we employ a composite loss function combining MSE with an Information Coefficient (IC) term:
$$\mathcal{L} = \frac{1}{M \cdot H} \sum_{i=1}^M \sum_{j=1}^H (y_{i,j} - \hat{y}_{i,j})^2 - \lambda \cdot \frac{1}{M} \sum_{i=1}^M \text{IC}(y_i, \hat{y}_i) \quad (10)$$[cite: 1]

Predicted Return Formula:
$$\hat{r} = \frac{\hat{p}_{t+H}}{p_t} - 1 \quad (11)$$[cite: 1]

Realized Volatility Formula:
$$\hat{\sigma}^2 = \sum_{i=1}^{H-1} (\log(\hat{p}_{i+1}) - \log(\hat{p}_i))^2 \quad (12)$$[cite: 1]

Investment Simulation Expected Return Formula:
$$R_{t \rightarrow t+H} = \frac{(\frac{1}{H}\sum_{i=1}^H \hat{p}_{t+i}) - p_t}{p_t} \quad (13)$$[cite: 1]

Prob-AR Student-t Distribution Formula:
$$p(x|\nu,\mu,\sigma) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\pi\nu}\sigma} \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}} \quad (14)$$[cite: 1]

## Appendix E: Additional Results[cite: 1]
> **Figure 8: Sensitivity analysis of Kronos's performance to sampling.**[cite: 1]
> *Context & Meaning:* Forecasting tasks generally benefit from lower temperatures ($T \approx 0.6$) to reduce randomness. Generative tasks and volatility forecasting benefit from higher temperatures ($T \approx 1.0$) to increase diversity[cite: 1].

> **Figure 9: Cumulative return curves of backtest.**[cite: 1]
> *Context & Meaning:* Kronos models generate steep, consistently upward-trending cumulative returns on the CSI300 and CSI800 indices, outperforming baseline models significantly[cite: 1].

## Appendix F & G: Full Experiment Results & Forecast Showcases[cite: 1]
Extensive tables (Tables 14-21) demonstrate the statistical superiority of Kronos across XSHG, XNAS, XJPX, XNSE, XKRX, XHKG, Crypto, and Forex markets[cite: 1]. 

> **Figures 10-19: Visual Forecast Showcases.**[cite: 1]
> *Context & Meaning:* A massive array of time-series plots depicting Kronos predicting actual market data (e.g., NVIDIA, BTC/USDT, BMW). The red predicted lines closely map to the intricate dynamics and shapes of the blue ground-truth lines across both price and volume variations[cite: 1].

## Appendix H: Discussion[cite: 1]
**Q1: K-line Information:** Empirical evidence demonstrates K-line data encapsulates the informational content of short-term driving factors[cite: 1]. 
**Q2: Tokenizer Effectiveness:** BSQ inherently suppresses noise and maps infinite states into a finite, discrete vocabulary, acting as a powerful regularizer[cite: 1]. 
**Q3: Subtoken Factorization:** Factorizing the $k=20$ bit token into $n=2$ subtokens represents the optimal trade-off, reducing vocabulary parameters by over 99.8% (from 1.7B to 3.4M) without introducing massive latency penalties[cite: 1].

> **Figure 12: Visualization of token usage patterns.**[cite: 1]
> *Context & Meaning:* Illustrates how different token categories map to common vs. rare K-bar shapes, indicating the learned codebook successfully captures a semantic hierarchy of market dynamics[cite: 1].