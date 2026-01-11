# Abstract Glossary

## Neural Architectures
* **LSTM RNN (Long Short-Term Memory):** A recurrent network with "memory gates" that captures long-term dependencies, solving the vanishing gradient problem of standard RNNs.
* **DNN (Feed Forward):** Standard deep networks without memory loops. The baseline this paper aims to outperform.

## Modeling Concepts
* **Acoustic Model:** The component that calculates the probability of a phoneme given an audio input.
* **HMM (Hidden Markov Model):** Statistical model used to handle the time evolution and transitions between phonemes.
* **CD (Context Dependent):** Modeling phonemes based on their neighbors (e.g., *triphones*) rather than in isolation.
* **CTC (Connectionist Temporal Classification):** A training technique that allows the network to learn sequences without needing precise frame-by-frame alignment labels.

## Optimization
* **Sequence Training:** Optimizing the model to minimize errors on the entire sentence sequence, rather than just individual frames.
* **Frame Stacking:** Concatenating multiple consecutive audio frames into a single input to give the network more immediate context.
* **Reduced Frame Rate:** Processing audio at longer time intervals (skipping frames) to significantly speed up decoding.


# Plan ideas
## Step 1: Data & Feature Preparation (the "Input")

The paper uses **Log-Mel Filterbank energy features**.

**Objective:** Load a small audio dataset (e.g., a short version of LibriSpeech or SpeechCommands to start simple).

**Actions:**

* Load the audio.
* Transform the waveform into a spectrogram (80 dimensions as in the paper).
* Visualize what the network "sees".

---

## Step 2: Implementing "Frame Stacking" (the "Fast")

This is the paper’s specific technique to speed up the model.

**Objective:** Code the function that stacks frames and reduces the frame rate.

**Actions:**

* **Input:** A sequence of 100 frames (10 ms).
* **Processing:** Stack groups of 3 or 8 frames + sub-sampling (skip frames).
* **Output:** A shorter but richer sequence.

---

## Step 3: LSTM Architecture (the "Accurate")

Neural network construction using PyTorch.

**Objective:** Create the acoustic model.

**Actions:**

* **Layer 1:** Input Layer (size = 80 × stacking factor).
* **Layer 2:** LSTM (with or without a linear projection layer for comparison).
* **Layer 3:** Fully Connected layer (projection to character classes).

---

## Step 4: CTC Loss & Training

This is where temporal alignment is handled without complex HMMs.

**Objective:** Connect the LSTM output to the CTC loss function.

**Actions:**

* Prepare labels (Text → Integers).
* Configure `nn.CTCLoss`.
* Run a training loop for a few epochs.

---

## Step 5: Decoding & Extension (the Result)

**Objective:** Check whether the model "writes" what it hears.

**Actions:**

* **Greedy Decoder:** Select the maximum probability at each time step.
* **Extension (your idea):** Compare a model *with* frame stacking vs *without* frame stacking to evaluate speed gains and accuracy differences.




# Plan ideas

### 1. Introduction et contexte (≈ 1 min)

- Qu’est-ce que la reconnaissance automatique de la parole (ASR)
- Rôle du **modèle acoustique** dans un système ASR
- Limites des approches classiques :
  - GMM-HMM
  - DNN frame-based
- Motivation pour l’utilisation des **RNN / LSTM** :
  - la parole est un signal **séquentiel et temporel**

**Message clé :**  
> La reconnaissance vocale est un problème séquentiel → les RNN sont naturellement adaptés.

---

### 2. RNN et LSTM pour la reconnaissance vocale (≈ 1.5 min)

- Rappel rapide sur :
  - les RNN
  - les LSTM et le problème du vanishing gradient
- Différence entre :
  - RNN unidirectionnels (faible latence)
  - RNN bidirectionnels (meilleure précision)
- Architectures LSTM profondes (empilement de couches)

Lien avec l’article :
- Architectures unidirectionnelles et bidirectionnelles
- Avantages des modèles profonds pour l’ASR

---

### 3. Le problème fondamental de l’alignement (≈ 1 min)

- En ASR :
  - entrée : séquence de frames audio
  - sortie : séquence de phonèmes ou de mots
- L’alignement frame ↔ label est **inconnu**
- Limites de l’entraînement par cross-entropy classique
- Nécessité d’un mécanisme d’alignement automatique

---

### 4. Connectionist Temporal Classification (CTC) (≈ 2 min)

- Principe du **CTC**
- Introduction du **blank label**
- Plusieurs alignements valides pour une même transcription
- Utilisation du forward–backward algorithm
- Différences avec les modèles HMM traditionnels

Illustrations possibles :
- Exemple simple de mot (“CAT”)
- Spikes de probabilités dans le temps

**Message clé :**  
> CTC permet d’apprendre **quoi prédire et quand le prédire** sans alignement explicite.

---

### 5. Améliorations clés proposées dans l’article Google (≈ 2 min)

#### 5.1 Frame stacking et subsampling
- Réduction du nombre de frames traitées
- Stabilisation de l’entraînement CTC
- Accélération du décodage

#### 5.2 Context-Dependent Phones
- Phones context-indépendants vs context-dépendants
- Importance du contexte phonétique
- Gains significatifs en WER

#### 5.3 Entraînement discriminatif de séquence (sMBR)
- Limites des critères CE et CTC
- Optimisation directe du Word Error Rate (WER)

**Message clé :**  
> Les performances viennent autant de l’architecture que des choix d’ingénierie.

---

### 6. Extension et ouverture (≈ 1 min)

Exploration d’une extension au-delà de l’article :
- Modèles acoustiques au niveau mot (CTC word models)
- Comparaison conceptuelle avec :
  - Attention-based models
  - Transformers / Conformers
- Discussion sur la pertinence actuelle de CTC

---

### 7. Conclusion (≈ 30 s)

- Résumé des concepts clés
- Transition vers la démonstration pratique
- Importance de CTC comme brique fondamentale de l’ASR moderne

---

## Plan du code demo (GitHub)

### Structure du dépôt

```text
speech-recognition-rnn-ctc/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── mini_librispeech/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lstm_ctc_training.ipynb
│   ├── 03_decoding_and_results.ipynb
│
└── src/
    ├── model.py
    ├── train.py
    └── decode.py
