# Speech_Recognition
A deep learning project to explore the topic of speech recognition
# Recurrent Neural Network Acoustic Models for Speech Recognition 

This project explores **RNN Acoustic Models**, based on the paper **"Fast and Accurate Recurrent Neural Network Acoustic Models for Speech Recognition"** by Hasim Sak et al. (Google).

### Resources
* **Video Presentation:** [Link too Video](https://youtu.be/cUy66VpI4dU)
* **Original Paper:** [Link to PDF](https://arxiv.org/pdf/1507.06947)

### Quick Start

```bash
pip install -r requirements.txt
```

```asr.ipynb``` -> an exploration of the topic 
```pipeline.ipynb``` -> a pipeline to try a model (see *Model Versions*) and recon your own voice (works only with english)

### Model Versions
All the following models have been trained for 50 epochs with a learning rate of 5e-5 and a batch size of 10.

* **V1** : SpeechRecognition | bidirectional | 5 layers of 256 | reduced_percentage : None  
* **V2** : SpeechRecognition | unidirectional | 5 layers of 512 | reduced_percentage : None 
* **V3** : SpeechRecognitionStacking | bidirectional | 5 layers of 256 | reduced_percentage : None | stride of 3 frames | stack of 8 frames  
* **V4** : SpeechRecognitionStacking | unidirectional | 5 layers of 512 | reduced_percentage : None | stride of 3 frames | stack of 8 frames 


### Abstract Glossary

#### Neural Architectures
* **LSTM RNN (Long Short-Term Memory):** A recurrent network with "memory gates" that captures long-term dependencies, solving the vanishing gradient problem of standard RNNs.
* **DNN (Feed Forward):** Standard deep networks without memory loops. The baseline this paper aims to outperform.

#### Modeling Concepts
* **Acoustic Model:** The component that calculates the probability of a phoneme given an audio input.
* **HMM (Hidden Markov Model):** Statistical model used to handle the time evolution and transitions between phonemes.
* **CD (Context Dependent):** Modeling phonemes based on their neighbors (e.g., *triphones*) rather than in isolation.
* **CTC (Connectionist Temporal Classification):** A training technique that allows the network to learn sequences without needing precise frame-by-frame alignment labels.

#### Optimization
* **Sequence Training:** Optimizing the model to minimize errors on the entire sentence sequence, rather than just individual frames.
* **Frame Stacking:** Concatenating multiple consecutive audio frames into a single input to give the network more immediate context.
* **Reduced Frame Rate:** Processing audio at longer time intervals (skipping frames) to significantly speed up decoding.
