# SoundFlow: Intelligent Music Understanding and Generation

Authors: *Chenxi Guo, Jiayi Peng, Kexin Lyu, Yiran Tao* 

## Project Overview

SoundFlow is a deep learning-based system designed to comprehensively understand, classify, recommend, and generate music. Leveraging the NSynth dataset and techniques like CNNs, Res2Net, and conditional audio synthesis, this project explores the rich landscape of music through multiple interconnected tasks:

- Source Classification: Identify the sound production source (acoustic, electronic, synthetic).

- Instrument Family Classification: Recognize the broad family to which an instrument belongs.

- Sound Quality Classification: Detect multi-label sound qualities (e.g., bright, dark, reverb).

- Audio Recommendation System: Recommend similar musical sounds based on feature embeddings.

- Music Generation: Generate novel audio samples conditioned on musical attributes.

## Folder Structure

```
6600_FINAL_PROJECT/
├── code/
│   ├── 1_Source_Classification.ipynb            # Classify sound source
│   ├── 2_Family_Classification.ipynb            # Classify instrument families
│   ├── 3_Quality_Classification_CNN.ipynb       # CNN for quality classification
│   ├── 3_Quality_Classification_Res2Net.ipynb   # Res2Net for quality classification
│   ├── 4_Audio_Recommender.ipynb                # Nearest neighbor recommendation
│   ├── 5_Music_Generator.ipynb                  # Conditional music generation
│
├── README.md                                    # Project documentation
└── .gitignore
```

## Dataset

NSynth Dataset: An audio dataset from Google's Magenta project containing over 300,000 musical notes with rich labels like pitch, instrument family, source, and qualities. We sampled 4,096 audio clips for training and evaluation in this project.

Reference: https://drive.google.com/drive/folders/1SLvylh43clwYBra7qpypPmSmuhmalOZv?usp=drive_link

├── data/
│   ├── audio/                                   # Audio files (.wav)
│   └── examples.json                            # Metadata from NSynth

## Key Techniques

- 1D CNN for Raw Audio Classification: Efficient temporal feature extraction for source and family classification.

- 2D CNN and Res2Net for Spectrograms: Stronger receptive fields for capturing fine-grained sound qualities.

- Audio Recommendation with Embeddings: Nearest neighbor search in learned latent space.

- Conditional Music Generation: Log-mel + delta feature prediction followed by Griffin-Lim reconstruction.

## How to Run

1. Install dependencies:

    `pip install torch torchvision torchaudio librosa scikit-learn matplotlib numpy`

2. Follow the notebooks sequentially:

    - 1_Source_Classification.ipynb

    - 2_Family_Classification.ipynb

    - 3_Quality_Classification_CNN.ipynb or 3_Quality_Classification_Res2Net.ipynb

    - 4_Audio_Recommender.ipynb

    - 5_Music_Generator.ipynb

3. Listen to the generated audio outputs and explore recommendations!

## Future Work

- Enhanced Generation with Diffusion Models: Incorporate diffusion-based models to produce higher-quality and longer audio samples.

- Integrate Self-Supervised Learning: Utilize self-supervised techniques to better leverage unlabeled audio data, reducing reliance on annotated datasets.

- Multimodal Extensions: Combine audio with text or visual modalities (e.g., lyrics or album art) for richer classification and generation tasks.

- User-Centered Personalization: Extend the recommendation system to learn from user feedback and listening history for more personalized and dynamic suggestions.


## *Ready to explore the SoundFlow? Let's make the future of music with AI!*