#  Deepfake Audio Detection System

## Project Description

This project is a machine learning system for detecting deepfake audio using acoustic features extracted from speech signals.

The model is based on a **Random Forest classifier** trained on multilingual and multi-domain datasets, including ASVspoof and in-the-wild audio data.

The system extracts **eGeMAPS acoustic features** using the OpenSMILE toolkit and performs binary classification:

- REAL audio
- FAKE (synthetic / deepfake) audio

---

## Key Idea

The model analyzes **acoustic and linguistic-proxy features** of speech, including:

- Pitch (fundamental frequency)
- Formant structure (F1, F2, F3)
- Voice stability (jitter, shimmer)
- Spectral and energy characteristics
- Loudness dynamics

These features capture subtle distortions introduced by speech synthesis systems.

---

## Pipeline

1. Audio input (WAV/MP3)
2. Feature extraction using OpenSMILE (eGeMAPS v02)
3. Feature alignment with training schema
4. Random Forest inference
5. Probability estimation (REAL / FAKE)
6. Feature-based interpretation of results

---

## Model Details

- Model: Random Forest Classifier
- Features: eGeMAPS (88 features)
- Training datasets:
  - ASVspoof2019
  - ASVspoof2021
  - ASVspoof5
  - InTheWild
  - ODSS
- Class balancing: applied via `compute_class_weight`

---

## Web Interface (Gradio)

The project includes an interactive web interface built with **Gradio**, where users can:

- Upload an audio file
- Receive prediction (REAL / FAKE)
- View confidence scores
- See main acoustic characteristics influencing the decision

---

##  Output Example
<img width="736" height="532" alt="Снимок экрана 2026-06-25 в 16 38 20" src="https://github.com/user-attachments/assets/6f91912d-c454-4e88-a851-9ef55195d435" />

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Run application

```bash
python app.py
```

### 3. Open browser

```
http://localhost:7860
```

---

## Project Structure

```
final_model/
│
├── app.py                  # Gradio interface
├── train_final.py         # Model training script
├── predict.py             # Inference script
├── rf_deepfake_final.pkl  # Trained model
├── feature_names.pkl      # Feature order
├── feature_mean.pkl       # Feature normalization stats
├── feature_std.pkl
└── README.md
```

---

## Notes

* Large model file is excluded from GitHub due to size limits.
* OpenSMILE is required for feature extraction.

