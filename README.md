# 🎵 Music Generation with AI — Task 3

> An end-to-end deep learning pipeline that composes original piano music using LSTM neural networks trained on classical MIDI data.

**Author:** Roshni Sukhnani | [@weiwei-gitch](https://github.com/weiwei-gitch)  
**Internship:** DecodeLabs — Task 3

---

## 🧠 What It Does

This project builds an AI model that **learns musical patterns** from classical MIDI files and **generates brand new compositions** — no human input required after training.

```
Classical MIDI Files → Parse Notes → Train LSTM → Generate Music 🎵
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.14` | Core language |
| `music21` | MIDI parsing and generation |
| `PyTorch` | LSTM neural network |
| `NumPy` | Data processing |
| `tqdm` | Training progress bars |

---

## 📁 Project Structure

```
music_ai/
│
├── data/
│   ├── midi_files/        ← Input MIDI training data (14 files)
│   ├── notes.pkl          ← Preprocessed note sequences
│   └── model.pth          ← Trained model weights
│
├── output/
│   └── generated_music.mid ← AI-composed music output
│
├── preprocess.py          ← Step 2: Parse MIDI → note sequences
├── model.py               ← Step 3: LSTM architecture
├── train.py               ← Step 4: Model training
└── generate.py            ← Step 5: Music generation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install music21 numpy tqdm
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Add MIDI Files
Place `.mid` files inside `data/midi_files/`  
*(Classical piano recommended — [bitmidi.com](https://bitmidi.com))*

### 3. Preprocess
```bash
python preprocess.py
```

### 4. Train the Model
```bash
python train.py
```

### 5. Generate Music
```bash
python generate.py
```

Open `output/generated_music.mid` with VLC or Windows Media Player 🎵

---

## 📊 Results

| Metric | Value |
|---|---|
| MIDI files used | 14 |
| Total notes extracted | 11,254 |
| Unique notes/chords | 480 |
| Model parameters | 4,391,648 |
| Training epochs | 20 |
| Final training loss | 4.08 |
| Notes generated | 200 |

---

## 🏗️ Model Architecture

```
Input (480) → LSTM(512) → LSTM(512) → Dense(256) → ReLU → Output(480)
```

- **2-layer LSTM** with hidden size 512
- **Dropout (0.3)** to prevent overfitting
- **CrossEntropyLoss** + **Adam optimizer** (lr=0.001)
- **Temperature sampling (0.8)** for creative note selection

---

## 🎼 Pipeline Explained

### Step 1 — Data Collection
Downloaded 14 classical piano MIDI files from [bitmidi.com](https://bitmidi.com) covering composers like Beethoven, Mozart, and Chopin.

### Step 2 — Preprocessing
Used `music21` to parse each MIDI file into individual note and chord names. Extracted 11,254 total notes with a vocabulary of 480 unique notes/chords.

### Step 3 — Model Design
Built a 2-layer LSTM network with 4.3M parameters using PyTorch. The model takes sequences of 50 notes and predicts the next note.

### Step 4 — Training
Trained for 20 epochs with batch size 64. Loss decreased from 4.79 → 4.08, showing the model successfully learned musical patterns.

### Step 5 — Generation
Fed a random 50-note seed sequence to the trained model and generated 200 new notes using temperature-based sampling for musical variety.

---

## 📌 Notes

- Training runs on **CPU** — no GPU required
- Generation takes under **5 seconds**
- Output is a standard `.mid` file playable on any media player

---

*Built as part of DecodeLabs Internship — Task 3*
