# EEG Emotion Prediction Dashboard

A full-stack ML project for EEG-based emotion classification. Includes multiple model architectures (CNN, Random Forest, SVM, Neural Networks), training pipelines, evaluation with confusion matrices, and a web dashboard for demo and visualization.

## Demo (5 minutes)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd "Prediction Dashboard"
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python run_demo.py
```

This will:
- Train models (if not already saved)
- Generate evaluation plots
- Start the web dashboard at **http://127.0.0.1:8000/**

### 3. Explore

- **Home** – Project overview
- **Models** – Compare Random Forest vs Shallow NN confusion matrices
- **Predict** – Run live predictions with sample EEG features

---

## Project Structure

```
├── models/
│   └── eeg_classifier.py    # CNN, Deep NN, Shallow NN, RF, SVM
├── eeg_api/                 # Django API & dashboard views
├── eeg_dashboard/           # Django project settings
├── train_and_save.py        # Train and persist models (70/15/15 split)
├── evaluate_and_visualize.py # Confusion matrices + ROC-AUC plots
├── run_demo.py              # One-command demo launcher
└── Datasets/                # BIDS EEG data (add locally; demo uses synthetic data)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `python run_demo.py` | Full demo: train, evaluate, start server |
| `python train_and_save.py --model random_forest` | Train a specific model |
| `python evaluate_and_visualize.py` | Regenerate plots (confusion matrix + ROC-AUC) |
| `python manage.py runserver` | Start Django server only |

## Model Types

- **CNN** – Convolutional neural network (64×64 input)
- **Shallow NN** – Single hidden layer (4096 → 64 → 4)
- **Deep NN** – 3 hidden layers (128 → 64 → 32)
- **Random Forest** – Ensemble of 100 trees
- **SVM** – RBF kernel with probability output

## Emotion Classes

`happy` | `sad` | `anxious` | `neutral`

## Tech Stack

- **Backend:** Django 4.2, Django REST Framework
- **ML:** TensorFlow/Keras, scikit-learn
- **Visualization:** matplotlib, seaborn
- **EEG:** MNE-Python (for real data preprocessing)

---

## Interview Demo Tips

**Talking points:**
1. **Problem** – EEG signals are high-dimensional; emotion labels are subjective.
2. **Approach** – Compare multiple model families (tree-based vs neural) on the same task.
3. **Evaluation** – Confusion matrices highlight which emotions are confused (e.g. sad vs anxious).
4. **Pipeline** – Data → preprocessing → train → evaluate → visualize → API/dashboard.
5. **Tech stack** – Django (web), TensorFlow/Keras (deep learning), scikit-learn (traditional ML).

**Demo flow:** Run `python run_demo.py` → open http://127.0.0.1:8000 → walk through Home → Models → Predict.

## License

MIT
