import json
from django.shortcuts import render
from django.conf import settings
from pathlib import Path

import numpy as np
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.eeg_classifier import EEGClassifier

EMOTION_CLASSES = ['happy', 'sad', 'anxious', 'neutral']


def home(request):
    return render(request, 'eeg_api/home.html')


def models_view(request):
    eval_dir = Path(settings.BASE_DIR) / 'evaluation_plots'
    saved_dir = Path(settings.BASE_DIR) / 'saved_models'
    rf_img = eval_dir / 'confusion_matrix_random_forest.png'
    nn_img = eval_dir / 'confusion_matrix_shallow_nn.png'
    comp_img = eval_dir / 'confusion_matrices_comparison.png'
    roc_rf = eval_dir / 'roc_auc_random_forest.png'
    roc_nn = eval_dir / 'roc_auc_shallow_nn.png'
    roc_comp = eval_dir / 'roc_auc_comparison.png'

    split_info = None
    split_path = saved_dir / 'split_info.json'
    if split_path.exists():
        try:
            with open(split_path) as f:
                split_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    context = {
        'rf_img': '/static/confusion_matrix_random_forest.png' if rf_img.exists() else None,
        'nn_img': '/static/confusion_matrix_shallow_nn.png' if nn_img.exists() else None,
        'comparison_img': '/static/confusion_matrices_comparison.png' if comp_img.exists() else None,
        'roc_rf_img': '/static/roc_auc_random_forest.png' if roc_rf.exists() else None,
        'roc_nn_img': '/static/roc_auc_shallow_nn.png' if roc_nn.exists() else None,
        'roc_comparison_img': '/static/roc_auc_comparison.png' if roc_comp.exists() else None,
        'split_info': split_info,
    }
    return render(request, 'eeg_api/models.html', context)


def predict_view(request):
    result = None
    if request.method == 'POST':
        model_type = request.POST.get('model', 'random_forest')
        saved_path = Path(settings.BASE_DIR) / 'saved_models' / f'eeg_{model_type}.{"pkl" if model_type in ("random_forest", "svm") else "keras"}'

        if saved_path.exists():
            clf = EEGClassifier()
            clf.load_model(str(saved_path), model_type)
            clf.set_current_model(model_type)

            np.random.seed(int(request.POST.get('seed', 0)) or None)
            X = np.random.randn(1, 4096).astype(np.float32)
            preds, probs = clf.predict(X)

            pred_idx = int(preds[0])
            prob_arr = probs[0]
            result = {
                'emotion': EMOTION_CLASSES[pred_idx],
                'confidence': float(np.max(prob_arr)),
                'probs': {EMOTION_CLASSES[i]: float(prob_arr[i]) for i in range(4)},
            }
        else:
            result = {
                'emotion': 'N/A',
                'confidence': 0,
                'probs': {c: 0 for c in EMOTION_CLASSES},
                'error': 'Model not found. Run: python run_demo.py'
            }

    return render(request, 'eeg_api/predict.html', {'result': result})
