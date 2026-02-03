"""
One-command demo launcher for the EEG Prediction Dashboard.
Trains models (if needed), generates plots, starts the web server.
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(BASE_DIR, 'saved_models')
PLOTS = os.path.join(BASE_DIR, 'evaluation_plots')


def main():
    os.chdir(BASE_DIR)
    venv_python = os.path.join(BASE_DIR, 'venv', 'Scripts', 'python.exe')
    if not os.path.exists(venv_python):
        venv_python = os.path.join(BASE_DIR, 'venv', 'bin', 'python')
    python = venv_python if os.path.exists(venv_python) else sys.executable

    # 1. Train models if not saved (or if test split missing from old runs)
    rf_path = os.path.join(SAVED, 'eeg_random_forest.pkl')
    nn_path = os.path.join(SAVED, 'eeg_shallow_nn.keras')
    test_data_path = os.path.join(SAVED, 'test_data.npz')
    needs_train = not os.path.exists(test_data_path)

    if not os.path.exists(rf_path) or needs_train:
        print("Training Random Forest (70/15/15 split)...")
        subprocess.run([python, 'train_and_save.py', '--model', 'random_forest', '--samples', '600'], check=True)
    if not os.path.exists(nn_path) or needs_train:
        print("Training Shallow NN (70/15/15 split)...")
        subprocess.run([python, 'train_and_save.py', '--model', 'shallow_nn', '--samples', '600'], check=True)

    # 2. Generate evaluation plots (uses held-out test set)
    print("Generating evaluation plots...")
    subprocess.run([python, 'evaluate_and_visualize.py'], check=True)

    # 3. Start server
    print("\n" + "=" * 50)
    print("Starting dashboard at http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")
    subprocess.run([python, 'manage.py', 'runserver'])


if __name__ == '__main__':
    main()
