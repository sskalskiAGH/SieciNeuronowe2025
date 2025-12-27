import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from multiprocessing import Pool, cpu_count
import time
from openpyxl import load_workbook

# ===============================
# FUNKCJE AKTYWACJI
# ===============================
def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ===============================
# INICJALIZACJA WAG + BIASÓW
# ===============================
def initialize_parameters(layer_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

# ===============================
# FORWARD / BACKWARD
# ===============================
def forward(X, weights, biases):
    activations = [X]
    for W, b in zip(weights[:-1], biases[:-1]):
        A = relu(activations[-1] @ W + b)
        activations.append(A)
    A_out = softmax(activations[-1] @ weights[-1] + biases[-1])
    activations.append(A_out)
    return activations

def backward(activations, weights, y):
    deltas = [activations[-1] - y]
    for i in reversed(range(len(weights) - 1)):
        delta = deltas[0] @ weights[i + 1].T
        delta *= relu(activations[i + 1], derivative=True)
        deltas.insert(0, delta)

    grads_W, grads_b = [], []
    for i in range(len(weights)):
        grads_W.append(activations[i].T @ deltas[i] / len(y))
        grads_b.append(np.mean(deltas[i], axis=0, keepdims=True))
    return grads_W, grads_b

# ===============================
# TRENING
# ===============================
def train(X, y, hidden_layers, lr, epochs,
          optimizer="sgd", momentum=0.9, beta1=0.9, beta2=0.999):

    layer_sizes = [X.shape[1]] + hidden_layers + [y.shape[1]]
    weights, biases = initialize_parameters(layer_sizes)

    vW = [np.zeros_like(w) for w in weights]
    vb = [np.zeros_like(b) for b in biases]
    mW = [np.zeros_like(w) for w in weights]
    mb = [np.zeros_like(b) for b in biases]
    sW = [np.zeros_like(w) for w in weights]
    sb = [np.zeros_like(b) for b in biases]
    eps = 1e-8

    for _ in range(epochs):
        activations = forward(X, weights, biases)
        grads_W, grads_b = backward(activations, weights, y)

        for i in range(len(weights)):
            if optimizer == "sgd":
                weights[i] -= lr * grads_W[i]
                biases[i] -= lr * grads_b[i]

            elif optimizer == "momentum":
                vW[i] = momentum * vW[i] - lr * grads_W[i]
                vb[i] = momentum * vb[i] - lr * grads_b[i]
                weights[i] += vW[i]
                biases[i] += vb[i]

            elif optimizer == "adam":
                mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                sW[i] = beta2 * sW[i] + (1 - beta2) * (grads_W[i] ** 2)
                sb[i] = beta2 * sb[i] + (1 - beta2) * (grads_b[i] ** 2)

                weights[i] -= lr * mW[i] / (np.sqrt(sW[i]) + eps)
                biases[i] -= lr * mb[i] / (np.sqrt(sb[i]) + eps)

    return weights, biases

# ===============================
# PREDYKCJA
# ===============================
def predict(X, weights, biases):
    A = X
    for W, b in zip(weights[:-1], biases[:-1]):
        A = relu(A @ W + b)
    A = softmax(A @ weights[-1] + biases[-1])
    return np.argmax(A, axis=1)

# ===============================
# JEDNA KOMBINACJA
# ===============================
def run_combination(params):
    lr, epochs, hidden, opt, mom, repeat, X_train, y_train_oh, y_train_lbl, X_test, y_test_lbl = params

    acc_train, acc_test, prec, rec, f1 = [], [], [], [], []

    for _ in range(repeat):
        weights, biases = train(
            X_train, y_train_oh,
            hidden_layers=hidden,
            lr=lr,
            epochs=epochs,
            optimizer=opt,
            momentum=mom if mom else 0.0
        )

        pred_train = predict(X_train, weights, biases)
        pred_test = predict(X_test, weights, biases)

        acc_train.append(accuracy_score(y_train_lbl, pred_train))
        acc_test.append(accuracy_score(y_test_lbl, pred_test))
        prec.append(precision_score(y_test_lbl, pred_test, average="macro", zero_division=0))
        rec.append(recall_score(y_test_lbl, pred_test, average="macro", zero_division=0))
        f1.append(f1_score(y_test_lbl, pred_test, average="macro", zero_division=0))

    return {
        "hidden_layers": str(hidden),
        "learning_rate": lr,
        "epochs": epochs,
        "optimizer": opt,
        "momentum": mom,
        "acc_train_mean": np.mean(acc_train),
        "acc_train_best": np.max(acc_train),
        "acc_test_mean": np.mean(acc_test),
        "acc_test_best": np.max(acc_test),
        "precision_mean": np.mean(prec),
        "recall_mean": np.mean(rec),
        "f1_mean": np.mean(f1)
    }

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    # ---- DANE ----
    df = pd.read_excel("sensor_readings_24_outcome.xlsx")
    X = df[[f"US{i}" for i in range(1, 25)]].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    y = df["Class"].values
    classes = np.unique(y)
    y_idx = np.array([np.where(classes == c)[0][0] for c in y])
    y_oh = np.eye(len(classes))[y_idx]

    train_mask = df["Set"] == "train"
    test_mask = df["Set"] == "test"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_oh = y_oh[train_mask]
    y_train_lbl = y_idx[train_mask]
    y_test_lbl = y_idx[test_mask]

    # ---- HIPERPARAMETRY ----
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    epochs_list = [1500, 1200, 900, 500]
    hidden_layer_configs = [
        [128, 64, 32, 16], [64, 32, 16, 8], [32, 16, 8, 4],
        [64, 32, 16], [32, 16, 8], [16, 8, 4],
        [64, 32], [32, 16], [16, 8], [8, 4],
        [64], [32], [16], [8]
    ]
    optimizers = ["sgd", "momentum", "adam"]
    momentum_values = [0.6, 0.7, 0.8, 0.9]
    repeat = 4

    all_combinations = []
    for lr, ep, hid, opt in product(learning_rates, epochs_list, hidden_layer_configs, optimizers):
        moms = [None] if opt != "momentum" else momentum_values
        for m in moms:
            all_combinations.append(
                (lr, ep, hid, opt, m, repeat,
                 X_train, y_train_oh, y_train_lbl,
                 X_test, y_test_lbl)
            )

    total = len(all_combinations)
    print(f"Liczba kombinacji: {total}")

    # ---- PLIK WYNIKÓW ----
    output_file = "porownanie_optymalizatorow_multiproc.xlsx"
    columns = [
        "hidden_layers", "learning_rate", "epochs", "optimizer", "momentum",
        "acc_train_mean", "acc_train_best",
        "acc_test_mean", "acc_test_best",
        "precision_mean", "recall_mean", "f1_mean"
    ]
    pd.DataFrame(columns=columns).to_excel(output_file, index=False)

    start = time.time()

    # ---- MULTIPROCESSING + ZAPIS NA BIEŻĄCO ----
    with Pool(cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(run_combination, all_combinations), 1):

            wb = load_workbook(output_file)
            ws = wb.active
            ws.append(list(res.values()))
            wb.save(output_file)

            elapsed = time.time() - start
            avg = elapsed / i
            remaining = avg * (total - i)

            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)

            print(
                f"[{i}/{total}] {res['hidden_layers']} | {res['optimizer']} | "
                f"lr={res['learning_rate']} | ep={res['epochs']} | mom={res['momentum']} | "
                f"TEST_mean={res['acc_test_mean']:.3f} | "
                f"ETA {int(h)}h {int(m)}m {int(s)}s"
            )

    print("\nZapisano:", output_file)
