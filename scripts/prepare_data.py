"""Prepara il dataset Speech Commands per il training.

Estrae le feature MFCC da ogni clip usando lo stesso preprocessing della
libreria C++, e salva tre archivi .npz (train/val/test).

L'estrazione richiede alcuni minuti: si fa una volta sola, poi i training
successivi caricano i .npz in pochi secondi.
"""

import os
import numpy as np
from scipy.io import wavfile

from features.mfcc import extract_mfcc, frame_count, SAMPLE_RATE, NUM_COEFFS

# --- Configurazione ---

DATA_DIR = "data/datasets/speech_commands_extracted"
OUT_DIR = "data/prepared"

KEYWORDS = ["go", "stop", "up", "down"]
LABELS = KEYWORDS + ["_unknown_", "_silence_"]
UNKNOWN_LABEL = len(KEYWORDS)
SILENCE_LABEL = len(KEYWORDS) + 1

NUM_UNKNOWN = 4000
NUM_SILENCE = 4000
SEED = 1234

NUM_FRAMES = frame_count(SAMPLE_RATE)

rng = np.random.default_rng(SEED)


# --- Lettura audio ---

def read_wav_padded(path):
    """Legge un WAV e lo porta esattamente a SAMPLE_RATE campioni.

    Alcune clip del dataset sono piu' corte di un secondo: vengono
    riempite di zeri in coda, altrimenti produrrebbero meno frame e le
    matrici avrebbero forme diverse.
    """
    rate, data = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if len(data) < SAMPLE_RATE:
        data = np.pad(data, (0, SAMPLE_RATE - len(data)))
    return data[:SAMPLE_RATE]


# --- Split ---

def load_split_list(path):
    """Carica un file di lista in un set di percorsi relativi."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def which_split(rel_path, val_set, test_set):
    """Determina lo split di un file dal suo percorso relativo.

    La divisione del dataset e' per PARLANTE, non casuale: le clip della
    stessa persona finiscono tutte nello stesso insieme. Dividere a caso
    gonfierebbe l'accuratezza, perche' il modello riconoscerebbe le voci
    viste in addestramento invece delle parole.
    """
    if rel_path in val_set:
        return "val"
    if rel_path in test_set:
        return "test"
    return "train"


# --- Raccolta dei file ---

def collect_keywords(val_set, test_set, buckets):
    """Aggiunge tutte le clip delle parole chiave, divise per split."""
    for label, word in enumerate(KEYWORDS):
        folder = os.path.join(DATA_DIR, word)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"missing keyword folder: {folder}")

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".wav"):
                continue
            rel = f"{word}/{filename}"
            split = which_split(rel, val_set, test_set)
            buckets[split].append((os.path.join(folder, filename), label))


def collect_unknown(val_set, test_set, buckets):
    """Campiona clip dalle parole NON chiave.

    Il dataset ha 31 parole oltre alle nostre quattro, per oltre 100.000
    clip: usarle tutte sbilancerebbe il training verso la classe unknown.
    Se ne campiona un numero paragonabile alle altre classi.
    """
    candidates = []
    for word in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, word)
        if not os.path.isdir(folder):
            continue
        if word in KEYWORDS or word.startswith("_"):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".wav"):
                candidates.append((os.path.join(folder, filename),
                                   f"{word}/{filename}"))

    n = min(NUM_UNKNOWN, len(candidates))
    picked = rng.choice(len(candidates), size=n, replace=False)

    for idx in picked:
        path, rel = candidates[idx]
        split = which_split(rel, val_set, test_set)
        buckets[split].append((path, UNKNOWN_LABEL))


def build_silence(buckets_arrays):
    """Genera clip di silenzio ritagliando i file di rumore di fondo.

    Non esistono clip di silenzio pronte: si ritagliano spezzoni casuali
    dalle registrazioni lunghe in _background_noise_.

    L'ampiezza viene scalata di un fattore casuale in modo che il modello
    veda rumore a volumi diversi, dal quasi-assoluto al chiaramente
    udibile, invece di imparare una singola intensita'.
    """
    noise_dir = os.path.join(DATA_DIR, "_background_noise_")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"missing folder: {noise_dir}")

    tracks = []
    for filename in sorted(os.listdir(noise_dir)):
        if filename.endswith(".wav"):
            rate, data = wavfile.read(os.path.join(noise_dir, filename))
            if rate == SAMPLE_RATE and len(data) > SAMPLE_RATE:
                tracks.append(data.astype(np.float64))

    if not tracks:
        raise RuntimeError("no usable background noise tracks found")

    # Queste clip non compaiono nelle liste ufficiali, quindi lo split
    # lo decidiamo noi con le stesse proporzioni approssimative.
    for i in range(NUM_SILENCE):
        track = tracks[rng.integers(len(tracks))]
        start = rng.integers(0, len(track) - SAMPLE_RATE)
        clip = track[start:start + SAMPLE_RATE] * rng.uniform(0.0, 1.0)
        clip = clip.astype(np.int16)

        r = rng.random()
        split = "val" if r < 0.1 else ("test" if r < 0.2 else "train")

        features = extract_mfcc(clip)
        buckets_arrays[split].append((features, SILENCE_LABEL))


# --- Estrazione ---

def extract_all(items, extra, split_name):
    """Estrae le feature da una lista di file, aggiungendo quelle gia'
    calcolate in 'extra' (le clip di silenzio, generate in memoria)."""
    total = len(items) + len(extra)
    X = np.zeros((total, NUM_FRAMES, NUM_COEFFS), dtype=np.float32)
    y = np.zeros(total, dtype=np.int32)

    for i, (path, label) in enumerate(items):
        if i % 500 == 0:
            print(f"  {split_name}: {i}/{total}")
        X[i] = extract_mfcc(read_wav_padded(path))
        y[i] = label

    offset = len(items)
    for j, (features, label) in enumerate(extra):
        X[offset + j] = features
        y[offset + j] = label

    # Mescola: le clip arrivano raggruppate per classe, e un ordine
    # sistematico puo' influenzare il training.
    order = rng.permutation(total)
    return X[order], y[order]


# --- Main ---

def main():
    val_set = load_split_list(os.path.join(DATA_DIR, "validation_list.txt"))
    test_set = load_split_list(os.path.join(DATA_DIR, "testing_list.txt"))
    print(f"split lists: {len(val_set)} validation, {len(test_set)} test")

    buckets = {"train": [], "val": [], "test": []}
    collect_keywords(val_set, test_set, buckets)
    collect_unknown(val_set, test_set, buckets)

    silence = {"train": [], "val": [], "test": []}
    print("generating silence clips...")
    build_silence(silence)

    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\nextracting {split} "
              f"({len(buckets[split])} files + {len(silence[split])} silence)")
        X, y = extract_all(buckets[split], silence[split], split)

        out_path = os.path.join(OUT_DIR, f"{split}.npz")
        np.savez_compressed(out_path, X=X, y=y)

        counts = np.bincount(y, minlength=len(LABELS))
        print(f"  saved {out_path}  shape={X.shape}")
        for label, count in zip(LABELS, counts):
            print(f"    {label:12s} {count}")


if __name__ == "__main__":
    main()