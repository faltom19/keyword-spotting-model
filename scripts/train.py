import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

NPZ_DIR = "data/prepared"
OUT_DIR = "data/model_finish"

KEYWORDS = ["go", "stop", "up", "down"]
LABELS = KEYWORDS + ["_unknown_", "_silence_"]
UNKNOWN_LABEL = len(KEYWORDS)
SILENCE_LABEL = len(KEYWORDS) + 1
CONV_FILTERS = (8, 16, 32)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 8

data_test = np.load(NPZ_DIR + '/test.npz')
data_train = np.load(NPZ_DIR + '/train.npz')
data_val = np.load(NPZ_DIR + '/val.npz')

print("TEST:")
for key in data_test.files:
    print(key, data_test[key].shape)

print("\nTRAIN:")
for key in data_train.files:
    print(key, data_train[key].shape)

print("\nVALIDATION:")
for key in data_val.files:
    print(key, data_val[key].shape)

X_test = data_test["X"]
y_test = data_test["y"]

X_train = data_train["X"]
y_train = data_train["y"]

X_val = data_val["X"]
y_val = data_val["y"]

os.makedirs(OUT_DIR, exist_ok=True)

mean = X_train.mean(axis=(0, 1))   # forma (13,)
std = X_train.std(axis=(0, 1))     # forma (13,)

out_path = os.path.join(OUT_DIR, "norm_stats.npz")
np.savez(out_path, mean=mean, std=std)
print("mean:", np.round(mean, 2))
print("std: ", np.round(std, 2))

# Le stesse statistiche del training applicate a tutti gli insiemi
X_train = (X_train - mean) / (std + 1e-8)
X_val   = (X_val   - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)

print("after norm - mean:", np.round(X_train.mean(), 4), "std:", np.round(X_train.std(), 4))

X_train = X_train[..., np.newaxis]
X_val = X_val [..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

print(X_train.shape)   # (18696, 61, 13, 1)
print(X_val.shape)
print(X_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(61, 13, 1)),

    tf.keras.layers.Conv2D(8, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2),       

    tf.keras.layers.Conv2D(16, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2),      
    
    tf.keras.layers.Conv2D(32, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(2),    

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation="softmax"),
])

model.summary()

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="max"
        )
    ],
)

model.save(OUT_DIR+"/Final.keras")

_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy model: ", accuracy*100)

# --- Matrice di confusione ---
# La cella (i, j) conta quante volte la classe vera i e' stata predetta
# come j. La diagonale sono i successi, tutto il resto sono errori.

probs = model.predict(X_test, verbose=0)
predicted = np.argmax(probs, axis=1)

cm = tf.math.confusion_matrix(y_test, predicted, num_classes=len(LABELS)).numpy()

print("\nconfusion matrix (rows = true, columns = predicted)")
print(f"{'':12s}" + "".join(f"{label:>11s}" for label in LABELS))
for i, label in enumerate(LABELS):
    row = "".join(f"{count:>11d}" for count in cm[i])
    print(f"{label:12s}{row}")

# --- Accuratezza per classe ---
# Diagonale diviso totale della riga: quanto bene il modello riconosce
# ciascuna classe, indipendentemente da quanto e' frequente.

print("\nper-class accuracy")
for i, label in enumerate(LABELS):
    total = cm[i].sum()
    correct = cm[i, i]
    print(f"  {label:12s} {correct / total * 100:5.1f}%  ({correct}/{total})")