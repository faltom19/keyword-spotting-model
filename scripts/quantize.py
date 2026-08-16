import os
import numpy as np
import tensorflow as tf

NPZ_DIR = "data/prepared"
KERAS_DIR = "data/model_finish"
KEYWORDS = ["go", "stop", "up", "down"]
LABELS = KEYWORDS + ["_unknown_", "_silence_"]

model = tf.keras.models.load_model(KERAS_DIR + '/Final.keras')


data_test = np.load(NPZ_DIR + '/test.npz')
data_train = np.load(NPZ_DIR + '/train.npz')

stats = np.load(KERAS_DIR + '/norm_stats.npz')
mean = stats["mean"]
std = stats["std"]

X_test = data_test["X"]
y_test = data_test["y"]

X_train = data_train["X"]
y_train = data_train["y"]


# Le stesse statistiche del training applicate a tutti gli insiemi
X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"float32 accuracy: {acc * 100:.2f}%")


def representative_dataset():
    rng = np.random.default_rng(1234)
    indices = rng.choice(X_train.shape[0], size=200, replace=False)
    for i in indices:
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

output_path = KERAS_DIR + "/model_int8.tflite"

with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"Modello salvato in: {output_path}")

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero = input_details["quantization"]
out_scale, out_zero = output_details["quantization"]

print("Scale: ", in_scale , " e Zero: ", in_zero)

counter = 0

for i in range(len(X_test)):
    quantized = np.round(X_test[i:i+1] / in_scale) + in_zero
    quantized = np.clip(quantized, -128, 127)
    quantized = quantized.astype(np.int8)

    interpreter.set_tensor(input_details["index"], quantized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    predicted = np.argmax(output)
    if predicted == y_test[i]:
        counter += 1

    if i % 500 == 0:
        print(f"  {i}/{len(X_test)}")

print("accuracy after quant: ", counter / len(X_test) * 100)
print(f"accuracy loss: {(acc * 100) - (counter / len(X_test) * 100):.2f} points")
print("dimensione: ", len(tflite_model) / 1024)

# --- Export come header C ---
# L'ESP32 non ha filesystem: il modello va compilato dentro il firmware
# come dati costanti, che finiscono in flash insieme al codice.

def to_c_array(data, per_line=12):
    """Formatta una sequenza di byte come lista di letterali esadecimali."""
    lines = []
    for i in range(0, len(data), per_line):
        chunk = data[i:i + per_line]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    return ",\n".join(lines)


def to_float_array(values):
    """Formatta un array di float con precisione sufficiente.

    Le cifre contano: mean e std devono coincidere con quelle usate in
    addestramento, altrimenti il modello riceve feature fuori scala.
    """
    return ", ".join(f"{v:.8f}f" for v in values)


header_path = os.path.join(KERAS_DIR, "kws_model.h")

with open(header_path, "w") as f:
    f.write(f"""// Generato automaticamente da scripts/quantize.py -- non modificare a mano.
//
// Modello di keyword spotting quantizzato a INT8, con i parametri
// necessari a preprocessare le feature nello stesso modo usato in
// addestramento.

#ifndef KWS_MODEL_H
#define KWS_MODEL_H

#include <cstddef>
#include <cstdint>

// Numero di classi e relative etichette, nell'ordine usato dal modello.
constexpr std::size_t kKwsNumLabels = {len(LABELS)};
inline const char* const kKwsLabels[kKwsNumLabels] = {{
  {", ".join(f'"{label}"' for label in LABELS)}
}};

// Forma dell'input atteso dal modello.
constexpr std::size_t kKwsNumFrames = {X_test.shape[1]};
constexpr std::size_t kKwsNumCoeffs = {X_test.shape[2]};

// Normalizzazione delle feature.
// Da applicare PRIMA della quantizzazione:
//     normalizzata = (mfcc - mean) / std
constexpr float kKwsFeatureMean[kKwsNumCoeffs] = {{
  {to_float_array(mean)}
}};
constexpr float kKwsFeatureStd[kKwsNumCoeffs] = {{
  {to_float_array(std)}
}};

// Quantizzazione dell'input.
// Da applicare DOPO la normalizzazione:
//     int8 = clamp(round(normalizzata / scale) + zeroPoint, -128, 127)
constexpr float kKwsInputScale = {in_scale:.10f}f;
constexpr int kKwsInputZeroPoint = {in_zero};

// Dequantizzazione dell'output, per ottenere le probabilita':
//     probabilita' = (int8 - zeroPoint) * scale
constexpr float kKwsOutputScale = {out_scale:.10f}f;
constexpr int kKwsOutputZeroPoint = {out_zero};

// Il modello serializzato.
// alignas(16) e' richiesto da TensorFlow Lite Micro: senza allineamento
// alcune architetture generano errori di accesso alla memoria.
constexpr unsigned int kKwsModelLen = {len(tflite_model)};
alignas(16) const unsigned char kKwsModel[kKwsModelLen] = {{
{to_c_array(tflite_model)}
}};

#endif // KWS_MODEL_H
""")

print(f"header salvato in: {header_path}  ({len(tflite_model)} bytes)")