import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# --- 1. PREPARACIÓN DE DATOS ---
base_path = 'g:/Mi unidad/01_MASTER/MUIIA/08_Deep Learning/Trabajo/datos/datos'
train_csv_path = os.path.join(base_path, 'entrenamiento/entrenamiento.csv')

df = pd.read_csv(train_csv_path, sep=';')
df['Ruta'] = df['Imagen'].apply(lambda x: os.path.join(base_path, 'entrenamiento', x))

# Create mapping dictionary
clases = sorted(df['Clase'].unique())
clase_to_idx = {c: i for i, c in enumerate(clases)}
df['Label'] = df['Clase'].map(clase_to_idx)

grupos = df['GrupoFuncional'].unique()
grupo_to_idx = {g: i for i, g in enumerate(grupos)}

num_clases = len(clases)
clase_a_grupo_list = np.zeros(num_clases, dtype=np.int32)
for clase in clases:
    grupo = df[df['Clase'] == clase]['GrupoFuncional'].iloc[0]
    clase_a_grupo_list[clase_to_idx[clase]] = grupo_to_idx[grupo]

clase_a_grupo = tf.constant(clase_a_grupo_list, dtype=tf.int32)

print("Number of classes:", num_clases)

# Splitting dataframe
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def parse_image(ruta, label):
    image = tf.io.read_file(ruta)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # The image is scaled inside the model via Rescaling layer if preferred,
    # or we can do it here. Let's do it here.
    image = image / 255.0
    return image, label

def load_dataset(dataframe):
    rutas = dataframe['Ruta'].values
    labels = dataframe['Label'].values
    dataset = tf.data.Dataset.from_tensor_slices((rutas, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

train_ds = load_dataset(train_df).shuffle(2000).batch(BATCH_SIZE).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = load_dataset(val_df).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 2. FUNCIÓN DE PERDIDA ---
def loss_jerarquica(clase_a_grupo, alpha=0.5, gamma=0.3):
    clase_a_grupo = tf.cast(clase_a_grupo, tf.int32)

    def perdida(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_espacial = tf.expand_dims(y_true, axis=1)

        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        familias_true = tf.gather(clase_a_grupo, y_true_espacial)
        familias_all = tf.expand_dims(clase_a_grupo, axis=0) # [1, num_clases]
        matrices_iguales = tf.equal(familias_true, familias_all) # [batch, num_clases]

        cost_matrix = tf.where(matrices_iguales, tf.constant(gamma, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32))

        # Restar penalización cuando se clasifica correctamente
        indices_true = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=-1)
        cost_matrix = tf.tensor_scatter_nd_update(cost_matrix, indices_true, tf.zeros(tf.shape(y_true), dtype=tf.float32))

        h_loss = tf.reduce_sum(cost_matrix * y_pred, axis=1)

        return ce_loss + alpha * h_loss
    return perdida

# Redefinir métrica de Accuracy y Hierarchical Loss
# Para facilitar la comparación durante el entrenamiento
def accuracy(y_true, y_pred):
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

# --- 3. MODELO 1: CUSTOM CNN ---
def build_custom_cnn():
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2,2)(x)
    
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2,2)(x)
    
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2,2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_clases, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

print("\n--- Entrenando Modelo Custom CNN ---")
custom_model = build_custom_cnn()
custom_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
                     loss=loss_jerarquica(clase_a_grupo),
                     metrics=[accuracy])

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_custom = custom_model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[stop_early])

custom_model.save('modelo_custom.keras')

print("\n--- Entrenamiento Modelo Custom Terminado ---")
