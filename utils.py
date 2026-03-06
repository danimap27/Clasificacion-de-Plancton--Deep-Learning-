import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

def cargar_y_preparar_datos(csv_path, img_dir, img_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Carga el CSV, crea los tensores de mapeo grupo-clase,
    calcula los pesos de clase para balanceo y construye los tf.data.Dataset.
    """
    df = pd.read_csv(csv_path, sep=';')
    df['Ruta'] = df['Imagen'].apply(lambda x: os.path.join(img_dir, x))

    # Mapeo de Clases
    clases = sorted(df['Clase'].unique())
    clase_to_idx = {c: i for i, c in enumerate(clases)}
    df['Label'] = df['Clase'].map(clase_to_idx)
    num_clases = len(clases)

    # Mapeo a Grupos Funcionales
    grupos = sorted(df['GrupoFuncional'].unique())
    grupo_to_idx = {g: i for i, g in enumerate(grupos)}

    clase_a_grupo_list = np.zeros(num_clases, dtype=np.int32)
    for clase in clases:
        grupo = df[df['Clase'] == clase]['GrupoFuncional'].iloc[0]
        clase_a_grupo_list[clase_to_idx[clase]] = grupo_to_idx[grupo]

    clase_a_grupo_tensor = tf.constant(clase_a_grupo_list, dtype=tf.int32)

    # Cálculo de class_weights
    y = df['Label'].values
    pesos_array = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: peso for i, peso in enumerate(pesos_array)}

    # Train/Val Split
    train_df, val_df = train_test_split(df, test_size=validation_split, stratify=df['Label'], random_state=42)

    def parse_image(ruta, label):
        image = tf.io.read_file(ruta)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return image, label

    def df_to_dataset(dataframe):
        rutas = dataframe['Ruta'].values
        labels = dataframe['Label'].values
        ds = tf.data.Dataset.from_tensor_slices((rutas, labels))
        ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    # Data Augmentation más dinámico definido aparte, aquí solo devolvemos los ds limpios
    train_ds_base = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds_base, val_ds, class_weights_dict, clase_a_grupo_tensor, num_clases

def cargar_datos_test(csv_path, img_dir, img_size=(128, 128), batch_size=32):
    """
    Carga el dataset de prueba (test) para la evaluación final de los modelos.
    No requiere class_weights ni desordenar (shuffle).
    """
    df = pd.read_csv(csv_path, sep=';')
    df['Ruta'] = df['Imagen'].apply(lambda x: os.path.join(img_dir, x))

    # Mapeo de Clases usando el orden alfabético estricto que usamos en entrenamiento
    clases = sorted(df['Clase'].unique())
    clase_to_idx = {c: i for i, c in enumerate(clases)}
    df['Label'] = df['Clase'].map(clase_to_idx)

    def parse_image(ruta, label):
        image = tf.io.read_file(ruta)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return image, label

    rutas = df['Ruta'].values
    labels = df['Label'].values
    
    test_ds = tf.data.Dataset.from_tensor_slices((rutas, labels))
    test_ds = test_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch y prefetch (sin shuffle porque es solo para evaluación)
    test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return test_ds

def apply_data_augmentation(train_ds_base, batch_size):
    """
    Aplica rotaciones, giros aleatorios y transformaciones 
    que ayudan substancialmente a redes que ven ejemplos limitados de clases raras.
    """
    def augment_image(image, label):
        # Operaciones nativas de tf.image son órdenes de magnitud más rápidas en CPU/tf.data
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Añadir brillo o contraste aleatorio es más eficiente que rotaciones complejas en CPU
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    # Mapeamos usando tf.image nativo que paraleliza perfectamente
    train_ds = train_ds_base.cache().shuffle(2000).map(
        augment_image, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds

def build_loss_jerarquica(clase_a_grupo, gamma=0.3, alpha=0.5):
    groupos_func = tf.constant(clase_a_grupo, tf.int32)
    num_clases = len(clase_a_grupo)
    costes = tf.ones((num_clases, num_clases)) - tf.eye(num_clases)
    mismo_grupo = tf.equal(groupos_func[:,None], groupos_func[None,:])
    costes = tf.where(mismo_grupo, gamma*tf.ones_like(costes), costes)
    costes = costes - tf.eye(num_clases) * gamma
    
    def loss_fn(y_real, logits):
        y_real = tf.cast(tf.reshape(y_real, [-1]), tf.int32)
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_real, logits)
        jerarquico = tf.reduce_sum(tf.gather(costes, y_real) * logits, axis=1)
        return cross_entropy + alpha * jerarquico
    
    return loss_fn

def plot_training_history(history, title="Historial del Modelo"):
    """
    Función utilitaria para comparar uniformemente el rendimiento entre modelos.
    """
    acc = history.history.get('accuracy', history.history.get('sparse_categorical_accuracy', history.history.get('acc_metric')))
    val_acc = history.history.get('val_accuracy', history.history.get('val_sparse_categorical_accuracy', history.history.get('val_acc_metric')))
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.legend(loc='lower right')
    plt.title(f'Precisión - {title}')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.legend(loc='upper right')
    plt.title(f'Pérdida - {title}')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
