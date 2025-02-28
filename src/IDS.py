import os
import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

# Ignorar warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Clase para cargar archivos pickle
class AnomalIA_IDS:
    def __init__(self, model_path, scaler_path, pca_path):
        for path in [model_path, scaler_path, pca_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error: El archivo '{path}' no existe.")

        self.model = self.load_pickle(model_path)
        self.scaler = self.load_pickle(scaler_path)
        self.pca = self.load_pickle(pca_path)

        if not hasattr(self.pca, "n_components_"):
            raise ValueError("El PCA cargado no tiene el atributo 'n_components_'.")

        self.n_components = self.pca.n_components_
        print(f"✅ PCA detectado con {self.n_components} componentes.")

        # Obtener características originales
        self.original_features = getattr(self.scaler, "feature_names_in_", None)

        if self.original_features is None:
            raise ValueError("❌ No se pudieron obtener las características originales del scaler.")

        print(f"✅ Se encontraron {len(self.original_features)} características usadas en el entrenamiento.")

        # Si el PCA tiene más de 10 componentes, reducimos usando los primeros 10
        if self.n_components > 10:
            print(f"⚠️ El PCA tiene {self.n_components} componentes, se usarán solo 10.")
            self.n_components = 10

    def load_pickle(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error al cargar '{file_path}': {e}")

    def predict_intrusion(self, data_dict):
        try:
            # Crear DataFrame con los nombres originales de entrenamiento
            data_df = pd.DataFrame([data_dict])

            # Asegurar que las columnas coincidan
            for col in self.original_features:
                if col not in data_df:
                    data_df[col] = 0  # Agregar las faltantes con 0

            data_df = data_df[self.original_features]  # Ordenar columnas

            # Transformación con scaler y PCA
            data_scaled = self.scaler.transform(data_df)
            data_pca = self.pca.transform(data_scaled)[:, :self.n_components]  # Usar solo 10 componentes

            df_pca = pd.DataFrame(data_pca, columns=[f'pc{i+1}' for i in range(self.n_components)])

            # Predicción
            predicted_bin = self.model["stacking_bin"].predict(df_pca)[0]
            predicted_multi = self.model["stacking_multi"].predict(df_pca)[0] if predicted_bin == 1 else "BENIGN"

            return predicted_bin, predicted_multi

        except Exception as e:
            print(f"❌ Error en la predicción: {e}")
            return None, None

# Función para extraer características de los paquetes
def extract_features(packet):
    """
    Extrae características específicas de un paquete para el modelo.
    """
    try:
        # Extrae más características del paquete
        features = [
            packet.time,  # Timestamp del paquete
            packet.proto if isinstance(packet.proto, int) else 0,  # Protocolo
            packet.sport if isinstance(packet.sport, int) else 0,  # Puerto de origen
            packet.dport if isinstance(packet.dport, int) else 0,  # Puerto de destino
            len(packet),  # Longitud del paquete
            packet.flags.value if hasattr(packet.flags, 'value') else 0,  # Flags del paquete
            packet.ttl if isinstance(packet.ttl, int) else 0,  # Tiempo de vida
            packet.dst,  # Dirección de destino
            packet.src,  # Dirección de origen
            packet.payload if packet.payload else 0,  # Payload
            # Aquí puedes agregar más características según el análisis de los paquetes
        ]
        
        return features
    except Exception as e:
        print(f"Error al extraer características: {e}")
        return [0] * 70  # Retorna un conjunto de 70 características (ajustable según tu modelo)

# Función para procesar los paquetes y realizar la predicción
def packet_callback(packet):
    """
    Procesa cada paquete capturado y realiza la predicción de anomalía.
    """
    try:
        # Extraer características del paquete
        features = extract_features(packet)
        
        # Realizar la predicción utilizando el modelo IDS
        prediction = anomaly_detector.predict_intrusion(features)
        
        # Imprimir el resultado
        if prediction[0] == 0:
            print(f"Paquete Benigno: {packet.summary()}")
        else:
            print(f"Posible Ataque Detectado: {packet.summary()}")
    
    except Exception as e:
        print(f"Error al procesar el paquete: {e}")

# Rutas de los archivos
model_path = "hybrid_lightgbm_lr_model.pkl"
scaler_path = "scaler.pkl"
pca_path = "pca.pkl"

# Crear la instancia del IDS con el modelo entrenado
anomaly_detector = AnomalIA_IDS(model_path, scaler_path, pca_path)

# Comenzar la captura de paquetes en tiempo real
print("Comenzando la captura de paquetes...")
sniff(prn=packet_callback, store=0)