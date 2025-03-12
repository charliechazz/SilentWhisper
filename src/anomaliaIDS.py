import pickle
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.decomposition import PCA

warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Rutas de los archivos
model_path = "hybrid_lightgbm_lr_model.pkl"
scaler_path = "scaler.pkl"
pca_path = "pca.pkl"

# Datos de prueba
data = {
    'flow iat mean': 0, 'fwd iat total': 0, 'bwd iat min': 0, 
    'packet length variance': 0, 'active std': 0, 'flow duration': 0, 
    'bwd iat total': 0, 'fwd packet length std': 0, 'bwd iat max': 0, 
    'flow packets/s': 0
}

ids = AnomalIA_IDS(model_path, scaler_path, pca_path)
pred_bin, pred_multi = ids.predict_intrusion(data)
print(f"🔍 Predicción binaria: {pred_bin}, Tipo de Ataque: {pred_multi}")