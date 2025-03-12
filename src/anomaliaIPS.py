import joblib # type: ignore
#from sklearn.ensemble import RandomForestClassifier

class AnomalIA_IPS:
    def __init__(self, binary_model, multi_model):

        # Modelo de clasificación binaria;
        # 0: Actividad normal; 1: Actividad anómala 
        self.binary_model = joblib.load(binary_model)

        # Modelo de clasificación multiclase;
        # Tipo de ataque [0, 9]:
        self.multi_model = joblib.load(multi_model)
    
    def predict_anomaly(self, data_dict):
        try:
            data_array = [[data_dict[key] for key in sorted(data_dict.keys())]]
            predicted_class = self.binary_model.predict(data_array)
            return predicted_class[0]
        except:
            return None
    
    def predict_attack(self, data_dict):
        try:
            data_array = [[data_dict[key] for key in sorted(data_dict.keys())]]
            predicted_class = self.multi_model.predict(data_array)
            return predicted_class[0]
        except:
            return None


data = {
    "sttl": 254,
    "state_INT": 0,
    "ct_state_ttl": 0,
    "proto_tcp": 1,
    "swin": 255,
    "dload": 13446.84766,
    "state_CON": 0,
    "dwin": 255,
    "state_FIN": 1
}

ia = AnomalIA_IPS(
    "binary_ensemble_model.pkl",
    "boosting_ensemble_multimodel.pkl"
)

print(ia.predict_anomaly(data))
print(ia.predict_attack(data))