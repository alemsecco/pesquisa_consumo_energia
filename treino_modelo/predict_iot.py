import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class EnergyPredictor:
    def __init__(self, model_path=None):
        """Initialize predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file. If None, will look in default location.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'treino_modelo', 'modelo', 'model.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = joblib.load(model_path)
        self.history = pd.DataFrame()  # Store recent predictions for lag features
        
    def _clean_numeric(self, value):
        """Clean numeric input that might come as string."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        # Handle string values
        try:
            value = str(value).replace('\xa0', '').replace(' ', '')
            value = value.replace('.', '').replace(',', '.')
            return float(value)
        except:
            return np.nan
    
    def recommend_sustainable(self, iot_data):
        """Generate renewable energy recommendations based on weather conditions."""
        reg = str(iot_data.get('regiao', '')).lower()
        precip = self._clean_numeric(iot_data.get('precipitacao', 0))
        temp = self._clean_numeric(iot_data.get('temperatura', 0))
        vento = self._clean_numeric(iot_data.get('vento', 0))
        
        recs = []
        if vento >= 6.0:
            recs.append('Eólica')
        if precip >= 200 and reg in ('norte', 'nordeste'):
            recs.append('Hidrelétrica / Pequenas Centrais Hidrelétricas (PCH)')
        if precip < 100 and temp >= 22:
            recs.append('Solar fotovoltaica')
        if not recs:
            recs.append('Solar fotovoltaica (padrão)')
            
        return '; '.join(recs)
    
    def update_history(self, date, regiao, consumo):
        """Update historical data with actual consumption values."""
        new_row = pd.DataFrame({
            'MesAno': [date.strftime('%Y-%m')],
            'Regiao': [regiao],
            'Consumo': [self._clean_numeric(consumo)]
        })
        self.history = pd.concat([self.history, new_row], ignore_index=True)
        # Keep only recent history needed for lag features
        self.history = self.history.sort_values('MesAno').drop_duplicates(['MesAno', 'Regiao'], keep='last')
        self.history = self.history.tail(24)  # Keep 2 years of history for lag12
    
    def _calculate_lags(self, date, regiao):
        """Calculate lag features from historical data."""
        if len(self.history) == 0:
            return np.nan, np.nan, np.nan
        
        hist = self.history.copy()
        hist['MesAno_dt'] = pd.to_datetime(hist['MesAno'] + '-01')
        hist = hist.sort_values('MesAno_dt')
        
        # Get values for this region
        region_hist = hist[hist['Regiao'] == regiao].copy()
        if len(region_hist) == 0:
            return np.nan, np.nan, np.nan
        
        target_date = pd.to_datetime(date.strftime('%Y-%m-01'))
        
        # Calculate lags
        last_month = region_hist[region_hist['MesAno_dt'] < target_date].tail(1)['Consumo'].iloc[0] if len(region_hist) > 0 else np.nan
        last_year = region_hist[region_hist['MesAno_dt'] == (target_date - pd.DateOffset(months=12))]['Consumo'].iloc[0] if len(region_hist) > 0 else np.nan
        
        # Rolling mean of last 3 months
        roll3 = region_hist[region_hist['MesAno_dt'] < target_date].tail(3)['Consumo'].mean() if len(region_hist) >= 3 else np.nan
        
        return last_month, last_year, roll3
    
    def predict(self, iot_data):
        """Make prediction using IoT sensor data.
        
        Args:
            iot_data: dict with keys:
                - regiao: string (Norte, Nordeste, Sul, Sudeste, Centro-Oeste)
                - temperatura: float (°C)
                - precipitacao: float (mm)
                - pressao: float (mB)
                - vento: float (m/s)
                - date: datetime object (optional, defaults to current date)
        
        Returns:
            dict with:
                - predicted_consumption: float
                - sustainable_recommendation: string
                - confidence: float (0-1)
        """
        date = iot_data.get('date', datetime.now())
        
        # Prepare features
        lag1, lag12, roll3 = self._calculate_lags(date, iot_data['regiao'])
        
        X = pd.DataFrame({
            'Ano': [date.year],
            'Mes': [date.month],
            'lag1': [lag1],
            'lag12': [lag12],
            'roll3': [roll3],
            'TEMPERATURA MEDIA, MENSAL (AUT)(°C)': [self._clean_numeric(iot_data['temperatura'])],
            'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)': [self._clean_numeric(iot_data['precipitacao'])],
            'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)': [self._clean_numeric(iot_data['pressao'])],
            'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)': [self._clean_numeric(iot_data['vento'])],
            'Regiao': [iot_data['regiao']]
        })
        
        # Make prediction
        try:
            pred = self.model.predict(X)[0]
            # Simple confidence based on missing values
            confidence = 1.0 - (X.isna().sum().sum() / len(X.columns))
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
        
        return {
            'predicted_consumption': float(pred),
            'sustainable_recommendation': self.recommend_sustainable(iot_data),
            'confidence': float(confidence)
        }


def example_usage():
    """Example of how to use the predictor."""
    # Initialize predictor
    predictor = EnergyPredictor()
    
    # Add some historical data (you would get this from your database)
    for month in range(1, 13):
        predictor.update_history(
            date=datetime(2024, month, 1),
            regiao='Nordeste',
            consumo=1200000  # example value
        )
    
    # Example IoT data
    iot_data = {
        'regiao': 'Nordeste',
        'temperatura': 25.5,
        'precipitacao': 150.0,
        'pressao': 1013.2,
        'vento': 3.5,
        'date': datetime(2025, 1, 1)
    }
    
    # Make prediction
    result = predictor.predict(iot_data)
    
    if result:
        print(f"\nPrediction for {iot_data['date'].strftime('%Y-%m')}:")
        print(f"Region: {iot_data['regiao']}")
        print(f"Predicted consumption: {result['predicted_consumption']:,.2f}")
        print(f"Sustainable recommendation: {result['sustainable_recommendation']}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == '__main__':
    example_usage()