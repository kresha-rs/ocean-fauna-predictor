import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_climate_zone(latitude):
    if latitude > 66.5:
        return "Arctic"
    elif latitude > 23.5:
        return "Temperate North"
    elif latitude > 0:
        return "Tropical North"
    elif latitude > -23.5:
        return "Tropical South"
    elif latitude > -66.5:
        return "Temperate South"
    else:
        return "Antarctic"

def get_pelagic_zone(depth, zone_boundaries):
    if depth < zone_boundaries['Epipelagic'][1]:
        return 'Epipelagic'
    elif depth < zone_boundaries['Mesopelagic'][1]:
        return 'Mesopelagic'
    elif depth < zone_boundaries['Bathypelagic'][1]:
        return 'Bathypelagic'
    else:
        return 'Abyssopelagic'

def train_advanced_model(X, y, climate_zone=None):
    """Train model and capture performance metrics"""
    if len(X) < 20:
        avg_value = y.mean()
        return {
            'model_type': 'avg',
            'avg_value': avg_value,
            'train_metrics': {
                'mse': 0,
                'r2': 0
            },
            'test_metrics': {
                'mse': 0,
                'r2': 0
            }
        }
    
    # Split data into train and test sets
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Try Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred_train_rf = rf_model.predict(X_train)
        y_pred_test_rf = rf_model.predict(X_test)
        
        rf_train_mse = mean_squared_error(y_train, y_pred_train_rf)
        rf_test_mse = mean_squared_error(y_test, y_pred_test_rf)
        
        rf_train_r2 = r2_score(y_train, y_pred_train_rf)
        rf_test_r2 = r2_score(y_test, y_pred_test_rf) if len(np.unique(y_test)) > 1 else 0
        
        # Try Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        
        y_pred_train_linear = linear_model.predict(X_train)
        y_pred_test_linear = linear_model.predict(X_test)
        
        linear_train_mse = mean_squared_error(y_train, y_pred_train_linear)
        linear_test_mse = mean_squared_error(y_test, y_pred_test_linear)
        
        linear_train_r2 = r2_score(y_train, y_pred_train_linear)
        linear_test_r2 = r2_score(y_test, y_pred_test_linear) if len(np.unique(y_test)) > 1 else 0
        
        # Select best model based on test MSE
        if rf_test_mse < linear_test_mse:
            return {
                'model_type': 'rf',
                'model': rf_model,
                'train_metrics': {
                    'mse': rf_train_mse,
                    'r2': rf_train_r2
                },
                'test_metrics': {
                    'mse': rf_test_mse,
                    'r2': rf_test_r2
                }
            }
        else:
            return {
                'model_type': 'linear',
                'model': linear_model,
                'train_metrics': {
                    'mse': linear_train_mse,
                    'r2': linear_train_r2
                },
                'test_metrics': {
                    'mse': linear_test_mse,
                    'r2': linear_test_r2
                }
            }
    except Exception as e:
        # Fallback to simple average model
        avg_value = y.mean()
        return {
            'model_type': 'avg',
            'avg_value': avg_value,
            'train_metrics': {
                'mse': 0,
                'r2': 0
            },
            'test_metrics': {
                'mse': 0,
                'r2': 0
            }
        }

class EnhancedOceanOrganismPredictor:
    def __init__(self):
        self.temp_data = pd.DataFrame()
        self.sal_data = pd.DataFrame()
        self.organism_data = pd.DataFrame()
        self.biodiversity_data = pd.DataFrame()
        self.pelagic_zones = {
            'Epipelagic': (0, 200),
            'Mesopelagic': (200, 1000),
            'Bathypelagic': (1000, 4000),
            'Abyssopelagic': (4000, 11000)
        }
        self.climate_zone_temp_models = {}
        self.climate_zone_sal_models = {}
        self.geographic_biodiversity = {}
        self.models_initialized = False
    
    def load_ocean_data(self, csv_file):
        if not os.path.exists(csv_file):
            raise ValueError(f"Data file not found: {csv_file}")
            
        print(f"Loading ocean data from {csv_file}...")
        
        df_headers = pd.read_csv(csv_file, nrows=0)
        columns_to_use = ['LATITUDE', 'LONGITUDE', 'latitude', 'longitude', 'year', 'YEAR']
        
        temp_cols = [col for col in df_headers.columns if 'temperature' in col.lower()]
        sal_cols = [col for col in df_headers.columns if 'salinity' in col.lower()]
        
        columns_to_use.extend(temp_cols)
        columns_to_use.extend(sal_cols)
        
        columns_to_use = [col for col in columns_to_use if col in df_headers.columns]
        
        print(f"Reading {len(columns_to_use)} of {len(df_headers.columns)} columns...")
        df = pd.read_csv(csv_file, usecols=columns_to_use)
        print(f"Loaded ocean data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        df.columns = [col.replace('AND VALUES AT DEPTHS (M):', '') for col in df.columns]
        
        data_dict = self._organize_data(df)
        
        return data_dict
    
    def _organize_data(self, df):
        data_dict = {}
        
        if 'LATITUDE' in df.columns and 'latitude' not in df.columns:
            df['latitude'] = df['LATITUDE']
        if 'LONGITUDE' in df.columns and 'longitude' not in df.columns:
            df['longitude'] = df['LONGITUDE']
        if 'YEAR' in df.columns and 'year' not in df.columns:
            df['year'] = df['YEAR']
            
        if 'year' not in df.columns:
            temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
            year_info = [col for col in temp_cols if '_temperature_' in col]
            
            if year_info:
                col = year_info[0]
                years_str = col.split('_temperature_')[1]
                years = years_str.split('_')
                if len(years) == 2:
                    start_year = int(years[0])
                    end_year = int(years[1])
                    avg_year = (start_year + end_year) // 2
                    df['year'] = avg_year
                else:
                    df['year'] = 2023
            else:
                df['year'] = 2023
        
        df['climate_zone'] = df['latitude'].apply(get_climate_zone)
        
        temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
        if temp_cols:
            print("Processing temperature data...")
            temp_df = self._extract_parameter_data(df, temp_cols, 'temperature')
            if not temp_df.empty:
                data_dict['temperature'] = temp_df
                self.temp_data = temp_df
                print(f"Processed {len(temp_df)} temperature readings")
            else:
                raise ValueError("No valid temperature data could be extracted from the file")
        
        sal_cols = [col for col in df.columns if 'salinity' in col.lower()]
        if sal_cols:
            print("Processing salinity data...")
            sal_df = self._extract_parameter_data(df, sal_cols, 'salinity')
            if not sal_df.empty:
                data_dict['salinity'] = sal_df
                self.sal_data = sal_df
                print(f"Processed {len(sal_df)} salinity readings")
            else:
                raise ValueError("No valid salinity data could be extracted from the file")
            
        return data_dict
    
    def _extract_parameter_data(self, df, param_cols, param_name):
        result_data = []
        
        chunk_size = 5000
        total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        
        with tqdm(total=total_chunks, desc=f"Processing {param_name} data") as pbar:
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                for col in param_cols:
                    if 'AND VALUES AT DEPTHS' in col:
                        continue
                    
                    if f'm_{param_name}' in col:
                        try:
                            depth_str = col.split(f'm_{param_name}')[0]
                            depth = int(depth_str)
                            
                            valid_rows = chunk_df[pd.notna(chunk_df[col])]
                            
                            if not valid_rows.empty:
                                temp_result = pd.DataFrame({
                                    'latitude': valid_rows['latitude'],
                                    'longitude': valid_rows['longitude'],
                                    'year': valid_rows['year'],
                                    'climate_zone': valid_rows['climate_zone'],
                                    'depth': depth,
                                    param_name: valid_rows[col]
                                })
                                
                                result_data.append(temp_result)
                        except (ValueError, KeyError) as e:
                            continue
                
                pbar.update(1)
        
        if result_data:
            result_df = pd.concat(result_data, ignore_index=True)
            result_df['pelagic_zone'] = result_df['depth'].apply(
                lambda d: get_pelagic_zone(d, self.pelagic_zones)
            )
            return result_df
        
        return pd.DataFrame()
    
    def load_organism_data(self, csv_file):
        if not os.path.exists(csv_file):
            raise ValueError(f"Organism data file not found: {csv_file}")
            
        print(f"Loading organism data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        required_cols = ['Species', 'Temperature Min (°C)', 'Temperature Max (°C)', 
                      'Salinity Min (ppt)', 'Salinity Max (ppt)', 'pelagic_zone']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in organism data: {missing_cols}")
        
        for col in required_cols[1:5]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df[required_cols[1:5]].isna().all().any():
            raise ValueError("Some columns contain only non-numeric values")
        
        df['pelagic_zone'] = df['pelagic_zone'].str.capitalize()
        
        self.organism_data = df
        print(f"Loaded organism data with {df.shape[0]} species")
        return df
    
    def load_biodiversity_data(self, csv_file):
        if not os.path.exists(csv_file):
            print(f"Warning: Biodiversity data file not found: {csv_file}")
            print("Proceeding without biodiversity data...")
            return None
            
        print(f"Loading biodiversity data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        required_cols = ['latitude', 'longitude', 'climate_zone', 
                      'epipelagic_species', 'mesopelagic_species', 
                      'bathypelagic_species', 'abyssopelagic_species']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['latitude', 'longitude']:
                    raise ValueError(f"Missing required column in biodiversity data: {col}")
                else:
                    df[col] = None
        
        df['climate_zone'] = df['climate_zone'].fillna(df['latitude'].apply(get_climate_zone))
        
        self.biodiversity_data = df
        
        for _, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            zone = row['climate_zone']
            
            key = f"{zone}"
            if key not in self.geographic_biodiversity:
                self.geographic_biodiversity[key] = {
                    'Epipelagic': set(),
                    'Mesopelagic': set(),
                    'Bathypelagic': set(),
                    'Abyssopelagic': set()
                }
            
            if pd.notna(row['epipelagic_species']):
                species = [s.strip() for s in str(row['epipelagic_species']).split(';')]
                self.geographic_biodiversity[key]['Epipelagic'].update(species)
            
            if pd.notna(row['mesopelagic_species']):
                species = [s.strip() for s in str(row['mesopelagic_species']).split(';')]
                self.geographic_biodiversity[key]['Mesopelagic'].update(species)
                
            if pd.notna(row['bathypelagic_species']):
                species = [s.strip() for s in str(row['bathypelagic_species']).split(';')]
                self.geographic_biodiversity[key]['Bathypelagic'].update(species)
                
            if pd.notna(row['abyssopelagic_species']):
                species = [s.strip() for s in str(row['abyssopelagic_species']).split(';')]
                self.geographic_biodiversity[key]['Abyssopelagic'].update(species)
        
        print(f"Loaded biodiversity data with {len(df)} geographic regions")
        return df
    
    def train_models(self):
        if self.temp_data.empty or self.sal_data.empty:
            raise ValueError("No data available for training. Please load data first.")
        
        print("\nTraining temperature and salinity prediction models...")
        
        temp_models = {}
        sal_models = {}
        
        for climate_zone, temp_group in self.temp_data.groupby('climate_zone'):
            for pelagic_zone, zone_group in temp_group.groupby('pelagic_zone'):
                X = zone_group[['latitude', 'longitude', 'year', 'depth']].values
                y = zone_group['temperature'].values
                
                key = f"{climate_zone}_{pelagic_zone}"
                
                if len(X) > 10:
                    model = train_advanced_model(X, y, climate_zone)
                    temp_models[key] = model
                    print(f"Temperature model for {climate_zone} - {pelagic_zone} trained")
        
        for climate_zone, sal_group in self.sal_data.groupby('climate_zone'):
            for pelagic_zone, zone_group in sal_group.groupby('pelagic_zone'):
                X = zone_group[['latitude', 'longitude', 'year', 'depth']].values
                y = zone_group['salinity'].values
                
                key = f"{climate_zone}_{pelagic_zone}"
                
                if len(X) > 10:
                    model = train_advanced_model(X, y, climate_zone)
                    sal_models[key] = model
                    print(f"Salinity model for {climate_zone} - {pelagic_zone} trained")
        
        self.climate_zone_temp_models = temp_models
        self.climate_zone_sal_models = sal_models
        self.models_initialized = True
        
        return True
    
    def save_models(self, models_dir):
        """Save trained models to disk"""
        if not self.models_initialized:
            raise ValueError("Models not trained. Please train models first.")
        
        # Ensure directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Save temperature models
        temp_models_path = os.path.join(models_dir, 'temp_models.pkl')
        with open(temp_models_path, 'wb') as f:
            pickle.dump(self.climate_zone_temp_models, f)
        
        # Save salinity models
        sal_models_path = os.path.join(models_dir, 'sal_models.pkl')
        with open(sal_models_path, 'wb') as f:
            pickle.dump(self.climate_zone_sal_models, f)
        
        # Save biodiversity data
        bio_path = os.path.join(models_dir, 'biodiversity.pkl')
        with open(bio_path, 'wb') as f:
            pickle.dump(self.geographic_biodiversity, f)
            
        print(f"Models saved to {models_dir}")
        return True
    
    def load_models(self, models_dir):
        """Load trained models from disk"""
        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"Models directory {models_dir} not found. Models need training.")
            return False
        
        # Paths to model files
        temp_models_path = os.path.join(models_dir, 'temp_models.pkl')
        sal_models_path = os.path.join(models_dir, 'sal_models.pkl')
        bio_path = os.path.join(models_dir, 'biodiversity.pkl')
        
        # Check if model files exist
        if not (os.path.exists(temp_models_path) and os.path.exists(sal_models_path)):
            print("Model files not found. Models need training.")
            return False
        
        try:
            # Load temperature models
            with open(temp_models_path, 'rb') as f:
                self.climate_zone_temp_models = pickle.load(f)
            
            # Load salinity models
            with open(sal_models_path, 'rb') as f:
                self.climate_zone_sal_models = pickle.load(f)
            
            # Load biodiversity data if available
            if os.path.exists(bio_path):
                with open(bio_path, 'rb') as f:
                    self.geographic_biodiversity = pickle.load(f)
            
            self.models_initialized = True
            print(f"Models loaded from {models_dir}")
            return True
        
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_ocean_conditions(self, latitude, longitude, year, zone):
        if zone not in self.pelagic_zones:
            raise ValueError(f"Invalid pelagic zone. Must be one of {list(self.pelagic_zones.keys())}")
        
        if not self.models_initialized:
            raise ValueError("Models not initialized. Please train or load models first.")
        
        if latitude < -90 or latitude > 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        if year < 1900 or year > 2200:
            raise ValueError("Year must be between 1900 and 2200")
        
        climate_zone = get_climate_zone(latitude)
        zone_min_depth, zone_max_depth = self.pelagic_zones[zone]
        
        if zone == 'Epipelagic':
            depths = [0, 50, 100, 150, 190]
        elif zone == 'Mesopelagic':
            depths = [210, 400, 600, 800, 990]
        elif zone == 'Bathypelagic':
            depths = [1010, 1500, 2000, 3000, 3990]
        else:
            depths = [4010, 5000, 6000, 8000, 10000]
        
        predictions = []
        
        for depth in depths:
            X = np.array([[latitude, longitude, year, depth]])
            
            key = f"{climate_zone}_{zone}"
            fallback_key = f"Temperate North_{zone}"
            
            if key in self.climate_zone_temp_models:
                temp_pred = predict_with_model(self.climate_zone_temp_models[key], X)[0]
            elif fallback_key in self.climate_zone_temp_models:
                temp_pred = predict_with_model(self.climate_zone_temp_models[fallback_key], X)[0]
            else:
                temp_model_keys = [k for k in self.climate_zone_temp_models.keys() if k.endswith(f"_{zone}")]
                if temp_model_keys:
                    temp_pred = predict_with_model(self.climate_zone_temp_models[temp_model_keys[0]], X)[0]
                else:
                    temp_pred = 15.0 if zone == 'Epipelagic' else 10.0 if zone == 'Mesopelagic' else 5.0 if zone == 'Bathypelagic' else 2.0
            
            if key in self.climate_zone_sal_models:
                sal_pred = predict_with_model(self.climate_zone_sal_models[key], X)[0]
            elif fallback_key in self.climate_zone_sal_models:
                sal_pred = predict_with_model(self.climate_zone_sal_models[fallback_key], X)[0]
            else:
                sal_model_keys = [k for k in self.climate_zone_sal_models.keys() if k.endswith(f"_{zone}")]
                if sal_model_keys:
                    sal_pred = predict_with_model(self.climate_zone_sal_models[sal_model_keys[0]], X)[0]
                else:
                    sal_pred = 35.0
            
            if climate_zone == "Antarctic":
                if zone == "Epipelagic":
                    temp_pred = min(temp_pred, 5.0)
                else:
                    temp_pred = min(temp_pred, 3.0)
            elif climate_zone == "Arctic":
                if zone == "Epipelagic":
                    temp_pred = min(temp_pred, 10.0)
                else:
                    temp_pred = min(temp_pred, 5.0)
            elif "Tropical" in climate_zone:
                if zone == "Epipelagic":
                    temp_pred = max(temp_pred, 20.0)
                elif zone == "Mesopelagic":
                    temp_pred = max(temp_pred, 10.0)
            
            predictions.append({
                'latitude': latitude,
                'longitude': longitude,
                'year': year,
                'depth': depth,
                'predicted_temperature': temp_pred,
                'predicted_salinity': sal_pred,
                'pelagic_zone': zone,
                'climate_zone': climate_zone
            })
        
        return pd.DataFrame(predictions)
    
    def find_suitable_organisms(self, zone_predictions):
        if self.organism_data.empty:
            raise ValueError("No organism data available.")
            
        avg_temp = zone_predictions['predicted_temperature'].mean()
        avg_sal = zone_predictions['predicted_salinity'].mean()
        zone = zone_predictions['pelagic_zone'].iloc[0]
        climate_zone = zone_predictions['climate_zone'].iloc[0]
        year = zone_predictions['year'].iloc[0]
        
        print(f"\nPredicted conditions for {zone} in {climate_zone} ({year}):")
        print(f"  Average Temperature: {avg_temp:.2f}°C")
        print(f"  Average Salinity: {avg_sal:.2f} ppt")
        
        temp_min_col = 'Temperature Min (°C)'
        temp_max_col = 'Temperature Max (°C)'
        sal_min_col = 'Salinity Min (ppt)'
        sal_max_col = 'Salinity Max (ppt)'
        
        tolerance = 2.0
        
        matching_organisms = self.organism_data[
            (self.organism_data[temp_min_col] - tolerance <= avg_temp) & 
            (self.organism_data[temp_max_col] + tolerance >= avg_temp) &
            (self.organism_data[sal_min_col] - tolerance <= avg_sal) &
            (self.organism_data[sal_max_col] + tolerance >= avg_sal) &
            (self.organism_data['pelagic_zone'].str.lower() == zone.lower())
        ]
        
        biodiversity_enriched = self._enrich_with_biodiversity_data(matching_organisms, climate_zone, zone)
        
        return biodiversity_enriched
    
    def _enrich_with_biodiversity_data(self, matching_df, climate_zone, pelagic_zone):
        if matching_df.empty and self.geographic_biodiversity:
            key = f"{climate_zone}"
            biodiversity_species = []
            
            if key in self.geographic_biodiversity and pelagic_zone in self.geographic_biodiversity[key]:
                species_set = self.geographic_biodiversity[key][pelagic_zone]
                species_list = list(species_set)
                
                if species_list:
                    biodiversity_df = pd.DataFrame({
                        'Species': species_list,
                        'Temperature Min (°C)': [0] * len(species_list),
                        'Temperature Max (°C)': [30] * len(species_list),
                        'Salinity Min (ppt)': [20] * len(species_list),
                        'Salinity Max (ppt)': [40] * len(species_list),
                        'pelagic_zone': [pelagic_zone] * len(species_list),
                        'source': ['biodiversity'] * len(species_list)
                    })
                    
                    if not matching_df.empty:
                        matching_df['source'] = 'organism_data'
                        return pd.concat([matching_df, biodiversity_df], ignore_index=True)
                    else:
                        return biodiversity_df
        
        if not matching_df.empty:
            matching_df['source'] = 'organism_data'
        
        return matching_df
    
    def analyze_location(self, latitude, longitude, year=2025):
        """Comprehensive analysis of all pelagic zones at a location"""
        climate_zone = get_climate_zone(latitude)
        
        print(f"\nAnalyzing ocean at {latitude}°, {longitude}° in {year} ({climate_zone})")
        
        results = {}
        all_organisms = []
        
        for zone in self.pelagic_zones.keys():
            try:
                predictions = self.predict_ocean_conditions(latitude, longitude, year, zone)
                organisms = self.find_suitable_organisms(predictions)
                
                results[zone] = {
                    'predictions': predictions,
                    'organisms': organisms
                }
                
                if not organisms.empty:
                    zone_df = organisms.copy()
                    zone_df['pelagic_zone'] = zone
                    all_organisms.append(zone_df)
            except Exception as e:
                print(f"Error analyzing {zone}: {e}")
        
        if all_organisms:
            all_organisms_df = pd.concat(all_organisms, ignore_index=True)
            return results, all_organisms_df
        else:
            return results, pd.DataFrame()


def initialize_predictor(ocean_data_file, organism_data_file, biodiversity_file=None, models_dir='models', force_retrain=False):
    """
    Initialize the ocean predictor with data and models
    Returns the predictor object or None if initialization fails
    """
    try:
        predictor = EnhancedOceanOrganismPredictor()
        
        # Load organism data first as it's smaller and critical
        predictor.load_organism_data(organism_data_file)
        
        # Try to load models from disk
        models_loaded = False
        if not force_retrain:
            models_loaded = predictor.load_models(models_dir)
        
        # If models were not loaded or force_retrain is True, train new models
        if not models_loaded or force_retrain:
            # Load ocean data (needed for training)
            predictor.load_ocean_data(ocean_data_file)
            
            # Load biodiversity data if provided
            if biodiversity_file:
                predictor.load_biodiversity_data(biodiversity_file)
            
            # Train models
            print("Training new prediction models...")
            predictor.train_models()
            
            # Save models for future use
            predictor.save_models(models_dir)
        else:
            # If biodiversity file was provided but models were loaded from disk,
            # still load biodiversity data since it might have changed
            if biodiversity_file:
                predictor.load_biodiversity_data(biodiversity_file)
        
        return predictor
    
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        return None