from flask import Flask, render_template, request, redirect, url_for
import os
import os.path
import pandas as pd
import signal
import sys
import atexit
import multiprocessing

# Configure multiprocessing to avoid resource leaks
# Must be done before importing from ocean_predictor
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from ocean_predictor import EnhancedOceanOrganismPredictor, initialize_predictor

app = Flask(__name__, 
           static_folder=os.path.abspath('./static'),
           static_url_path='/static')

# Configuration
DATA_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/ocean-prediction/ocean_fauna_prediction_dataset.csv"
ORGANISMS_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/organisms/organism_data.csv"
BIODIVERSITY_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/current stats/ocean_biodiversity.csv"
MODELS_DIR = "/Users/kreshars/models"

# Initialize predictor at module level but outside of request context
predictor = None

def initialize_app():
    global predictor
    if predictor is None:
        # Initialize predictor only once
        print("Initializing predictor...")
        predictor = initialize_predictor(
            DATA_PATH,
            ORGANISMS_PATH,
            BIODIVERSITY_PATH,
            models_dir=MODELS_DIR,
            force_retrain=False
        )
        print("Predictor initialized successfully.")

# Initialize when the module loads
initialize_app()

@app.route('/')
def index():
    return render_template('index.html', background='default_ocean.jpg')

@app.route('/specific-prediction', methods=['GET', 'POST'])
def specific_prediction():
    global predictor
    if predictor is None:
        initialize_app()
        
    if request.method == 'POST':
        # Process form data
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        year = int(request.form['year'])
        zone = request.form['zone']
        
        # Make prediction
        predictions = predictor.predict_ocean_conditions(latitude, longitude, year, zone)
        matching = predictor.find_suitable_organisms(predictions)
        
        # Determine background based on zone and location
        background = determine_background(zone, latitude)
        
        return render_template('results.html', 
                               predictions=predictions.to_dict('records'), 
                               organisms=matching.to_dict('records'),
                               zone=zone,
                               latitude=latitude,
                               longitude=longitude,
                               year=year,
                               background=background)
    
    # GET request: show the form - Add default background here
    zones = list(predictor.pelagic_zones.keys()) if predictor else []
    return render_template('specific_prediction.html', zones=zones, background='default_ocean.jpg')

@app.route('/comprehensive_analysis', methods=['GET', 'POST'])
def comprehensive_analysis():
    global predictor
    if predictor is None:
        initialize_app()
        
    if request.method == 'POST':
        try:
            # Process form data
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            year = int(request.form['year'])
            
            # Make prediction for all zones using the actual model
            results, all_organisms = predictor.analyze_location(latitude, longitude, year)
            
            # Determine background based on latitude
            background = determine_background('all', latitude)
            
            # Ensure we have data to render
            if results is None or (isinstance(results, dict) and len(results) == 0):
                return render_template('comprehensive_analysis.html',
                                      background='default_ocean.jpg',
                                      error_message="No results found for this location and year.")
            
            # Format the results properly for the template
            formatted_results = []
            
            # Process each zone's results
            for zone, zone_data in results.items():
                # Get predictions and organisms DataFrames
                predictions = zone_data.get('predictions', pd.DataFrame())
                organisms_df = zone_data.get('organisms', pd.DataFrame())
                
                # Calculate average temperature and salinity
                avg_temp = predictions['predicted_temperature'].mean().round(2) if not predictions.empty else 0.0
                avg_sal = predictions['predicted_salinity'].mean().round(2) if not predictions.empty else 0.0
                
                # Process organisms
                organisms = []
                if not organisms_df.empty:
                    # Convert DataFrame to list of dictionaries
                    organisms = organisms_df.to_dict('records')
                
                # Add this zone's data to formatted_results
                formatted_results.append({
                    'zone': zone,
                    'avg_temperature': avg_temp,
                    'avg_salinity': avg_sal,
                    'organisms': organisms,
                    'organism_count': len(organisms)
                })
            
            # Debug print to verify data
            print("Formatted results being sent to template:")
            for result in formatted_results:
                print(f"\nZone: {result['zone']}")
                print(f"Temperature: {result['avg_temperature']}°C")
                print(f"Salinity: {result['avg_salinity']} ppt")
                print(f"Organism count: {result['organism_count']}")
                print("Sample organisms:", [org.get('Species', 'Unknown') for org in result['organisms'][:3]])
            
            # Render template with our properly formatted results
            return render_template('comprehensive_results.html',
                                  results=formatted_results,
                                  latitude=latitude,
                                  longitude=longitude,
                                  year=year,
                                  background=background)
            
        except Exception as e:
            import traceback
            print(f"Error in comprehensive analysis: {str(e)}")
            print(traceback.format_exc())
            return render_template('comprehensive_analysis.html',
                                  background='default_ocean.jpg',
                                  error_message=f"Analysis failed: {str(e)}")
    
    # GET request: show the form
    return render_template('comprehensive_analysis.html', background='default_ocean.jpg')

def determine_background(zone, latitude):
    """Determine which background image to use based on latitude only"""
    # Based on latitude only
    if latitude > 60:
        return 'arctic.jpg'
    elif latitude < -60:
        return 'antarctic.jpg'
    elif latitude > 23.5:
        return 'temperate_north.jpg'
    elif latitude < -23.5:
        return 'temperate_south.jpg'
    elif latitude >= 0:
        return 'tropical_north.jpg'
    else:
        return 'tropical_south.jpg'

def cleanup_resources():
    """Clean up multiprocessing resources on shutdown"""
    print("Cleaning up multiprocessing resources...")
    if 'multiprocessing' in sys.modules:
        # Clean up any lingering processes
        for p in multiprocessing.active_children():
            try:
                p.terminate()
            except:
                pass
        # Clean up resources
        try:
            multiprocessing.current_process()._clean()
        except:
            pass

# Register cleanup function
atexit.register(cleanup_resources)

if __name__ == '__main__':
    try:
        # Use threaded=False and disable reloader to avoid multiprocessing issues
        app.run(debug=True, threaded=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        cleanup_resources()