import argparse
import os
import sys
import multiprocessing
from model_evaluation import evaluate_models_with_comprehensive_visuals as evaluate_model_metrics

# Set start method for multiprocessing (safe for cross-platform)
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

from ocean_predictor import EnhancedOceanOrganismPredictor, initialize_predictor
from model_evaluation import evaluate_model_metrics  # Import our new module

# Hardcoded paths for ease of deployment
DATA_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/ocean-prediction/ocean_fauna_prediction_dataset.csv"
ORGANISMS_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/organisms/organism_data.csv"
BIODIVERSITY_PATH = "/Users/kreshars/Downloads/ocean fauna classifier/data/current stats/ocean_biodiversity.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "."

def main():
    parser = argparse.ArgumentParser(description='Enhanced Ocean Organism Predictor')
    parser.add_argument('--data', default=DATA_PATH, help='Path to ocean data CSV file')
    parser.add_argument('--organisms', default=ORGANISMS_PATH, help='Path to organism data CSV file')
    parser.add_argument('--biodiversity', default=BIODIVERSITY_PATH, help='Path to biodiversity data CSV file')
    parser.add_argument('--models-dir', default=MODELS_DIR, help='Directory to save/load models')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of models')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--evaluate', action='store_true', help='Generate model evaluation plots')

    args = parser.parse_args()

    try:
        print("Initializing Ocean Organism Predictor...")
        predictor = initialize_predictor(
            args.data,
            args.organisms,
            args.biodiversity,
            models_dir=args.models_dir,
            force_retrain=args.retrain
        )

        if not predictor:
            print("Failed to initialize predictor. Check error messages above.")
            return 1

        # If --evaluate flag is passed, generate evaluation plots and exit
        if args.evaluate:
            print("Generating model evaluation plots...")
            metrics_df = evaluate_model_metrics(predictor, args.models_dir, args.output)
            print(f"Evaluation complete. Plots saved to {os.path.join(args.output, 'evaluation_plots')}")
            return 0

        while True:
            try:
                print("\n=== Enhanced Ocean Organism Predictor ===")
                print("1. Predict for a specific location, year, and zone")
                print("2. Comprehensive analysis of all zones at a location")
                print("3. Generate model evaluation plots")
                print("4. Exit")

                option = input("\nSelect option (1-4): ").strip()

                if option == '4':
                    break
                elif option == '3':
                    print("Generating model evaluation plots...")
                    metrics_df = evaluate_model_metrics(predictor, args.models_dir, args.output)
                    print(f"Evaluation complete. Plots saved to {os.path.join(args.output, 'evaluation_plots')}")
                    continue

                latitude = float(input("Enter latitude (-90 to 90): "))
                longitude = float(input("Enter longitude (-180 to 180): "))
                year = int(input("Enter year (1900–2100): "))

                if option == '1':
                    print("\nAvailable pelagic zones:")
                    zones = list(predictor.pelagic_zones.keys())
                    for i, zone in enumerate(zones, 1):
                        print(f"{i}. {zone}")

                    zone_index = int(input("\nSelect pelagic zone (1-4): ")) - 1
                    selected_zone = zones[zone_index] if 0 <= zone_index < len(zones) else 'Epipelagic'
                    print(f"\nPredicting ocean conditions for {selected_zone} zone at {latitude}°, {longitude}° in {year}...")

                    predictions = predictor.predict_ocean_conditions(latitude, longitude, year, selected_zone)

                    if not predictions.empty:
                        pred_file = os.path.join(args.output, f"predictions_{selected_zone}_{latitude}_{longitude}_{year}.csv")
                        predictions.to_csv(pred_file, index=False)
                        print(f"Saved predictions to {pred_file}")

                        matching = predictor.find_suitable_organisms(predictions)
                        if not matching.empty:
                            org_file = os.path.join(args.output, f"matching_organisms_{selected_zone}_{latitude}_{longitude}_{year}.csv")
                            matching.to_csv(org_file, index=False)
                            print(f"Saved matching organisms to {org_file}")

                            print(f"\nFound {len(matching)} matching organisms:")
                            for i, (_, org) in enumerate(matching.iterrows(), 1):
                                print(f"{i}. {org['Species']}")
                                if org.get('source') == 'biodiversity':
                                    print("   (From biodiversity data)")
                                else:
                                    print(f"   Temp: {org['Temperature Min (°C)']}°C to {org['Temperature Max (°C)']}°C")
                                    print(f"   Salinity: {org['Salinity Min (ppt)']} to {org['Salinity Max (ppt)']} ppt")
                        else:
                            print("\nNo matching organisms found.")
                    else:
                        print("\nNo prediction data generated.")

                elif option == '2':
                    print(f"\nAnalyzing all pelagic zones at {latitude}°, {longitude}° in {year}...")
                    results, all_orgs = predictor.analyze_location(latitude, longitude, year)

                    if not all_orgs.empty:
                        summary_file = os.path.join(args.output, f"comprehensive_{latitude}_{longitude}_{year}.csv")
                        all_orgs.to_csv(summary_file, index=False)
                        print(f"Saved comprehensive analysis to {summary_file}")

                        print("\nSummary by zone:")
                        for zone, zone_data in results.items():
                            orgs = zone_data['organisms']
                            if not orgs.empty:
                                print(f"\n{zone}: {len(orgs)} species")
                                samples = orgs['Species'].head(5).tolist()
                                print(f"  Sample: {', '.join(samples)}")
                            else:
                                print(f"\n{zone}: No suitable organisms found")
                    else:
                        print("\nNo suitable organisms found across any zone.")

            except ValueError as ve:
                print(f"Input error: {ve}")
            except Exception as e:
                print(f"An error occurred: {e}")
                import traceback
                traceback.print_exc()

    except Exception as critical_error:
        print(f"Critical error: {critical_error}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup multiprocessing resources
        if 'multiprocessing' in sys.modules:
            try:
                multiprocessing.current_process()._clean()
            except Exception:
                pass
            import gc
            gc.collect()

    return 0

if __name__ == "__main__":
    exit(main())