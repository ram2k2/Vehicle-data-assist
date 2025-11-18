import csv
import io
import statistics
from typing import Dict, Any, List

REQUIRED_COLUMNS = ["Total distance (km)", "Fuel efficiency", "High voltage battery State of Health (SOH)", "Current vehicle speed"]

def analyze_vehicle_data(csv_content: str, filename: str) -> Dict[str, Any]:
    """
    Parses a semicolon-delimited CSV string and extracts key vehicle performance metrics.
    """
    csvfile = io.StringIO(csv_content)
    reader = csv.reader(csvfile, delimiter=';')
    
    try:
        header = next(reader)
        column_indices = {col: header.index(col) for col in REQUIRED_COLUMNS if col in header}
        if len(column_indices) != len(REQUIRED_COLUMNS):
            missing = [col for col in REQUIRED_COLUMNS if col not in column_indices]
            return {"error": f"Missing required columns: {', '.join(missing)}"}
    except Exception:
        return {"error": "Error reading CSV header or file is empty."}

    data_lists = {col: [] for col in REQUIRED_COLUMNS}
    for row in reader:
        for original_col in REQUIRED_COLUMNS:
            try:
                # Get the index and strip whitespace
                value = row[column_indices[original_col]].strip()
                # Check for "Not Value" or "Not Available" indicators
                if value.upper() not in ["NV", "NA", ""]:
                    data_lists[original_col].append(float(value))
            except (IndexError, ValueError):
                continue

    summary = {"filename": filename}
    
    try:
        total_distance_data = data_lists.get("Total distance (km)", [])
        
        # Total Distance: difference between last and first recorded distance
        summary["Total Distance Traveled (km)"] = (
            total_distance_data[-1] - total_distance_data[0] 
            if len(total_distance_data) >= 2 else 0
        )
        
        # Average Fuel Efficiency
        summary["Average Fuel Efficiency"] = (
            statistics.mean(data_lists["Fuel efficiency"]) 
            if data_lists["Fuel efficiency"] else 0
        )
        
        # Latest Battery SOH
        battery_soh_data = data_lists.get("High voltage battery State of Health (SOH)", [])
        summary["Latest Battery SOH (%)"] = (
            battery_soh_data[-1] 
            if battery_soh_data else 0
        )
        
        # Average Vehicle Speed
        summary["Average Vehicle Speed (km/h)"] = (
            statistics.mean(data_lists["Current vehicle speed"]) 
            if data_lists["Current vehicle speed"] else 0
        )
        
        return summary
    
    except Exception as e:
        return {"error": f"A calculation error occurred during summarization: {e}"}
```eof
