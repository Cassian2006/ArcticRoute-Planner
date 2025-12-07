"""Converts the prior centerlines file to WGS84 coordinate order and writes corrected GeoJSON.

@role: pipeline
"""

"""
This script corrects the coordinate order in the prior centerlines GeoJSON file.

The original file 'prior_centerlines_all.geojson' was found to have latitude and longitude
swapped, leading to incorrect map rendering. This script reads the original file,
swaps the coordinates for each point in every geometry, and saves the corrected
data to a new file with the '_wgs84' suffix, ensuring it conforms to the standard
(longitude, latitude) order expected by tools like Folium and GeoPandas.
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


def swap_coords(geom):
    """Swaps the x and y coordinates of a geometry."""
    if geom.is_empty:
        return geom

    if geom.geom_type == 'LineString':
        return LineString([(y, x) for x, y in geom.coords])
    elif geom.geom_type == 'MultiLineString':
        return MultiLineString([LineString([(y, x) for x, y in line.coords]) for line in geom.geoms])
    else:
        # Return other geometry types unchanged
        return geom

def main():
    # Define file paths
    # Note: The original file is in the 'reports' directory, which is a non-standard location.
    # We will read from there but write the corrected file to a more appropriate 'data_processed' location.
    source_path = "ArcticRoute/reports/phaseE/center/prior_centerlines_all.geojson"
    output_path = "ArcticRoute/data_processed/prior/prior_centerlines_all_wgs84.geojson"

    print(f"Reading source file: {source_path}")
    try:
        gdf = gpd.read_file(source_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # The CRS is already marked as EPSG:4326, which is correct. The issue is the coordinate order.
    print(f"Original CRS: {gdf.crs}")
    print("Swapping (latitude, longitude) to (longitude, latitude)...")

    # Apply the coordinate swapping function to the geometry column
    gdf['geometry'] = gdf['geometry'].apply(swap_coords)

    # Set the CRS for the new GeoDataFrame to ensure it's correctly maintained
    gdf.crs = "EPSG:4326"

    print(f"Saving corrected file to: {output_path}")
    try:
        # Ensure the output directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        gdf.to_file(output_path, driver='GeoJSON')
        print("Successfully created the WGS84-compliant GeoJSON file.")

        # Verification step
        print("\nVerifying the new file...")
        new_gdf = gpd.read_file(output_path)
        print(f"New file CRS: {new_gdf.crs}")
        bounds = new_gdf.total_bounds
        print(f"New total bounds (minx, miny, maxx, maxy): {bounds}")
        if not new_gdf.empty:
            print("Sample of new coordinates:")
            print(list(new_gdf.geometry.iloc[0].coords)[:5])

    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()

