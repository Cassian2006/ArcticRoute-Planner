import geopandas as gpd

file_path = "ArcticRoute/reports/phaseE/center/prior_centerlines_all.geojson"

try:
    gdf = gpd.read_file(file_path)

    print(f"--- Analyzing {file_path} ---")
    print(f"CRS of the GeoDataFrame: {gdf.crs}")

    # Check the bounds to understand the coordinate range
    bounds = gdf.total_bounds
    print(f"Total bounds (minx, miny, maxx, maxy): {bounds}")

    # Print a sample of the coordinates to see their format
    if not gdf.empty:
        print("\nSample coordinates from the first geometry:")
        geom = gdf.geometry.iloc[0]
        if geom.geom_type == 'LineString':
            print(list(geom.coords)[:5])
        elif geom.geom_type == 'MultiLineString':
            print(list(geom.geoms[0].coords)[:5])

except Exception as e:
    print(f"An error occurred: {e}")

