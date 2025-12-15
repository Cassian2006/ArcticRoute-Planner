#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use copernicusmarine library to fetch catalog data
"""
import json
from pathlib import Path
from copernicusmarine import describe

print("Fetching Copernicus Marine catalog data...")

try:
    # Call describe function with contains as a list
    catalog = describe(contains=["sea ice"])
    
    # Convert to dict using Pydantic's model_dump
    catalog_dict = catalog.model_dump()
    
    # Save as JSON
    output_file = Path("reports/copernicus_catalog_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(catalog_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Success! File saved to: {output_file.absolute()}")
    print(f"  File size: {output_file.stat().st_size:,} bytes")
    
    # Display summary
    if isinstance(catalog_dict, dict) and "products" in catalog_dict:
        products = catalog_dict["products"]
        print(f"  Number of products: {len(products)}")
        for i, product in enumerate(products[:3], 1):
            if isinstance(product, dict):
                title = product.get("title", "N/A")
                print(f"    {i}. {title}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
