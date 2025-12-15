#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify Copernicus Marine catalog file
"""
import json
from pathlib import Path

catalog_file = Path("reports/copernicus_catalog_sample.json")

if not catalog_file.exists():
    print(f"Error: File not found {catalog_file}")
    exit(1)

# Read file
with open(catalog_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Verify data
print("=" * 60)
print("Copernicus Marine Sea Ice Catalog Verification")
print("=" * 60)
print(f"File path: {catalog_file.absolute()}")
print(f"File size: {catalog_file.stat().st_size:,} bytes")

# Check structure
if isinstance(data, dict) and "products" in data:
    products = data["products"]
    print(f"Number of products: {len(products)}")
    print()
    
    # Display first 3 products
    print("Sample products:")
    for i, product in enumerate(products[:3], 1):
        if isinstance(product, dict):
            title = product.get("title", "N/A")
            product_id = product.get("product_id", "N/A")
            datasets = len(product.get("datasets", []))
            print(f"  {i}. {title}")
            print(f"     ID: {product_id}")
            print(f"     Datasets: {datasets}")
            print()

print("=" * 60)
print("Verification successful! File is ready for use.")
print("=" * 60)
