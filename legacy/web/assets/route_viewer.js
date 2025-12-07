/* Leaflet route viewer helper */

async function loadJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    throw new Error(`Failed to load ${path}: ${resp.status}`);
  }
  return resp.json();
}

async function loadGeoJSON(path) {
  try {
    return await loadJSON(path);
  } catch (err) {
    console.warn(`[RouteViewer] geojson load failed: ${path}`, err);
    return null;
  }
}

async function initRouteViewer(options) {
  const {
    mapId,
    overlayConfigPath,
    overlayImagePath,
    routesGlob,
    hotspotsPath,
    enableHeatmap = true,
  } = options;

  const map = L.map(mapId, {
    preferCanvas: true,
  });

  const baseLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 18,
  }).addTo(map);

  const overlayLayers = {};
  const controlLayers = {
    "Basemap": baseLayer,
  };

  try {
    const overlayConfig = await loadJSON(overlayConfigPath);
    const bounds = overlayConfig.bounds;
    const imgOverlay = L.imageOverlay(overlayImagePath, bounds, { opacity: 0.4, interactive: false });
    imgOverlay.addTo(map);
    overlayLayers["Risk Overlay"] = imgOverlay;
  } catch (err) {
    console.warn("[RouteViewer] risk overlay unavailable", err);
  }

  let routesBounds = null;
  const routeLayers = {};
  for (const entry of routesGlob) {
    const geojson = await loadGeoJSON(entry.path);
    if (!geojson) {
      continue;
    }
    const layer = L.geoJSON(geojson, {
      style: {
        color: entry.color || "#ff0000",
        weight: 3,
      },
    });
    routeLayers[entry.name] = layer;
    overlayLayers[entry.name] = layer;
    layer.addTo(map);

    const layerBounds = layer.getBounds();
    if (layerBounds.isValid()) {
      routesBounds = routesBounds ? routesBounds.extend(layerBounds) : layerBounds;
    }
  }

  let hotspotLayer = null;
  let hotspotHeatLayer = null;
  try {
    const hotspots = await loadGeoJSON(hotspotsPath);
    if (hotspots && hotspots.features && hotspots.features.length > 0) {
      const heatPoints = [];
      if (enableHeatmap) {
        for (const feature of hotspots.features) {
          if (feature.geometry && feature.geometry.type === "Point") {
            const [lon, lat] = feature.geometry.coordinates;
            const weight = feature.properties && feature.properties.value ? feature.properties.value : 1.0;
            heatPoints.push([lat, lon, weight]);
          }
        }
      }

      hotspotLayer = L.geoJSON(hotspots, {
        pointToLayer: (feature, latlng) =>
          L.circleMarker(latlng, {
            radius: 5,
            fillColor: "#ffa500",
            color: "#cc7000",
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.8,
          }),
        onEachFeature: (feature, layer) => {
          if (feature.properties && feature.properties.value !== undefined) {
            layer.bindPopup(`Hotspot value: ${feature.properties.value}`);
          }
        },
      }).addTo(map);
      overlayLayers["Accident Hotspots"] = hotspotLayer;

      if (enableHeatmap && heatPoints.length > 0 && window.L && window.L.heatLayer) {
        hotspotHeatLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15, minOpacity: 0.3 });
        overlayLayers["Hotspot Heatmap"] = hotspotHeatLayer;
      }
    }
  } catch (err) {
    console.warn("[RouteViewer] hotspots unavailable", err);
  }

  L.control.layers(controlLayers, overlayLayers, { position: "topright" }).addTo(map);

  const overlayBounds = overlayLayers["Risk Overlay"] ? overlayLayers["Risk Overlay"].getBounds() : null;
  if (routesBounds && routesBounds.isValid()) {
    map.fitBounds(routesBounds.pad(0.1));
  } else if (overlayBounds && overlayBounds.isValid()) {
    map.fitBounds(overlayBounds.pad(0.1));
  } else {
    map.setView([70, 0], 3);
  }

  return {
    map,
    overlayLayers,
    routeLayers,
    hotspotLayer,
    hotspotHeatLayer,
  };
}
