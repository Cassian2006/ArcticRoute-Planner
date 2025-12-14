// Minimal API client for fuel-service and read-only endpoints
export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8123';
export const FUEL_BASE = import.meta.env.VITE_FUEL_BASE_URL || 'http://localhost:8123';

export async function listRoutes(ym: string) {
  const r = await fetch(`${API_BASE}/routes/list?ym=${encodeURIComponent(ym)}`);
  if (!r.ok) throw new Error(`routes.list failed: ${r.status}`);
  return r.json();
}

export async function getRoute(name: string) {
  const r = await fetch(`${API_BASE}/routes/get?name=${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(`routes.get failed: ${r.status}`);
  return r.json(); // GeoJSON
}

export async function layersMeta(ym: string) {
  const r = await fetch(`${API_BASE}/layers/meta?ym=${encodeURIComponent(ym)}`);
  if (!r.ok) throw new Error(`layers.meta failed: ${r.status}`);
  return r.json();
}

export async function predictFuel(routeGeoJSON: any, ym: string, vesselClass = 'cargo_iceclass') {
  const r = await fetch(`${FUEL_BASE}/fuel/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ route_geojson: routeGeoJSON, ym, vessel_class: vesselClass })
  });
  if (!r.ok) throw new Error(`fuel.predict failed: ${r.status}`);
  return r.json();
}


