import { useEffect, useState } from 'react';
import { listRoutes, getRoute, predictFuel } from '../utils/api';

export default function CO2Panel() {
  const [summary, setSummary] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const ym = '202412';

  useEffect(() => {
    (async () => {
      try {
        const routes = await listRoutes(ym);
        if (!routes || routes.length === 0) {
          setError('No routes found.');
          return;
        }
        const balanced = routes.find((x: any) => x.name.includes('_balanced.geojson')) || routes[0];
        const gj = await getRoute(balanced.name);
        const res = await predictFuel(gj, ym, 'cargo_iceclass');
        setSummary(res);
      } catch (e: any) {
        setError(e?.message || 'Failed to fetch CO2');
      }
    })();
  }, []);

  if (error) return <div className="p-3 bg-red-50 text-red-800 rounded">{error}</div>;
  if (!summary) return <div className="p-3 bg-gray-50 text-gray-700 rounded">Loading CO₂…</div>;

  return (
    <div className="p-3 bg-white rounded shadow border">
      <div className="text-sm text-gray-600">Backend: fuel_service</div>
      <div className="mt-2">
        <div className="text-gray-600 text-sm">Total length (nm)</div>
        <div className="text-lg font-semibold">{summary.total_length_nm?.toFixed(2)}</div>
      </div>
      <div className="mt-2">
        <div className="text-gray-600 text-sm">Total fuel (t)</div>
        <div className="text-lg font-semibold">{summary.total_fuel_tons?.toFixed(3)}</div>
      </div>
      <div className="mt-2">
        <div className="text-gray-600 text-sm">Total CO₂ (t)</div>
        <div className="text-lg font-semibold">{summary.total_co2_tons?.toFixed(3)}</div>
      </div>
    </div>
  );
}


