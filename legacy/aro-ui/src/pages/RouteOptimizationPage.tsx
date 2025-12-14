import React, { useEffect, useRef, useState } from 'react';
import { Map, View } from 'ol';
import { fromLonLat } from 'ol/proj';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import { Feature } from 'ol';
import { Point, LineString } from 'ol/geom';
import { Style, Circle, Fill, Stroke, Text } from 'ol/style';
import { Thermometer, Wind, AlertTriangle } from 'lucide-react';
import { useStore } from '../store';
import { Iceberg, Ship as ShipType } from '../types';
import CO2Panel from '../components/CO2Panel';

function RouteOptimizationPage() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const [selectedIceberg, setSelectedIceberg] = useState<Iceberg | null>(null);
  const [selectedShip, setSelectedShip] = useState<ShipType | null>(null);
  const {
    ports,
    selectedDeparture,
    selectedArrival,
    currentRoute,
    setDeparture,
    setArrival,
    calculateRoute
  } = useStore();

  useEffect(() => {
    if (!mapRef.current) return;

    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM()
        })
      ],
      view: new View({
        center: fromLonLat([0, 70]),
        zoom: 3
      })
    });

    map.on('click', (event) => {
      const feature = map.forEachFeatureAtPixel(event.pixel, feature => feature);
      if (feature) {
        const properties = feature.getProperties();
        if (properties.type === 'iceberg') {
          setSelectedIceberg(properties.data);
          setSelectedShip(null);
        } else if (properties.type === 'ship') {
          setSelectedShip(properties.data);
          setSelectedIceberg(null);
        } else {
          setSelectedIceberg(null);
          setSelectedShip(null);
        }
      } else {
        setSelectedIceberg(null);
        setSelectedShip(null);
      }
    });

    mapInstanceRef.current = map;

    return () => map.setTarget(undefined);
  }, []);

  useEffect(() => {
    if (!mapInstanceRef.current) return;

    const features = [];

    ports.forEach(port => {
      const feature = new Feature({
        geometry: new Point(fromLonLat([port.longitude, port.latitude])),
        name: port.name,
        type: 'port',
        data: port
      });

      feature.setStyle(new Style({
        image: new Circle({
          radius: 8,
          fill: new Fill({ color: '#4CAF50' }),
          stroke: new Stroke({ color: '#fff', width: 2 })
        }),
        text: new Text({
          text: port.name,
          offsetY: -15,
          fill: new Fill({ color: '#000' }),
          stroke: new Stroke({ color: '#fff', width: 3 })
        })
      }));

      features.push(feature);
    });

    if (currentRoute) {
      const routeFeature = new Feature({
        geometry: new LineString(currentRoute.coordinates.map(coord => fromLonLat(coord)))
      });

      routeFeature.setStyle(new Style({
        stroke: new Stroke({
          color: '#2196F3',
          width: 3
        })
      }));

      features.push(routeFeature);

      currentRoute.alternativeRoutes.forEach((route, index) => {
        const altRouteFeature = new Feature({
          geometry: new LineString(route.coordinates.map(coord => fromLonLat(coord)))
        });

        altRouteFeature.setStyle(new Style({
          stroke: new Stroke({
            color: '#90CAF9',
            width: 2,
            lineDash: [10, 10]
          })
        }));

        features.push(altRouteFeature);
      });

      currentRoute.ships?.forEach(ship => {
        const shipFeature = new Feature({
          geometry: new Point(fromLonLat([ship.longitude, ship.latitude])),
          type: 'ship',
          data: ship
        });

        shipFeature.setStyle(new Style({
          image: new Circle({
            radius: 6,
            fill: new Fill({ color: '#1976D2' }),
            stroke: new Stroke({ color: '#fff', width: 2 })
          }),
          text: new Text({
            text: ship.name,
            offsetY: -12,
            fill: new Fill({ color: '#1976D2' }),
            stroke: new Stroke({ color: '#fff', width: 3 })
          })
        }));

        features.push(shipFeature);
      });

      currentRoute.icebergs?.forEach(iceberg => {
        const icebergFeature = new Feature({
          geometry: new Point(fromLonLat([iceberg.longitude, iceberg.latitude])),
          type: 'iceberg',
          data: iceberg
        });

        const pathFeature = new Feature({
          geometry: new LineString(iceberg.predictedPath.map(coord => fromLonLat(coord)))
        });

        pathFeature.setStyle(new Style({
          stroke: new Stroke({
            color: '#F44336',
            width: 2,
            lineDash: [5, 5]
          })
        }));

        icebergFeature.setStyle(new Style({
          image: new Circle({
            radius: 8,
            fill: new Fill({ color: '#F44336' }),
            stroke: new Stroke({ color: '#fff', width: 2 })
          }),
          text: new Text({
            text: iceberg.name,
            offsetY: -12,
            fill: new Fill({ color: '#F44336' }),
            stroke: new Stroke({ color: '#fff', width: 3 })
          })
        }));

        features.push(pathFeature);
        features.push(icebergFeature);
      });
    }

    const vectorLayer = new VectorLayer({
      source: new VectorSource({
        features: features
      })
    });

    mapInstanceRef.current.getLayers().clear();
    mapInstanceRef.current.addLayer(new TileLayer({ source: new OSM() }));
    mapInstanceRef.current.addLayer(vectorLayer);
  }, [ports, currentRoute, selectedIceberg, selectedShip]);

  return (
    <div className="flex h-screen">
        {/* Sidebar */}
        <div className="w-96 bg-white shadow-lg p-6 overflow-y-auto">
          <h1 className="text-2xl font-bold mb-6">Arctic Route Optimization</h1>
          
          {/* Route Planning */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-4">Route Planning</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Departure Port
                </label>
                <select
                  className="w-full border rounded-md p-2"
                  value={selectedDeparture?.id || ''}
                  onChange={(e) => setDeparture(ports.find(p => p.id === e.target.value) || null)}
                >
                  <option value="">Select departure port</option>
                  {ports.map(port => (
                    <option key={port.id} value={port.id}>
                      {port.name}, {port.country} ({port.latitude.toFixed(4)}°, {port.longitude.toFixed(4)}°)
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Arrival Port
                </label>
                <select
                  className="w-full border rounded-md p-2"
                  value={selectedArrival?.id || ''}
                  onChange={(e) => setArrival(ports.find(p => p.id === e.target.value) || null)}
                >
                  <option value="">Select arrival port</option>
                  {ports.map(port => (
                    <option key={port.id} value={port.id}>
                      {port.name}, {port.country} ({port.latitude.toFixed(4)}°, {port.longitude.toFixed(4)}°)
                    </option>
                  ))}
                </select>
              </div>

              <button
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
                disabled={!selectedDeparture || !selectedArrival}
                onClick={calculateRoute}
              >
                Calculate Route
              </button>
            </div>
          </div>

          {/* CO2 Summary Panel */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-4">CO₂ Summary</h2>
            <CO2Panel />
          </div>

          {/* Selected Item Details */}
          {selectedIceberg && (
            <div className="mb-8 bg-red-50 p-4 rounded-lg">
              <h3 className="font-semibold text-red-800 mb-2">{selectedIceberg.name}</h3>
              <div className="space-y-2 text-sm text-red-700">
                <p>Position: {selectedIceberg.latitude.toFixed(4)}°, {selectedIceberg.longitude.toFixed(4)}°</p>
                <p>Size: {selectedIceberg.size}</p>
                <p>Drift Speed: {selectedIceberg.driftSpeed.toFixed(1)} knots</p>
                <p>Risk Probability: {selectedIceberg.riskProbability.toFixed(1)}%</p>
                <p>Last Seen: {new Date(selectedIceberg.lastSeen).toLocaleString()}</p>
                <p>Estimated Melt Date: {new Date(selectedIceberg.estimatedMeltDate).toLocaleDateString()}</p>
              </div>
            </div>
          )}

          {selectedShip && (
            <div className="mb-8 bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-2">{selectedShip.name}</h3>
              <div className="space-y-2 text-sm text-blue-700">
                <p>Position: {selectedShip.latitude.toFixed(4)}°, {selectedShip.longitude.toFixed(4)}°</p>
                <p>Speed: {selectedShip.speed.toFixed(1)} knots</p>
                <p>Heading: {selectedShip.heading.toFixed(1)}°</p>
                <p>Status: {selectedShip.status}</p>
                <p>Cargo Type: {selectedShip.cargoType}</p>
                <p>Risk Score: {selectedShip.riskScore.toFixed(1)}%</p>
                <p>ETA: {new Date(selectedShip.eta).toLocaleString()}</p>
              </div>
            </div>
          )}

          {/* Route Details */}
          {currentRoute && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b">
                  <h2 className="text-xl font-semibold">Route Details</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-gray-600">Distance:</span>
                      <p className="font-medium">{currentRoute.distance} km</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Est. Time:</span>
                      <p className="font-medium">{currentRoute.estimatedTime} hours</p>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Risk Level:</span>
                    <p className={`font-medium ${currentRoute.riskLevel === 'High' ? 'text-red-600' : currentRoute.riskLevel === 'Medium' ? 'text-yellow-600' : 'text-green-600'}`}>
                      {currentRoute.riskLevel}
                    </p>
                  </div>
                </div>
              </div>

              {/* Weather Forecast */}
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b">
                  <h2 className="text-xl font-semibold">Weather Conditions</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center">
                      <Thermometer className="w-5 h-5 mr-2" />
                      <div>
                        <span className="text-gray-600">Temperature</span>
                        <p className="font-medium">{currentRoute.weatherConditions.temperature.toFixed(1)}°C</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <Wind className="w-5 h-5 mr-2" />
                      <div>
                        <span className="text-gray-600">Wind Speed</span>
                        <p className="font-medium">{currentRoute.weatherConditions.windSpeed.toFixed(1)} knots</p>
                      </div>
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Wave Height:</span>
                    <p className="font-medium">{currentRoute.weatherConditions.waveHeight.toFixed(1)}m</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Sea Ice:</span>
                    <p className="font-medium">{currentRoute.weatherConditions.seaIceConcentration.toFixed(1)}%</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Visibility:</span>
                    <p className="font-medium">{currentRoute.weatherConditions.visibility}</p>
                  </div>
                </div>
              </div>

              {/* Traffic Analysis */}
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b">
                  <h2 className="text-xl font-semibold">Traffic Analysis</h2>
                </div>
                <div className="p-4 space-y-4">
                  <div>
                    <span className="text-gray-600">Active Ships:</span>
                    <p className="font-medium">{currentRoute.ships?.length || 0}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Icebergs Detected:</span>
                    <p className="font-medium">{currentRoute.icebergs?.length || 0}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Congestion Level:</span>
                    <p className="font-medium">{currentRoute.trafficCongestion.toFixed(1)}%</p>
                  </div>
                </div>
              </div>

              {/* Alerts */}
              {currentRoute.alerts.length > 0 && (
                <div className="bg-white rounded-lg shadow">
                  <div className="p-4 border-b">
                    <h2 className="text-xl font-semibold">Active Alerts</h2>
                  </div>
                  <div className="p-4">
                    <div className="space-y-4">
                      {currentRoute.alerts.map((alert, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg flex items-start space-x-3 ${alert.severity === 'High' ? 'bg-red-50 text-red-800' : alert.severity === 'Medium' ? 'bg-yellow-50 text-yellow-800' : 'bg-blue-50 text-blue-800'}`}>
                          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium">{alert.message}</p>
                            <p className="text-sm mt-1">Time to impact: {alert.timeToImpact} hours</p>
                            {alert.location && (
                              <p className="text-sm mt-1">
                                Location: {alert.location.latitude.toFixed(4)}°, {alert.location.longitude.toFixed(4)}°
                              </p>
                            )}
                            {alert.details && (
                              <p className="text-sm mt-1">{alert.details}</p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Map */}
        <div className="flex-1 relative">
          <div ref={mapRef} className="absolute inset-0" />
        </div>
      </div>
  );
}

export default RouteOptimizationPage;










