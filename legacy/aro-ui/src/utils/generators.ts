import { Port, Ship, Iceberg, Route, WeatherCondition, Alert } from '../types';

// Arctic and Antarctic sea coordinates for route validation
const seaCoordinates = {
  arctic: [
    // Northern Sea Route (Russia)
    [[30, 69], [180, 77]], // Main path
    [[30, 65], [180, 73]], // Southern variant
    // Northwest Passage (Canada)
    [[-140, 75], [-50, 75]],
    // Greenland Sea
    [[-40, 70], [20, 80]],
  ],
  antarctic: [
    // Drake Passage
    [[-70, -55], [-60, -65]],
    // Ross Sea
    [[160, -70], [180, -75]],
    // Weddell Sea
    [[-60, -75], [-30, -75]],
  ]
};

const ports: Port[] = [
  // Arctic Ports
  {
    id: 'p1',
    name: 'Murmansk',
    latitude: 68.9585,
    longitude: 33.0827,
    country: 'Russia',
    congestion: 85,
    description: 'Major Arctic port with year-round operations'
  },
  {
    id: 'p2',
    name: 'Pevek',
    latitude: 69.7019,
    longitude: 170.2999,
    country: 'Russia',
    congestion: 45,
    description: 'Northernmost port in Russia'
  },
  {
    id: 'p3',
    name: 'Longyearbyen',
    latitude: 78.2232,
    longitude: 15.6267,
    country: 'Norway',
    congestion: 65,
    description: 'Main settlement in Svalbard'
  },
  {
    id: 'p4',
    name: 'Nuuk',
    latitude: 64.1835,
    longitude: -51.7216,
    country: 'Greenland',
    congestion: 55,
    description: 'Capital of Greenland'
  },
  {
    id: 'p5',
    name: 'Tiksi',
    latitude: 71.6351,
    longitude: 128.8644,
    country: 'Russia',
    congestion: 40,
    description: 'Important port on Northern Sea Route'
  },
  {
    id: 'p6',
    name: 'Sabetta',
    latitude: 71.2714,
    longitude: 72.0686,
    country: 'Russia',
    congestion: 75,
    description: 'LNG export terminal'
  },
  {
    id: 'p7',
    name: 'Churchill',
    latitude: 58.7684,
    longitude: -94.1650,
    country: 'Canada',
    congestion: 35,
    description: 'Canadian Arctic port'
  },
  // Antarctic Stations/Ports
  {
    id: 'p8',
    name: 'McMurdo Station',
    latitude: -77.8419,
    longitude: 166.6863,
    country: 'Antarctica',
    congestion: 30,
    description: 'Largest Antarctic research station'
  },
  {
    id: 'p9',
    name: 'Rothera Research Station',
    latitude: -67.5674,
    longitude: -68.1255,
    country: 'Antarctica',
    congestion: 25,
    description: 'British Antarctic Survey main station'
  },
  {
    id: 'p10',
    name: 'Davis Station',
    latitude: -68.5760,
    longitude: 77.9689,
    country: 'Antarctica',
    congestion: 20,
    description: 'Australian Antarctic station'
  }
];

// Sea route waypoints ensuring ships follow water routes
const seaRouteWaypoints: Record<string, [number, number][]> = {
  'p1-p3': [ // Murmansk to Longyearbyen
    [33.0827, 68.9585], // Murmansk
    [33.5000, 71.0000], // Barents Sea
    [25.0000, 74.0000], // Northern route
    [20.0000, 76.0000], // Approaching Svalbard
    [15.6267, 78.2232]  // Longyearbyen
  ],
  'p1-p2': [ // Murmansk to Pevek (Northern Sea Route)
    [33.0827, 68.9585], // Murmansk
    [40.0000, 70.0000], // Barents Sea
    [60.0000, 73.0000], // Kara Sea
    [100.0000, 75.0000], // Laptev Sea
    [140.0000, 74.0000], // East Siberian Sea
    [170.2999, 69.7019]  // Pevek
  ],
  'p4-p7': [ // Nuuk to Churchill
    [-51.7216, 64.1835], // Nuuk
    [-55.0000, 65.0000], // Davis Strait
    [-70.0000, 65.0000], // Hudson Strait
    [-85.0000, 62.0000], // Hudson Bay
    [-94.1650, 58.7684]  // Churchill
  ],
  'p8-p9': [ // McMurdo to Rothera
    [166.6863, -77.8419], // McMurdo
    [180.0000, -75.0000], // Ross Sea
    [-150.0000, -72.0000], // Amundsen Sea
    [-100.0000, -70.0000], // Bellingshausen Sea
    [-68.1255, -67.5674]  // Rothera
  ]
};

// Function to check if a point is in water (simplified version)
const isInWater = (lon: number, lat: number): boolean => {
  // Check if point is within any sea route corridor
  for (const region of [...seaCoordinates.arctic, ...seaCoordinates.antarctic]) {
    const [[x1, y1], [x2, y2]] = region;
    if (lon >= Math.min(x1, x2) && lon <= Math.max(x1, x2) &&
        lat >= Math.min(y1, y2) && lat <= Math.max(y1, y2)) {
      return true;
    }
  }
  return false;
};

// Generate waypoints ensuring sea route
const generateSeaRouteWaypoints = (start: [number, number], end: [number, number]): [number, number][] => {
  const waypoints: [number, number][] = [start];
  const numPoints = 5;
  
  for (let i = 1; i < numPoints - 1; i++) {
    const t = i / (numPoints - 1);
    let point: [number, number];
    let attempts = 0;
    
    do {
      // Generate point along great circle with some random deviation
      const baseLon = start[0] + (end[0] - start[0]) * t;
      const baseLat = start[1] + (end[1] - start[1]) * t;
      
      // Add some random deviation, but try to stay in water
      const deviation = Math.min(5, attempts * 0.5); // Increase deviation with failed attempts
      point = [
        baseLon + (Math.random() - 0.5) * deviation,
        baseLat + (Math.random() - 0.5) * deviation
      ];
      attempts++;
    } while (!isInWater(point[0], point[1]) && attempts < 10);
    
    waypoints.push(point);
  }
  
  waypoints.push(end);
  return waypoints;
};

const generateIcebergs = (route: Route): Iceberg[] => {
  const icebergs: Iceberg[] = [];
  const numIcebergs = Math.floor(Math.random() * 3) + 2;

  for (let i = 0; i < numIcebergs; i++) {
    const routeIndex = Math.floor(Math.random() * (route.coordinates.length - 1));
    const basePosition = route.coordinates[routeIndex];
    
    // Ensure iceberg is in water
    let offset: number;
    let icebergLon: number;
    let icebergLat: number;
    
    do {
      offset = (Math.random() - 0.5) * 4;
      icebergLon = basePosition[0] + offset;
      icebergLat = basePosition[1] + offset;
    } while (!isInWater(icebergLon, icebergLat));

    icebergs.push({
      id: `iceberg-${route.id}-${i}`,
      name: `Titan-${String(i + 1).padStart(2, '0')}`,
      latitude: icebergLat,
      longitude: icebergLon,
      size: ['Small', 'Medium', 'Large'][Math.floor(Math.random() * 3)],
      driftSpeed: Math.random() * 2,
      riskProbability: Math.random() * 100,
      lastSeen: new Date().toISOString(),
      predictedPath: generatePredictedPath(icebergLat, icebergLon),
      description: `Large iceberg moving at ${(Math.random() * 2).toFixed(1)} knots`,
      estimatedMeltDate: new Date(Date.now() + Math.random() * 7776000000).toISOString()
    });
  }

  return icebergs;
};

const generateShips = (route: Route): Ship[] => {
  const ships: Ship[] = [];
  const numShips = Math.floor(Math.random() * 3) + 2; // Fewer ships per route

  for (let i = 0; i < numShips; i++) {
    const progress = Math.random();
    const routeIndex = Math.floor(progress * (route.coordinates.length - 1));
    const position = route.coordinates[routeIndex];

    ships.push({
      id: `ship-${route.id}-${i}`,
      name: `Arctic Voyager ${i + 1}`,
      latitude: position[1],
      longitude: position[0],
      speed: 12 + Math.random() * 8, // 12-20 knots
      destination: route.arrival.name,
      riskScore: Math.random() * 100,
      cargoType: ['Container', 'Bulk', 'Tanker', 'LNG', 'Research'][Math.floor(Math.random() * 5)],
      eta: new Date(Date.now() + Math.random() * 86400000 * 5).toISOString(),
      heading: Math.random() * 360,
      status: ['En Route', 'Anchored', 'Loading'][Math.floor(Math.random() * 3)]
    });
  }

  return ships;
};

const generatePredictedPath = (lat: number, lon: number): [number, number][] => {
  const path: [number, number][] = [];
  const numPoints = 5;
  let currentLat = lat;
  let currentLon = lon;
  
  for (let i = 0; i < numPoints; i++) {
    let newLon: number;
    let newLat: number;
    
    do {
      const latOffset = (Math.random() - 0.5) * 0.5;
      const lonOffset = (Math.random() - 0.5) * 0.5;
      newLat = currentLat + latOffset;
      newLon = currentLon + lonOffset;
    } while (!isInWater(newLon, newLat));
    
    currentLat = newLat;
    currentLon = newLon;
    path.push([currentLon, currentLat]);
  }
  
  return path;
};

const generateWeatherCondition = (route: Route): WeatherCondition => {
  // Check if route has valid departure and arrival
  if (!route?.departure?.latitude || !route?.arrival?.latitude) {
    // Return default weather conditions if route data is incomplete
    return {
      temperature: 0,
      windSpeed: 15,
      visibility: 'Good',
      forecast: 'Clear',
      waveHeight: 1,
      seaIceConcentration: 0,
      predictions: Array.from({ length: 24 }, (_, i) => ({
        hour: i,
        temperature: 0,
        windSpeed: 15,
        condition: 'Clear'
      }))
    };
  }

  // Base conditions on route location
  const isArctic = route.departure.latitude > 0;
  const season = new Date().getMonth(); // 0-11
  const isSummer = season >= 4 && season <= 8;
  
  // Temperature range based on region and season
  const baseTemp = isArctic ? 
    (isSummer ? -5 : -25) : // Arctic
    (isSummer ? 0 : -15);   // Antarctic
  
  const temp = baseTemp + (Math.random() - 0.5) * 10;
  
  // Wind is stronger in winter
  const baseWind = isSummer ? 15 : 25;
  const wind = baseWind + Math.random() * 20;
  
  const conditions = ['Clear', 'Partly Cloudy', 'Overcast', 'Snow', 'Blizzard'];
  const condition = conditions[Math.floor(Math.random() * conditions.length)];
  
  return {
    temperature: temp,
    windSpeed: wind,
    visibility: ['Poor', 'Moderate', 'Good'][Math.floor(Math.random() * 3)] as 'Poor' | 'Moderate' | 'Good',
    forecast: condition,
    waveHeight: 1 + Math.random() * 4,
    seaIceConcentration: Math.random() * 100,
    predictions: Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      temperature: temp + (Math.random() - 0.5) * 5,
      windSpeed: wind + (Math.random() - 0.5) * 10,
      condition: conditions[Math.floor(Math.random() * conditions.length)]
    }))
  };
};

const generateAlerts = (route: Route, icebergs: Iceberg[]): Alert[] => {
  const alerts: Alert[] = [];

  // Iceberg collision risks
  icebergs.forEach(iceberg => {
    if (iceberg.riskProbability > 70) {
      alerts.push({
        type: 'Collision',
        severity: 'High',
        message: `High risk of collision with ${iceberg.name}`,
        timeToImpact: Math.floor(Math.random() * 12),
        location: {
          latitude: iceberg.latitude,
          longitude: iceberg.longitude
        }
      });
    }
  });

  // Weather alerts based on route-specific conditions
  if (route.weatherConditions.windSpeed > 30) {
    alerts.push({
      type: 'Weather',
      severity: 'Medium',
      message: 'Severe weather conditions expected',
      timeToImpact: Math.floor(Math.random() * 24),
      details: `Wind speeds exceeding ${Math.round(route.weatherConditions.windSpeed)} knots`
    });
  }

  if (route.weatherConditions.seaIceConcentration > 80) {
    alerts.push({
      type: 'Weather',
      severity: 'High',
      message: 'High sea ice concentration',
      timeToImpact: Math.floor(Math.random() * 12),
      details: `Sea ice concentration at ${Math.round(route.weatherConditions.seaIceConcentration)}%`
    });
  }

  // Traffic congestion
  if (route.trafficCongestion > 80) {
    alerts.push({
      type: 'Traffic',
      severity: 'Medium',
      message: 'High traffic area ahead',
      timeToImpact: Math.floor(Math.random() * 6),
      details: 'Multiple vessels converging'
    });
  }

  return alerts;
};

const getRouteWaypoints = (departure: Port, arrival: Port): [number, number][] => {
  const routeKey = `${departure.id}-${arrival.id}`;
  
  // Use predefined route if available
  if (seaRouteWaypoints[routeKey]) {
    return seaRouteWaypoints[routeKey];
  }

  // Generate sea route
  return generateSeaRouteWaypoints(
    [departure.longitude, departure.latitude],
    [arrival.longitude, arrival.latitude]
  );
};

export const generateRoute = (departureId: string, arrivalId: string): Route | null => {
  const departure = ports.find(p => p.id === departureId);
  const arrival = ports.find(p => p.id === arrivalId);

  if (!departure || !arrival) return null;

  const coordinates = getRouteWaypoints(departure, arrival);
  const baseRoute: Route = {
    id: `${departure.id}-${arrival.id}`,
    departure,
    arrival,
    distance: calculateDistance(departure, arrival),
    estimatedTime: Math.floor(Math.random() * 120) + 48,
    riskLevel: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)] as 'Low' | 'Medium' | 'High',
    weatherConditions: generateWeatherCondition({ departure, arrival } as Route),
    trafficCongestion: Math.random() * 100,
    coordinates,
    alerts: [],
    alternativeRoutes: []
  };

  // Generate alternative routes
  const numAlternatives = Math.floor(Math.random() * 2) + 1;
  for (let i = 0; i < numAlternatives; i++) {
    const altCoordinates = coordinates.map(coord => {
      let newLon: number;
      let newLat: number;
      do {
        const offset = (Math.random() - 0.5) * 2;
        newLon = coord[0] + offset;
        newLat = coord[1] + offset;
      } while (!isInWater(newLon, newLat));
      return [newLon, newLat] as [number, number];
    });

    baseRoute.alternativeRoutes.push({
      coordinates: altCoordinates,
      distance: calculateDistance(departure, arrival) * (1 + Math.random() * 0.2),
      estimatedTime: Math.floor(Math.random() * 120) + 48,
      riskLevel: ['Low', 'Medium'][Math.floor(Math.random() * 2)] as 'Low' | 'Medium'
    });
  }

  const icebergs = generateIcebergs(baseRoute);
  const ships = generateShips(baseRoute);
  baseRoute.alerts = generateAlerts(baseRoute, icebergs);
  baseRoute.ships = ships;
  baseRoute.icebergs = icebergs;

  return baseRoute;
};

const calculateDistance = (p1: Port, p2: Port): number => {
  const R = 6371; // Earth's radius in km
  const dLat = (p2.latitude - p1.latitude) * Math.PI / 180;
  const dLon = (p2.longitude - p1.longitude) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(p1.latitude * Math.PI / 180) * Math.cos(p2.latitude * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return Math.round(R * c);
};

export const getAllPorts = () => ports;