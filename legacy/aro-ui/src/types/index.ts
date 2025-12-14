import { Matrix } from 'ml-matrix';

export interface Port {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  country: string;
  congestion: number;
  description: string;
  historicalData?: {
    averageTraffic: number[];
    weatherPatterns: string[];
    seasonalRisks: string[];
  };
}

export interface Ship {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  speed: number;
  destination: string;
  riskScore: number;
  cargoType: string;
  eta: string;
  heading: number;
  status: 'En Route' | 'Anchored' | 'Loading';
  fuelEfficiency?: number;
  vesselType?: string;
  iceClass?: string;
  previousPositions?: [number, number][];
}

export interface Iceberg {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  size: string;
  driftSpeed: number;
  riskProbability: number;
  lastSeen: string;
  predictedPath: [number, number][];
  description: string;
  estimatedMeltDate: string;
  volume?: number;
  surfaceArea?: number;
  stabilityIndex?: number;
  meltRate?: number;
  environmentalImpact?: {
    wildlifeThreat: number;
    ecosystemDisruption: number;
  };
}

export interface WeatherPrediction {
  hour: number;
  temperature: number;
  windSpeed: number;
  condition: string;
  precipitation?: number;
  humidity?: number;
  pressure?: number;
  uvIndex?: number;
}

export interface WeatherCondition {
  temperature: number;
  windSpeed: number;
  visibility: 'Poor' | 'Moderate' | 'Good';
  forecast: string;
  waveHeight: number;
  seaIceConcentration: number;
  predictions: WeatherPrediction[];
  atmosphericPressure?: number;
  waterTemperature?: number;
  currentDirection?: number;
  currentSpeed?: number;
}

export interface AlertLocation {
  latitude: number;
  longitude: number;
}

export interface Alert {
  type: 'Collision' | 'Weather' | 'Traffic' | 'Environmental';
  severity: 'Low' | 'Medium' | 'High';
  message: string;
  timeToImpact: number;
  location?: AlertLocation;
  details?: string;
  confidence?: number;
  recommendations?: string[];
  impactAssessment?: {
    economic: number;
    environmental: number;
    safety: number;
  };
}

export interface AlternativeRoute {
  coordinates: [number, number][];
  distance: number;
  estimatedTime: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  fuelConsumption?: number;
  costEstimate?: number;
  environmentalImpact?: number;
  confidenceScore?: number;
}

export interface Route {
  id: string;
  departure: Port;
  arrival: Port;
  distance: number;
  estimatedTime: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  weatherConditions: WeatherCondition;
  trafficCongestion: number;
  coordinates: [number, number][];
  alerts: Alert[];
  alternativeRoutes: AlternativeRoute[];
  ships?: Ship[];
  icebergs?: Iceberg[];
  mlPredictions?: {
    collisionProbability: number;
    weatherRiskMatrix: Matrix;
    trafficPredictions: number[];
    routeOptimality: number;
    confidenceIntervals: {
      lower: number;
      upper: number;
    };
  };
  environmentalMetrics?: {
    carbonFootprint: number;
    marineLifeImpact: number;
    noiseLevel: number;
  };
}

export interface AIModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  rocCurve: [number, number][];
}

export interface ModelPerformance {
  routePrediction: AIModelMetrics;
  weatherForecasting: AIModelMetrics;
  collisionPrediction: AIModelMetrics;
  trafficAnalysis: AIModelMetrics;
}