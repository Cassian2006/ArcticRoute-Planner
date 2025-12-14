import { create } from 'zustand';
import { Port, Route, Ship, Iceberg } from '../types';
import { generateRoute, getAllPorts } from '../utils/generators';

interface AppState {
  selectedDeparture: Port | null;
  selectedArrival: Port | null;
  currentRoute: Route | null;
  ships: Ship[];
  icebergs: Iceberg[];
  ports: Port[];
  setDeparture: (port: Port | null) => void;
  setArrival: (port: Port | null) => void;
  calculateRoute: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  selectedDeparture: null,
  selectedArrival: null,
  currentRoute: null,
  ships: [],
  icebergs: [],
  ports: getAllPorts(),

  setDeparture: (port) => set({ selectedDeparture: port }),
  
  setArrival: (port) => set({ selectedArrival: port }),

  calculateRoute: () => {
    const { selectedDeparture, selectedArrival } = get();
    if (!selectedDeparture || !selectedArrival) return;

    const route = generateRoute(selectedDeparture.id, selectedArrival.id);
    if (!route) return;

    set({ currentRoute: route });
  }
}));