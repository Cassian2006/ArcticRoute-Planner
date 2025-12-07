import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import './index.css';
import RouteOptimizationPage from './pages/RouteOptimizationPage';
import LegacyUIPage from './pages/LegacyUIPage';
import MapsUIPage from './pages/MapsUIPage';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<RouteOptimizationPage />} />
          <Route path="legacy-ui" element={<LegacyUIPage />} />
          <Route path="maps-ui" element={<MapsUIPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
