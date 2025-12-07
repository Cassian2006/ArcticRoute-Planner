import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { Map, Users, BarChart } from 'lucide-react';

const navItems = [
  { href: '/', label: 'Route Optimization', icon: Map },
  { href: '/legacy-ui', label: 'Legacy UI', icon: Users },
  { href: '/maps-ui', label: 'Maps UI', icon: BarChart },
];

function App() {
  const location = useLocation();

  return (
    <div className="flex h-screen bg-gray-100">
      <aside className="w-64 flex-shrink-0 bg-gray-800 text-white flex flex-col">
        <div className="h-16 flex items-center justify-center text-xl font-bold border-b border-gray-700">
          ARO-System
        </div>
        <nav className="flex-1 px-2 py-4 space-y-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.label}
                to={item.href}
                className={`flex items-center px-4 py-2.5 text-sm font-medium rounded-lg transition-colors ${
                  isActive
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

export default App;
