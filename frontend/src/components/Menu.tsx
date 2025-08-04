import React from 'react';
import { useLocation } from 'react-router-dom';

interface MenuProps {
  projectName: string;
  onNavigate?: (path: string) => void;
}

const icons: Record<string, React.FC<{ className?: string }>> = {
  Home: () => <img src="/src/assets/home.svg" alt="Home" className="w-4 h-4" />,
  FileText: () => <img src="/src/assets/file.svg" alt="FileText" className="w-4 h-4" />,
  Target: () => <img src="/src/assets/home.svg" alt="Target" className="w-4 h-4" />,
  Shield: () => <img src="/src/assets/home.svg" alt="Shield" className="w-4 h-4" />,
  Cpu: () => <img src="/src/assets/home.svg" alt="Cpu" className="w-4 h-4" />,
  BarChart3: () => <img src="/src/assets/home.svg" alt="BarChart3" className="w-4 h-4" />,
  Users: () => <img src="/src/assets/home.svg" alt="Users" className="w-4 h-4" />,
  Building: () => <img src="/src/assets/home.svg" alt="Building" className="w-4 h-4" />,
  Folder: () => <img src="/src/assets/home.svg" alt="Folder" className="w-4 h-4" />,
  User: () => <img src="/src/assets/home.svg" alt="User" className="w-4 h-4" />,
};

const Menu: React.FC<MenuProps> = ({ projectName, onNavigate }) => {
  const location = useLocation();

  const menuItems = [
    { type: 'header', label: 'Menu' },
    { type: 'item', label: 'Home', icon: 'Home', path: '/' },
    { type: 'header', label: 'Tools' },
    { type: 'item', label: 'Policy Analysis', icon: 'Target', path: '/policy-analysis' },
    { type: 'item', label: 'Opportunity Identification', icon: 'Target' },
    { type: 'item', label: 'Compliance Risk Assessment', icon: 'Shield', path: '/compliance-risk-assessment' },
    { type: 'item', label: 'ML Test Widget', icon: 'Cpu', path: '/ml-test-widget' },
    { type: 'item', label: 'Reports Generation', icon: 'BarChart3' },
    { type: 'item', label: 'Policy Engagement', icon: 'Users' },
    { type: 'item', label: 'Vendor Management', icon: 'Building' },
    { type: 'header', label: 'Project Folders:' },
    { type: 'item', label: projectName, icon: 'Folder' },
    { type: 'item', label: 'UX@Berkeley', icon: 'Folder' },
    { type: 'header', label: 'Reports:' },
    { type: 'item', label: 'July 13 - 2:46am', icon: 'FileText' },
    { type: 'item', label: 'July 5th - 11:53am', icon: 'FileText' },
    { type: 'item', label: 'Jan. 7th - 12:11am', icon: 'FileText' },
    { type: 'item', label: 'Jan. 30th - 6:40pm', icon: 'FileText' },
    { type: 'header', label: 'Team Members' },
    { type: 'item', label: 'Yan', icon: 'User' },
    { type: 'item', label: 'Heidy', icon: 'User' },
    { type: 'item', label: 'Max', icon: 'User' },
    { type: 'item', label: 'Smaran', icon: 'User' },
    { type: 'item', label: 'Dhyan', icon: 'User' },
  ];

  const renderIcon = (iconName: string) => {
    const IconComponent = icons[iconName];
    if (IconComponent) {
      return <IconComponent className="w-4 h-4" />;
    }
    return <span className="w-4 h-4" />;
  };

  return (
    <div className="w-64 h-full bg-white flex flex-col border-r border-[#d9d9d9] rounded-3xl shadow-lg">
      <div className="px-4 py-4 border-b border-[#f0f0f0] flex items-center gap-2">
        <img src="/icons/LogoVector.svg" alt="Logo" className="w-6 h-6" />
        <span
          className="text-[20.13px] text-[#1975d4] font-bold inline-block"
          style={{ fontFamily: "'Libre Baskerville', serif", width: '123px' }}
        >GovernAIce</span>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {menuItems.map((item, index) =>
          item.type === 'header' ? (
            <h3
              key={index}
              className="text-[#9ea2ae] text-xs font-semibold mt-4 mb-1 px-2"
            >
              {item.label}
            </h3>
          ) : (
            <button
              key={index}
              onClick={() => {
                if (item.path && onNavigate) onNavigate(item.path);
              }}
              className={`w-full flex items-center gap-2 px-3 py-1 text-sm rounded-none ${
                item.path && location.pathname === item.path
                  ? 'bg-gray-200 text-[#1975d4] font-bold'
                  : 'bg-transparent text-black hover:bg-gray-100'
              }`}
            >
              {renderIcon(item.icon || '')}
              <span className="truncate">{item.label}</span>
            </button>
          )
        )}
      </div>
      <div className="p-4 border-t border-[#f0f0f0] flex items-center gap-2">
        <div className="w-8 h-8 bg-[#1975d4] rounded-full flex items-center justify-center">
          <span className="text-white text-sm font-bold">H</span>
        </div>
        <span className="text-sm text-[#000000] font-medium">Heidy</span>
      </div>
    </div>
  );
};

export default Menu; 
