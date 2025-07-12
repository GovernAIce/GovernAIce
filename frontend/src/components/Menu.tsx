import React, { useState } from 'react';

interface MenuProps {
  projectName: string;
}

const icons: Record<string, React.FC<{ className?: string }>> = {
  // SVG icon components can be imported or inlined here if needed
};

const Menu: React.FC<MenuProps> = ({ projectName }) => {
  const [selected, setSelected] = useState<number | null>(null);

  const menuItems = [
    { type: 'header', label: 'Menu' },
    { type: 'item', label: 'Home', icon: 'Home' },
    { type: 'header', label: 'Tools' },
    { type: 'item', label: 'Policy Analysis', icon: 'FileText' },
    { type: 'item', label: 'Opportunity Identification', icon: 'Target' },
    { type: 'item', label: 'Compliance Risk Assessment', icon: 'Shield' },
    { type: 'item', label: 'Reports Generation', icon: 'BarChart3' },
    { type: 'item', label: 'Policy Engagement', icon: 'Users' },
    { type: 'item', label: 'Vendor Management', icon: 'Building' },
    { type: 'header', label: 'Project Folders:' },
    { type: 'item', label: projectName, icon: 'Folder' },
    { type: 'item', label: 'UX@Berkeley', icon: 'Folder' },
    { type: 'header', label: 'Reports:' },
    { type: 'item', label: 'Feb. 28th - 9:46am', icon: 'FileText' },
    { type: 'item', label: 'Jan. 5th - 11:53am', icon: 'FileText' },
    { type: 'item', label: 'Jan. 7th - 12:11am', icon: 'FileText' },
    { type: 'item', label: 'Jan. 30th - 6:40pm', icon: 'FileText' },
    { type: 'header', label: 'Team Members' },
    { type: 'item', label: 'Name Name', icon: 'User' },
    { type: 'item', label: 'Name Name', icon: 'User' },
    { type: 'item', label: 'Name Name', icon: 'User' },
    { type: 'item', label: 'Name Name', icon: 'User' },
    { type: 'item', label: 'Name Name', icon: 'User' },
  ];

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
              onClick={() => setSelected(index)}
              className={`w-full flex items-center gap-2 px-3 py-1 text-sm rounded-none ${
                item.label === 'Home'
                  ? 'bg-gray-200 text-[#1975d4] font-bold'
                  : 'bg-transparent text-black hover:bg-gray-100'
              }`}
            >
              {/* Replace with icon import or inline SVG as needed */}
              <span className="w-4 h-4" />
              <span className="truncate">{item.label}</span>
            </button>
          )
        )}
      </div>
      <div className="p-4 border-t border-[#f0f0f0] flex items-center gap-2">
        <div className="w-8 h-8 bg-[#d9d9d9] rounded-full flex items-center justify-center">
          {/* <span className="w-4 h-4 text-[#9ea2ae]">U</span> */}
        </div>
        <span className="text-sm text-[#000000] font-medium">Name Name</span>
      </div>
    </div>
  );
};

export default Menu; 
