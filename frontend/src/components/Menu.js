const Menu = ({ projectName, onSelect }) => {
  const [selected, setSelected] = React.useState(null);

  const menuItems = [
    { type: "header", label: "Menu" },
    { type: "item", label: "Home", icon: "Home", index: 0 },
    { type: "header", label: "Tools" },
    { type: "item", label: "Policy Analysis", icon: "FileText", index: 1 },
    { type: "item", label: "Opportunity Identification", icon: "Target", index: 2 },
    { type: "item", label: "Compliance Risk Assessment", icon: "Shield", index: 3 },
    { type: "item", label: "Reports Generation", icon: "BarChart3", index: 4 },
    { type: "item", label: "Policy Engagement", icon: "Users", index: 5 },
    { type: "item", label: "Vendor Management", icon: "Building", index: 6 },
    { type: "item", label: "Generate Compliance Report Suite", icon: "FileText", index: 7 },
    { type: "item", label: "Generate Policy Feedback & Liaison Report", icon: "Users", index: 8 },
    { type: "item", label: "Use Case 2 - Policy Opportunity Discovery", icon: "Target", index: 9 },
    { type: "header", label: "Project Folders:" },
    { type: "item", label: projectName, icon: "Folder", index: 10 },
    { type: "item", label: "UX@Berkeley", icon: "Folder", index: 11 },
    { type: "header", label: "Reports:" },
    { type: "item", label: "Feb. 28th - 9:46am", icon: "FileText", index: 12 },
    { type: "item", label: "Jan. 5th - 11:53am", icon: "FileText", index: 13 },
    { type: "item", label: "Jan. 7th - 12:11am", icon: "FileText", index: 14 },
    { type: "item", label: "Jan. 30th - 6:40pm", icon: "FileText", index: 15 },
    { type: "header", label: "Team Members" },
    { type: "item", label: "Name Name", icon: "User", index: 16 },
    { type: "item", label: "Name Name", index: 17 },
    { type: "item", label: "Name Name", index: 18 },
    { type: "item", label: "Name Name", index: 19 },
    { type: "item", label: "Name Name", index: 20 },
  ];

  const handleSelect = (index) => {
    setSelected(index);
    onSelect(index);
  };

  return (
    <div className="w-64 h-screen bg-white flex flex-col border-r border-[#d9d9d9] rounded-r-lg shadow-lg">
      <div className="px-4 py-4 border-b border-[#f0f0f0] flex items-center gap-2">
        <img src="/icons/LogoVector.svg" alt="Logo" className="w-6 h-6" />
        <span
          className="text-[20.13px] text-[#1975d4] font-bold inline-block"
          style={{ fontFamily: "'Libre Baskerville', serif", width: "123px" }}
        >
          GovernAIce
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {menuItems.map((item, index) =>
          item.type === "header" ? (
            <h3
              key={index}
              className="text-[#9ea2ae] text-xs font-semibold mt-4 mb-1 px-2"
            >
              {item.label}
            </h3>
          ) : (
            <button
              key={index}
              onClick={() => handleSelect(item.index)}
              className={`w-full flex items-center gap-2 px-3 py-1 text-sm rounded-none ${
                selected === item.index
                  ? "bg-gray-200 text-[#1975d4] font-bold"
                  : "text-black hover:bg-gray-100"
              }`}
              disabled={!item.index}
            >
              {item.icon && React.createElement(icons[item.icon], {
                className: `w-4 h-4 ${
                  selected === item.index ? "text-[#1975d4]" : "text-black"
                }`,
              })}
              <span className="truncate">{item.label}</span>
            </button>
          )
        )}
      </div>
      <div className="p-4 border-t border-[#f0f0f0] flex items-center gap-2">
        <div className="w-8 h-8 bg-[#d9d9d9] rounded-full flex items-center justify-center">
          {React.createElement(icons["User"], {
            className: "w-4 h-4 text-[#9ea2ae]",
          })}
        </div>
        <span className="text-sm text-[#000000] font-medium">Name Name</span>
      </div>
    </div>
  );
};