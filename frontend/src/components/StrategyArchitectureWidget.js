const StrategyArchitectureWidget = () => (
  <Card className="custom-border p-4">
    {React.createElement(icons["Info"], { className: "absolute top-2 right-2 w-4 h-4 cursor-pointer" })}
    <div className="flex flex-col gap-2.5 items-center">
      <h3 className="text-xl text-[#1975d4] font-bold">Strategy Architecture</h3>
      <div className="w-full h-64 flex items-center justify-center">
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <circle cx="100" cy="100" r="80" fill="#e0f2fe" />
          <circle cx="100" cy="100" r="60" fill="#bae6fd" />
          <circle cx="100" cy="100" r="40" fill="#7dd3fc" />
          <text x="100" y="95" textAnchor="middle" fontSize="12" fill="#000">People and Planet</text>
          <path d="M100 20 A80 80 0 0 1 160 100" stroke="#89c4f4" strokeWidth="20" fill="none" />
          <text x="140" y="60" textAnchor="middle" fontSize="10" fill="#000">Collect and present data</text>
          <path d="M160 100 A80 80 0 0 1 100 180" stroke="#6ee7b7" strokeWidth="20" fill="none" />
          <text x="140" y="140" textAnchor="middle" fontSize="10" fill="#000">Deploy and Use</text>
          <path d="M100 180 A80 80 0 0 1 40 100" stroke="#a3e4db" strokeWidth="20" fill="none" />
          <text x="60" y="140" textAnchor="middle" fontSize="10" fill="#000">Verify and validate</text>
          <path d="M40 100 A80 80 0 0 1 100 20" stroke="#93c5fd" strokeWidth="20" fill="none" />
          <text x="60" y="60" textAnchor="middle" fontSize="10" fill="#000">Plan and Design</text>
        </svg>
      </div>
    </div>
  </Card>
);