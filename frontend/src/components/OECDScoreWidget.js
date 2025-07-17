const OECDScoreWidget = () => (
  <Card className="custom-border relative p-4">
    <img
      src="/public/icons/info.svg"
      alt="Info"
      className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-col gap-2.5 items-center">
      <h3 className="text-xl text-[#1975d4] font-bold">OECD Score: 80</h3>
      <p className="text-sm text-black">
        Compare use case to the OECD AI Risk Framework and give an overall score about the compliance level
      </p>
      <div className="w-full h-70 flex items-center">
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <polygon points="100,20 150,50 170,100 150,150 100,180 50,150 30,100 50,50" fill="none" stroke="#d9d9d9" strokeWidth="1"/>
          <polygon points="100,40 130,60 150,100 130,140 100,160 70,140 50,100 70,60" fill="none" stroke="#d9d9d9" strokeWidth="1"/>
          <polygon points="100,60 110,80 130,100 110,120 100,140 90,120 70,100 90,80" fill="none" stroke="#d9d9d9" strokeWidth="1"/>
          <polygon points="100,30 140,65 160,100 140,135 100,170 60,135 40,100 60,65" fill="#1975d4" fillOpacity="0.3" stroke="#1975d4" strokeWidth="2"/>
          <line x1="100" y1="100" x2="100" y2="20" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="150" y2="50" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="170" y2="100" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="150" y2="150" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="100" y2="180" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="50" y2="150" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="30" y2="100" stroke="#d9d9d9" strokeWidth="1"/>
          <line x1="100" y1="100" x2="50" y2="50" stroke="#d9d9d9" strokeWidth="1"/>
        </svg>
      </div>
    </div>
  </Card>
);