const EURiskLevelFrameworkWidget = () => (
  <Card className="custom-border relative p-4">
    <img
      src="/public/icons/info.svg"
      alt="Info"
      className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-col gap-2.5 items-center">
      <h3 className="text-xl text-[#f9c74f] font-bold">EU Risk Level Framework: 5/8</h3>
      <p className="text-sm text-black">
        Compare use case to the EU AI Risk Framework and give overall score about the risk and mitigation levels
      </p>
      <div className="w-full h-70 flex items-center">
        <div className="relative w-full h-full p-4">
          <div className="flex items-end justify-center h-full gap-2">
            <div className="flex flex-col items-center gap-1">
              <div className="flex gap-1">
                <div className="w-4 h-16 bg-[#f9c74f] rounded-t"></div>
                <div className="w-4 h-12 bg-[#f9c74f] rounded-t"></div>
                <div className="w-4 h-20 bg-[#f9c74f] rounded-t"></div>
              </div>
              <span className="text-xs text-[#9ea2ae]">Figma</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <div className="flex gap-1">
                <div className="w-4 h-20 bg-[#f9c74f] rounded-t"></div>
                <div className="w-4 h-16 bg-[#f9c74f] rounded-t"></div>
                <div className="w-4 h-24 bg-[#f9c74f] rounded-t"></div>
              </div>
              <span className="text-xs text-[#9ea2ae]">AI</span>
            </div>
          </div>
          <div className="absolute bottom-0 right-0 text-xs">
            <div className="flex flex-wrap gap-2">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-[#f9c74f] rounded"></div>
                <span className="text-[#9ea2ae]">2020</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-[#f9c74f] rounded"></div>
                <span className="text-[#9ea2ae]">2021</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-[#f9c74f] rounded"></div>
                <span className="text-[#9ea2ae]">2022</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-[#f9c74f] rounded"></div>
                <span className="text-[#9ea2ae]">2023</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-[#f9c74f] rounded"></div>
                <span className="text-[#9ea2ae]">2024</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Card>
);