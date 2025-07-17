const Header = ({ projectName, setProjectName }) => {
  const [editMode, setEditMode] = React.useState(false);
  const [currentTime, setCurrentTime] = React.useState("");

  React.useEffect(() => {
    const now = new Date();
    setCurrentTime(
      now.toLocaleString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
        timeZone: "Asia/Kolkata",
      })
    );
  }, []);

  return (
    <div className="bg-white border-b border-[#d9d9d9] p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-8">
          <h2 className="text-[#1975d4] text-2xl font-semibold">Good Evening Yan</h2>
          {editMode ? (
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="text-[#1975d4] text-2xl font-semibold border border-[#d9d9d9] rounded p-2 w-64 focus:outline-none focus:ring-2 focus:ring-[#1975d4]"
              placeholder="Enter project name"
              aria-label="Edit project name"
            />
          ) : (
            <h3 className="text-[#1975d4] text-2xl font-semibold">{projectName}</h3>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[#9ea2ae] text-sm">last updated: Jul 12, 2025 - {currentTime} IST</span>
          <div className="flex items-center gap-2">
            <span className="text-[#1975d4] font-semibold text-lg bg-white border-2 border-[#1975d4] px-4 py-1 rounded">Edit Mode</span>
            <label className="relative inline-flex items-center cursor-pointer" aria-label="Toggle Edit Mode">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={editMode}
                onChange={() => setEditMode(!editMode)}
              />
              <div className="w-14 h-7 bg-gray-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[#1975d4] rounded-full peer peer-checked:bg-[#1975d4] peer-checked:after:translate-x-7 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:border-gray-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all"></div>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};