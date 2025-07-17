const App = () => {
  const [projectName, setProjectName] = React.useState("Project Name");
  const [selectedMenuItem, setSelectedMenuItem] = React.useState(null);

  return (
    <div className="flex h-screen bg-[#e6f0fa]">
      <Menu projectName={projectName} onSelect={setSelectedMenuItem} />
      <div className="flex-1 flex flex-col">
        <Header projectName={projectName} setProjectName={setProjectName} />
        <Dashboard selectedMenuItem={selectedMenuItem} />
      </div>
    </div>
  );
};