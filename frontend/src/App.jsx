import React, { useState } from 'react';
import Menu from './components/Menu';
import Header from './components/Header';
import Dashboard from './components/Dashboard';

const App = () => {
  const [projectName, setProjectName] = useState('Project Name');
  console.log("App rendered");

  return (
    <div className="flex h-screen bg-[#e6f0fa] p-4">
      <Menu projectName={projectName} />
      <div className="flex-1 flex flex-col ml-6 h-full">
        <div className="w-full bg-white rounded-3xl shadow p-6 mb-4" style={{ flex: '0 0 auto' }}>
          <Header projectName={projectName} setProjectName={setProjectName} />
        </div>
        <Dashboard />
      </div>
    </div>
  );
};

export default App;
