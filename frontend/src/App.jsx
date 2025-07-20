<<<<<<< HEAD
<<<<<<< HEAD
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
=======
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import Menu from './components/Menu';
import Header from './components/Header';
import PolicyAnalysis from './components/PolicyAnalysis';
import ComplianceRiskAssessment from './components/ComplianceRiskAssessment';
import { CountryProvider } from './contexts/CountryContext';
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce

const AppLayout = () => {
  const [projectName, setProjectName] = useState('Project Name');
  const navigate = useNavigate();

  return (
    <div className="flex h-screen bg-[#e6f0fa] p-4">
      <Menu projectName={projectName} onNavigate={navigate} />
      <div className="flex-1 flex flex-col ml-6 h-full">
        <div className="w-full bg-white rounded-3xl shadow p-6 mb-4" style={{ flex: '0 0 auto' }}>
          <Header projectName={projectName} setProjectName={setProjectName} />
        </div>
        <Routes>
          <Route path="/policy-analysis" element={<PolicyAnalysis />} />
          <Route path="/compliance-risk-assessment" element={<ComplianceRiskAssessment />} />
          <Route path="*" element={<Navigate to="/policy-analysis" replace />} />
        </Routes>
      </div>
    </div>
  );
};

<<<<<<< HEAD
export default App
=======
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import Menu from './components/Menu';
import Header from './components/Header';
import PolicyAnalysis from './components/PolicyAnalysis';
import ComplianceRiskAssessment from './components/ComplianceRiskAssessment';
import { CountryProvider } from './contexts/CountryContext';

const AppLayout = () => {
  const [projectName, setProjectName] = useState('Project Name');
  const navigate = useNavigate();

  return (
    <div className="flex h-screen bg-[#e6f0fa] p-4">
      <Menu projectName={projectName} onNavigate={navigate} />
      <div className="flex-1 flex flex-col ml-6 h-full">
        <div className="w-full bg-white rounded-3xl shadow p-6 mb-4" style={{ flex: '0 0 auto' }}>
          <Header projectName={projectName} setProjectName={setProjectName} />
        </div>
        <Routes>
          <Route path="/policy-analysis" element={<PolicyAnalysis />} />
          <Route path="/compliance-risk-assessment" element={<ComplianceRiskAssessment />} />
          <Route path="*" element={<Navigate to="/policy-analysis" replace />} />
        </Routes>
      </div>
    </div>
  );
};

=======
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
const App = () => (
  <CountryProvider>
    <Router>
      <AppLayout />
    </Router>
  </CountryProvider>
);

export default App;
<<<<<<< HEAD
>>>>>>> dfab21b9ab606bef9aae393aba8cefb32de97b9f
=======
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
