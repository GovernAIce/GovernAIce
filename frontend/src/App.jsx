import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import Menu from './components/Menu';
import Header from './components/Header';
import PolicyAnalysis from './components/PolicyAnalysis';
import ComplianceRiskAssessment from './components/ComplianceRiskAssessment';
import MLTestWidget from './components/MLTestWidget';
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
          <Route path="/ml-test-widget" element={<MLTestWidget />} />
          <Route path="*" element={<Navigate to="/policy-analysis" replace />} />
        </Routes>
      </div>
    </div>
  );
};

const App = () => (
  <CountryProvider>
    <Router>
      <AppLayout />
    </Router>
  </CountryProvider>
);

export default App;
