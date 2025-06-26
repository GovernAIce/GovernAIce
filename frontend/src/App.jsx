import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UploadButton from './components/UploadButton';
import ProjectCard from './components/ProjectCard';
import GapAnalysis from './components/GapAnalysis';

function App() {
  const [projects, setProjects] = useState([]);
  const [selectedFramework, setSelectedFramework] = useState('EU_AI_ACT');

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/documents/');
      const data = await response.json();
      setProjects(data.map(doc => ({
        name: doc.title,
        status: doc.framework_selections[0]?.analysis_status || 'Pending',
        lastUpdate: new Date(doc.upload_date).toLocaleString(),
        doc_id: doc.doc_id
      })));
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('frameworks', selectedFramework);
    formData.append('frameworks', 'CPRA'); // Adding CPRA as per Postman example

    try {
      const response = await fetch('http://localhost:8000/api/upload-and-analyze/', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setProjects(prev => [...prev, {
          name: data.filename,
          status: 'Processing',
          lastUpdate: new Date().toLocaleString(),
          doc_id: data.doc_id
        }]);
      }
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  return (
    <div className="container">
      <Navbar />
      <h1 className="title">Stress test your AI for compliance, bias, and safety</h1>
      <p className="subtitle">Actionable insights. Practical recommendations. All grounded in real-world benchmarks, regulations, and open-source tools.</p>
      <UploadButton onUpload={handleUpload} setSelectedFramework={setSelectedFramework} />
      <div className="grid">
        <div className="projects-column">
          <h2 className="section-title">Projects</h2>
          {projects.map((project, index) => (
            <ProjectCard key={index} {...project} />
          ))}
          <button className="view-report-btn" onClick={fetchDocuments}>Refresh</button>
        </div>
        <div className="gap-analysis-column">
          {projects.length > 0 && <GapAnalysis docId={projects[0].doc_id} framework={selectedFramework} />}
        </div>
      </div>
    </div>
  );
}

export default App;