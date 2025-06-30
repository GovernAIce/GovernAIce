import React, { useState } from 'react';

const UploadButton = ({ onUpload, setSelectedFramework }) => {
  const [file, setFile] = useState(null);
  const [framework, setFramework] = useState('EU_AI_ACT');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleFrameworkChange = (e) => {
    setFramework(e.target.value);
    setSelectedFramework(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) {
      onUpload(file);
      setFile(null);
    }
  };

  return (
    <div className="upload-section">
      <form onSubmit={handleSubmit} className="upload-form">
        <input
          type="file"
          onChange={handleFileChange}
          className="file-input"
        />
        <select
          value={framework}
          onChange={handleFrameworkChange}
          className="framework-select"
        >
          <option value="EU_AI_ACT">EU AI Act</option>
          <option value="NIST_RMF">NIST AI RMF</option>
          <option value="CPRA">CPRA</option>
        </select>
        <button
          type="submit"
          disabled={!file}
          className="upload-btn"
        >
          Upload Document
        </button>
      </form>
    </div>
  );
};

export default UploadButton;