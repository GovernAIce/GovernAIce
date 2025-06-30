// src/components/ProjectCard.jsx
import React from 'react';

const ProjectCard = ({ name, status, lastUpdate }) => (
  <div className="project-card">
    <span className="project-name">{name}</span>
    <span className={`project-status ${status.toLowerCase()}`}>
      {status}
    </span>
    <span className="project-update">{lastUpdate}</span>
  </div>
);

export default ProjectCard;