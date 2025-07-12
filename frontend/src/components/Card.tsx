import React from 'react';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ className = '', children, ...props }) => (
  <div className={`bg-white rounded-2xl shadow-lg p-5 h-full ${className}`} {...props}>
    {children}
  </div>
);

export default Card; 
