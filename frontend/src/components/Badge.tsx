import React from 'react';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  className?: string;
  children: React.ReactNode;
}

const Badge: React.FC<BadgeProps> = ({ className = '', children, ...props }) => (
  <span className={`px-3 py-1 rounded-full bg-[#1975d4] text-white ${className}`} {...props}>
    {children}
  </span>
);

export default Badge; 
