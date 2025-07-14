import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

const Input: React.FC<InputProps> = ({ className = '', ...props }) => (
  <input className={`w-full border border-[#d9d9d9] rounded p-2 ${className}`} {...props} />
);

export default Input; 
