import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'ghost' | 'outline';
  className?: string;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({ variant = 'default', className = '', children, ...props }) => (
  <button
    className={`px-4 py-2 rounded-[50px] ${
      variant === 'ghost'
        ? 'bg-transparent text-[#9ea2ae] hover:text-[#000000]'
        : variant === 'outline'
        ? 'border border-[#d9d9d9] bg-transparent text-[#9ea2ae] hover:text-[#000000]'
        : 'bg-gradient-to-r from-[#9FD8FF] to-[#2196f3] text-white opacity-100 transition-opacity'
    } ${className}`}
    {...props}
  >
    {children}
  </button>
);

export default Button; 
