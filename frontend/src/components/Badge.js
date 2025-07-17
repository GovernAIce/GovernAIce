const Badge = ({ className = "", children, ...props }) => (
  <span className={`px-3 py-1 rounded-full bg-[#1975d4] text-white ${className}`} {...props}>
    {children}
  </span>
);