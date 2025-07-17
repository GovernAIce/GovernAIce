const Card = ({ className = "", children, ...props }) => (
  <div className={`bg-white rounded-lg p-5 ${className}`} {...props}>
    {children}
  </div>
);