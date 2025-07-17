const RiskChecklistWidget = () => (
  <Card className="custom-border p-4">
    {React.createElement(icons["Info"], { className: "absolute top-2 right-2 w-4 h-4 cursor-pointer" })}
    <div className="flex flex-col gap-2.5">
      <h3 className="text-xl text-[#1975d4] font-bold">Risk Checklist based on NIST Risk Management</h3>
      <p className="text-sm text-black">
        This HR product is at the pilot stage. According to the NIST Risk Management Framework, this product’s life cycle includes four stages and six major risks. Here is the list for checking:
        <ul className="list-disc pl-5 mt-2">
          <li>Plan and design</li>
          <li>Collect data</li>
          <li>Build the model</li>
        </ul>
      </p>
    </div>
  </Card>
);