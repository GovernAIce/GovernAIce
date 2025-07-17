const ComplianceReportWidget = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <Card className="custom-border fixed inset-0 bg-white bg-opacity-90 z-50 p-6 overflow-y-auto">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl text-[#1975d4] font-bold">Compliance Report Suite</h3>
        <button
          onClick={onClose}
          className="text-[#9ea2ae] hover:text-[#000000] text-2xl"
        >
          ×
        </button>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <UploadProjectWidget />
        <ExplorePolicyWidget />
        <RelevantPoliciesWidget />
        <div>
          <Card className="p-4">
            <h4 className="text-[#1975d4] font-bold">Compliance Report Based on the EU AI Act</h4>
            <p className="text-sm text-black">
              This HR product does not include Level 3 - High Risk and Level 4 - Unacceptable Risk. However, it may contain Level 2 - Limited Risk given its features such as...
              <br /><br />
              Thus, the following measures should be taken into consideration...
              <br /><br />
              The EU AI Office also requires AI companies to get these certifications before full launch, including...
            </p>
          </Card>
        </div>
        <RiskChecklistWidget />
        <StrategyArchitectureWidget />
      </div>
    </Card>
  );
};