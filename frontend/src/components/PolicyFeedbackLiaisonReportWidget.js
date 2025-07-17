const PolicyFeedbackLiaisonReportWidget = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <Card className="custom-border fixed inset-0 bg-white bg-opacity-90 z-50 p-6 overflow-y-auto">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl text-[#1975d4] font-bold">Policy Feedback & Liaison Report</h3>
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
        <RelevantRegulatorsWidget />
        <ContactsWidget />
        <GenerateConsultationFeedbackWidget />
      </div>
    </Card>
  );
};