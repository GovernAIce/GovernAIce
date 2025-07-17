const GenerateConsultationFeedbackWidget = () => (
  <Card className="custom-border p-4">
    {React.createElement(icons["Info"], { className: "absolute top-2 right-2 w-4 h-4 cursor-pointer" })}
    <div className="flex flex-col gap-2.5">
      <h3 className="text-xl text-[#1975d4] font-bold">Generate Consultation/Feedback</h3>
      <p className="text-sm text-black">
        National Institute of Standards and Technology
        <br /><br />
        To whom it may concern,
        <br /><br />
        Casual is an AI startup committed to facilitating talent discovery, hiring, and development. Today, we are writing as the compliance team to provide an opinion on the NIST's recent industry standards consultation. Specifically, today we will be responding to section 1(a) of the policy, concerning "privacy analysis of the AI..."
      </p>
    </div>
  </Card>
);