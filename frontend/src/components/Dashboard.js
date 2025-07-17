const Dashboard = ({ selectedMenuItem }) => {
  return (
    <div className="grid grid-cols-3 gap-4 p-6">
      {selectedMenuItem === 0 ? (
        <>
          <UploadProjectWidget />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget />
          <OverallScoreWidget />
          <PoliciesRegulatorsWidget />
          <ChatWithMeWidget />
          <ExcellenciesMajorGapsWidget />
        </>
      ) : selectedMenuItem === 3 ? (
        <>
          <UploadProjectWidget />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget />
          <OECDScoreWidget />
          <NISTAILifecycleWidget />
          <EURiskLevelFrameworkWidget />
          <ExcellenciesMajorGapsWidget />
          <ChatWithMeWidget />
        </>
      ) : selectedMenuItem === 7 ? (
        <>
          <UploadProjectWidget />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget />
          <div>
            <Card className="custom-border p-4">
              <h3 className="text-xl text-[#1975d4] font-bold">Compliance Report Based on the EU AI Act</h3>
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
        </>
      ) : selectedMenuItem === 8 ? (
        <>
          <UploadProjectWidget />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget />
          <RelevantRegulatorsWidget />
          <ContactsWidget />
          <GenerateConsultationFeedbackWidget />
        </>
      ) : selectedMenuItem === 9 ? (
        <>
          <div className="col-span-3 grid grid-cols-2 gap-4 bg-[#cde8ff] p-4 rounded-xl">
            <UploadProjectWidget />
            <ExplorePolicyWidget />
          </div>
          <div className="col-span-3 mt-4">
            <Card className="bg-white rounded-xl p-4">
              <h3 className="text-xl text-[#1975d4] font-bold mb-4">Results</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-[#f5f9ff] p-4 rounded-lg">
                  <h4 className="text-[#1975d4] font-bold text-lg mb-2">Government-led Incentives</h4>
                  <p className="text-sm text-black">
                    <strong>Subsidies:</strong>
                    <ul className="list-disc pl-5 mt-1">
                      <li>State Labor Development Commission - Career Development Subsidies</li>
                    </ul>
                    <br />
                    <strong>Investment:</strong>
                    <ul className="list-disc pl-5 mt-1">
                      <li>Federal and Cal launched investment</li>
                    </ul>
                    <br />
                    <strong>National Strategies:</strong>
                    <ul className="list-disc pl-5 mt-1">
                      <li>Singapore listed Employment as one of its core domains to adopt AI</li>
                      <li>UK listed Labor Development as one of its pilot initiatives for AI development</li>
                    </ul>
                  </p>
                </div>
                <div className="bg-[#f5f9ff] p-4 rounded-lg">
                  <h4 className="text-[#1975d4] font-bold text-lg mb-2">Application Support</h4>
                  <p className="text-sm text-black">
                    Based on these government-led business opportunities, Cuscal could develop the following strategies to apply for these fundings and credits:
                    <ol className="list-decimal pl-5 mt-2">
                      <li>Engage with state labor commissions for career development grants.</li>
                      <li>Apply for federal investment programs in workforce development.</li>
                      <li>Align product strategy with Singapore and UK’s national AI goals for international eligibility.</li>
                    </ol>
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </>
      ) : (
        <>
          <UploadProjectWidget />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget />
          <OverallScoreWidget />
          <PoliciesRegulatorsWidget />
          <ChatWithMeWidget />
          <ExcellenciesMajorGapsWidget />
        </>
      )}
    </div>
  );
};