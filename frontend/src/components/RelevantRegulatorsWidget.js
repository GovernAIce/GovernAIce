const RelevantRegulatorsWidget = () => (
  <Card className="custom-border p-4">
    {React.createElement(icons["Info"], { className: "absolute top-2 right-2 w-4 h-4 cursor-pointer" })}
    <div className="flex flex-col gap-2.5">
      <h3 className="text-xl text-[#1975d4] font-bold">Relevant Regulators</h3>
      <p className="text-sm text-black">
        The U.S.A.
        <ul className="list-disc pl-5 mt-2">
          <li>Department of Labor and Employment</li>
          <li>State Labor Development Commission</li>
        </ul>
        EU
        <ul className="list-disc pl-5 mt-2">
          <li>Labor Development Bureau</li>
        </ul>
        Japan
        <ul className="list-disc pl-5 mt-2">
          <li>National Labor Administration</li>
        </ul>
      </p>
    </div>
  </Card>
);