const ContactsWidget = () => (
  <Card className="custom-border p-4">
    {React.createElement(icons["Info"], { className: "absolute top-2 right-2 w-4 h-4 cursor-pointer" })}
    <div className="flex flex-col gap-2.5">
      <h3 className="text-xl text-[#1975d4] font-bold">Contacts</h3>
      <p className="text-sm text-black">
        These are the official contact information...
        <br /><br />
        You can also use these channels to provide policy feedback and consultations...
        <br /><br />
        X: ...
        <br />
        TikTok: ...
        <br />
        Website: ...
      </p>
    </div>
  </Card>
);