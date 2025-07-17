const ExplorePolicyWidget = () => {
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedCountry, setSelectedCountry] = React.useState("Country");
  const [selectedDomain, setSelectedDomain] = React.useState("Domain");

  return (
    <Card className="custom-border relative p-4">
      <img
        src="/public/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col gap-2.5">
        <h3 className="text-xl text-[#1975d4] font-bold">Explore Policy</h3>
        <div className="relative">
          {React.createElement(icons["Search"], { className: "absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-[#9ea2ae]" })}
          <Input
            placeholder="Search"
            className="pl-10 pr-10 text-sm text-gray-800 bg-transparent outline-none"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <icons.X
              className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-[#9ea2ae] cursor-pointer"
              onClick={() => setSearchQuery("")}
            />
          )}
        </div>
        <select
          className="custom-border rounded-lg p-2 text-sm w-full appearance-none bg-transparent"
          value={selectedCountry}
          onChange={(e) => setSelectedCountry(e.target.value)}
        >
          <option>Country</option>
          <option>New Country</option>
        </select>
        <select
          className="custom-border rounded-lg p-2 text-sm w-full appearance-none bg-transparent"
          value={selectedDomain}
          onChange={(e) => setSelectedDomain(e.target.value)}
        >
          <option>Domain</option>
          <option>New Domain</option>
        </select>
        <Button className="w-full text-white text-sm">Compare</Button>
      </div>
    </Card>
  );
};