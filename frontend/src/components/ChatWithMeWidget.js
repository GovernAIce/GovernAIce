const ChatWithMeWidget = () => {
  const [message, setMessage] = React.useState('');

  return (
    <Card className="custom-border relative p-4 h-full flex flex-col justify-between">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col gap-2.5">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-[#1975d4] rounded-full flex items-center justify-center">
            <span className="text-white text-sm">😊</span>
          </div>
          <h3 className="text-xl text-[#1975d4] font-bold">Chat with Me</h3>
        </div>
        <p className="text-sm text-black">
          I am Govii, how can I help you with compliance project today?
        </p>
      </div>
      <div className="flex items-center border-2 border-gray-300 rounded-full p-2 mt-4">
        <Input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type Here"
          className="flex-1 text-sm text-gray-700 bg-transparent outline-none"
        />
        {React.createElement(icons["Send"], { className: "w-5 h-5 text-[#1975d4]" })}
      </div>
    </Card>
  );
};