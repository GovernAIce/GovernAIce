import React, { useState, useRef } from 'react';
import Card from './Card';

interface ChatWithMeWidgetProps {
  className?: string;
}

interface ChatMessage {
  sender: 'user' | 'bot';
  message: string;
}

const ChatWithMeWidget: React.FC<ChatWithMeWidgetProps> = ({ className = '' }) => {
  const [message, setMessage] = useState('');
  const [history, setHistory] = useState<ChatMessage[]>([
    { sender: 'bot', message: 'I am Govii, how can I help you with compliance project today?' }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const sendMessage = async () => {
    if (!message.trim()) return;
    setHistory(h => [...h, { sender: 'user', message }]);
    setMessage('');
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      if (data.reply) {
        setHistory(h => [...h, { sender: 'bot', message: data.reply }]);
      } else {
        setError('No reply from server.');
      }
    } catch (e) {
      setError('Failed to send message.');
    } finally {
      setLoading(false);
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Card className={`custom-border relative p-4 flex flex-col justify-between ${className}`}>
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-3 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col gap-2.5 flex-1 overflow-y-auto mb-2" style={{ maxHeight: '320px' }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="w-6 h-6 bg-[#1975d4] rounded-full flex items-center justify-center">
            <span className="text-white text-sm">😊</span>
          </div>
          <h3 className="text-xl text-[#1975d4] font-bold">Chat with Me</h3>
        </div>
        <div className="flex flex-col gap-2 overflow-y-auto" style={{ minHeight: 120 }}>
          {history.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`rounded-lg px-3 py-2 max-w-[80%] text-sm whitespace-pre-line ${
                  msg.sender === 'user'
                    ? 'bg-[#1975d4] text-white self-end'
                    : 'bg-gray-100 text-gray-900 self-start'
                }`}
              >
                {msg.message}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-lg px-3 py-2 bg-gray-100 text-gray-400 text-sm">Govii is typing...</div>
            </div>
          )}
          {error && (
            <div className="flex justify-start">
              <div className="rounded-lg px-3 py-2 bg-red-100 text-red-700 text-sm">{error}</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>
      <div className="flex items-center border-2 border-gray-300 rounded-full p-2 mt-4">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type here and press Enter to send"
          className="flex-1 text-sm text-gray-700 bg-transparent outline-none resize-none border-none h-8"
          rows={1}
        />
        <button
          className="w-8 h-8 flex items-center justify-center text-[#1975d4] ml-2 disabled:opacity-50"
          onClick={sendMessage}
          disabled={loading || !message.trim()}
          aria-label="Send"
        >
          <span className="text-2xl">→</span>
        </button>
      </div>
    </Card>
  );
};

export default ChatWithMeWidget; 
