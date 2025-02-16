import React, { useState } from "react";

const ChatInterface = ({ onSendMessage, messages }) => {
  const [message, setMessage] = useState("");

  const handleSend = () => {
    if (message.trim()) {
      onSendMessage({ text: message });
      setMessage("");
    }
  };

  return (
    <div>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <p key={index} className={msg.sender === "user" ? "user-msg" : "ai-msg"}>
            <strong>{msg.sender === "user" ? "You: " : "AI: "}</strong>
            {msg.text}
          </p>
        ))}
      </div>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
};

export default ChatInterface;