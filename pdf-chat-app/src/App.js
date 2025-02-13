import React, { useState } from 'react';
import PdfUploader from './components/PdfUploader';
import ChatInterface from './components/ChatInterface';
import './App.css';

const App = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [messages, setMessages] = useState([]); // Added missing state

  const handlePdfUpload = (file) => {
    setPdfFile(file);
  };

  const handleSendMessage = async (message) => {
    const formData = new FormData();
    formData.append('message', message.text);
    if (pdfFile) {
      formData.append('pdf', pdfFile);
    }

    const response = await fetch('/api/rag', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setMessages((prevMessages) => [...prevMessages, { text: data.response, sender: 'rag' }]);
  };

  return (
    <div className="App">
      <h1>PDF Chat App</h1>
      <PdfUploader onUpload={handlePdfUpload} />
      <ChatInterface onSendMessage={handleSendMessage} messages={messages} />
    </div>
  );
};

export default App;
