import React, { useState } from 'react';
import PdfUploader from './components/PdfUploader';
import ChatInterface from './components/ChatInterface';
import './App.css';

const App = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [messages, setMessages] = useState([]);

  const handlePdfUpload = (file) => {
    setPdfFile(file);
    // You can send the file to your server here if needed
  };

  const handleSendMessage = async (message) => {
    // Replace this with your API call to the RAG model
    const response = await fetch('/api/rag', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: message.text, pdf: pdfFile }),
    });
    const data = await response.json();
    // Update the chat interface with the response from the RAG model
    setMessages((prevMessages) => [...prevMessages, { text: data.response, sender: 'rag' }]);
  };

  return (
    <div className="App">
      <h1>PDF Chat App</h1>
      <PdfUploader onUpload={handlePdfUpload} />
      <ChatInterface onSendMessage={handleSendMessage} />
    </div>
  );
};

export default App;