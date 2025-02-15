import React, { useState } from "react";
import PdfUploader from "./components/PdfUploader";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

const App = () => {
  const [pdfUploaded, setPdfUploaded] = useState(false);
  const [messages, setMessages] = useState([]);

  const handlePdfUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setPdfUploaded(true);
        alert("PDF uploaded and processed successfully!");
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert("Upload failed. Please try again.");
      console.error(error);
    }
  };

  const handleSendMessage = async (message) => {
    if (!pdfUploaded) {
      alert("Please upload a PDF first!");
      return;
    }

    setMessages((prevMessages) => [...prevMessages, { text: message.text, sender: "user" }]);

    try {
      // Step 1: Retrieve relevant chunks from the PDF
      const queryResponse = await fetch("http://localhost:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message.text }),
      });

      const queryData = await queryResponse.json();
      console.log("Query Response:", queryData);

      if (!queryResponse.ok) {
        alert(`Error: ${queryData.error}`);
        return;
      }

      // Step 2: Generate AI response using LLM
      const generateResponse = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message.text }),
      });

      const generateData = await generateResponse.json();
      console.log("Generated Response:", generateData);

      if (!generateResponse.ok) {
        alert(`Error: ${generateData.error}`);
        return;
      }

      setMessages((prevMessages) => [
        ...prevMessages,
        { text: generateData.response, sender: "rag" },
      ]);
    } catch (error) {
      console.error("Error processing query:", error);
      alert("Something went wrong. Try again later.");
    }
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