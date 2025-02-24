import React, { useState } from "react";
import PdfUploader from "./components/PdfUploader";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

const API_BASE_URL = "http://localhost:5000"; // Ensure this matches your backend

const App = () => {
  const [pdfUploaded, setPdfUploaded] = useState(false);
  const [pdfFileName, setPdfFileName] = useState(""); // Store uploaded PDF name
  const [messages, setMessages] = useState([]);

  const handlePdfUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setPdfUploaded(true);
        setPdfFileName(file.name); // Store filename for reference
        alert("PDF uploaded and processed successfully!");
      } else {
        alert(`Error: ${data.error || "Unknown error"}`);
      }
    } catch (error) {
      alert("Upload failed. Please try again.");
      console.error("Upload error:", error);
    }
  };

  /**
   * The `handleSendMessage` function sends a message to a server, processes the response, and updates
   * the messages displayed in the user interface accordingly.
   * @param messageText - The `handleSendMessage` function you provided is an asynchronous function
   * that handles sending a message. It first adds the user's message to the messages array, then makes
   * a POST request to a specified API endpoint with the user's query. Depending on the response
   * received, it updates the messages array with the retrieved
   */
  const handleSendMessage = async (messageText) => {
    // if (!pdfUploaded) {
    //   alert("Please upload a PDF first!");
    //   return;
    // }

    setMessages((prevMessages) => [...prevMessages, { text: messageText, sender: "user" }]);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: messageText }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      console.log("Response received:", data);

      if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        const responseMessage = {
          text: `📄 Retrieved Info:\n\"${data.retrieved_chunks[0].text_chunk}\" (Page ${data.retrieved_chunks[0].page_number})\n\n🤖 AI Response:\n${data.generated_response}`,
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, responseMessage]);
      } else {
        setMessages((prevMessages) => [...prevMessages, { text: "No relevant data found.", sender: "bot" }]);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      setMessages((prevMessages) => [...prevMessages, { text: "Error processing query.", sender: "bot" }]);
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
