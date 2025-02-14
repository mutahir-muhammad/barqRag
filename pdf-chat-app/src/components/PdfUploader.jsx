import React, { useState } from "react";
import * as pdfjsLib from "pdfjs-dist"; // Correct import
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";

// Make sure to use the correct worker URL.
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;

const PdfUploader = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [sentences, setSentences] = useState([]);

  const onFileChange = async (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(URL.createObjectURL(selectedFile));
      setError("");
      await extractSentencesFromPdf(selectedFile);
    } else {
      setError("Please select a valid PDF file.");
    }
  };

  const extractSentencesFromPdf = async (pdfFile) => {
    const fileBuffer = await pdfFile.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument(fileBuffer);
    const pdf = await loadingTask.promise;
    const extractedSentences = [];

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();

      textContent.items.forEach((item) => {
        const text = item.str;
        const splitSentences = text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
        extractedSentences.push(...splitSentences.map(s => s.trim()));
      });
    }

    setSentences(extractedSentences);
  };

  return (
    <div>
      <input type="file" accept="application/pdf" onChange={onFileChange} />
      {error && <p style={{ color: "red" }}>{error}</p>}
      <div>
        <h2>Extracted Sentences:</h2>
        <ul>
          {sentences.map((sentence, index) => (
            <p key={index}>{sentence}</p>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default PdfUploader;
