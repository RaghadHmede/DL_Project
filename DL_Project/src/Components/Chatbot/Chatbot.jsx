import React, { useState, useRef, useEffect } from "react";
import ChatbotIcon from "./ChatbotIcon";
import ChatForm from "./ChatForm";
import ChatMessage from "./ChatMessage";
import "./ChatBot.css";
import ParticlesComponent from "../Particles/Particles";

const Chatbot = () => {
  const [chatHistory, setChatHistory] = useState([
    {
      role: "model",
      text: "Bonjour 👋 Entrez une phrase en français, et je la traduirai en anglais !",
    },
  ]);
  const [isListening, setIsListening] = useState(false);
  const [loadingTranslation, setLoadingTranslation] = useState(false);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState(false);
  const chatBodyRef = useRef();
  const recognition = useRef(null);

  useEffect(() => {
    if ("webkitSpeechRecognition" in window) {
      recognition.current = new window.webkitSpeechRecognition();
      recognition.current.lang = "fr-FR";
      recognition.current.interimResults = false;
      recognition.current.maxAlternatives = 1;

      recognition.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log("Transcript:", transcript);

        setChatHistory((prev) => {
          const newHistory = [...prev, { role: "user", text: transcript }];
          if (!isGeneratingResponse && !loadingTranslation) {
            generateBotResponse(newHistory);
          }
          return newHistory;
        });
      };

      recognition.current.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
      };

      recognition.current.onend = () => {
        console.log("Speech recognition ended.");
        setIsListening(false);
      };
    } else {
      console.error("Speech recognition not supported in this browser.");
    }
  }, [isGeneratingResponse, loadingTranslation]);

  const startListening = () => {
    if (recognition.current) {
      setIsListening(true);
      recognition.current.start();
      console.log("Speech recognition started.");
    }
  };

  const stopListening = () => {
    if (recognition.current) {
      recognition.current.stop();
      console.log("Speech recognition stopped.");
    }
  };

  const translateSentence = async (sentence) => {
    try {
      console.log("Sending request to translate:", sentence);
      const response = await fetch("http://127.0.0.1:5000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence }),
      });

      if (!response.ok) {
        throw new Error(`Translation failed with status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received translation:", data);
      return data.translation || "Translation not found.";
    } catch (error) {
      console.error("Error during translation:", error);
      return "❌ Une erreur est survenue pendant la traduction.";
    }
  };

  const generateBotResponse = async (history) => {
    if (loadingTranslation || isGeneratingResponse) return;

    setIsGeneratingResponse(true);
    setLoadingTranslation(true);
    setChatHistory((prev) => [
      ...prev,
      { role: "model", text: "⏳ Traduction en cours..." },
    ]);

    const lastUserMessage = history[history.length - 1].text;

    try {
      const translation = await translateSentence(lastUserMessage);

      setChatHistory((prev) => {
        const updated = prev.filter(
          (msg) => msg.text !== "⏳ Traduction en cours..."
        );
        return [...updated, { role: "model", text: translation }];
      });
    } catch (error) {
      setChatHistory((prev) => {
        const updated = prev.filter(
          (msg) => msg.text !== "⏳ Traduction en cours..."
        );
        return [
          ...updated,
          { role: "model", text: "Sorry, something went wrong." },
        ];
      });
    } finally {
      setIsGeneratingResponse(false);
      setLoadingTranslation(false);
    }
  };

  return (
    <div className="container show-chatbot">
      <div className="particles-wrapper">
        <ParticlesComponent id="Particles" />
      </div>
      <div className="chatbot-popup">
        <div className="chat-header">
          <div className="header-info">
            <ChatbotIcon />
            <h2 className="logo-text">French-to-English Chatbot</h2>
          </div>
        </div>

        <div ref={chatBodyRef} className="chat-body">
          {chatHistory.map((chat, index) => (
            <ChatMessage key={index} chat={chat} />
          ))}
        </div>

        <div className="chat-footer">
          <button
            className="microphone-button"
            onClick={isListening ? stopListening : startListening}
          >
            {isListening ? "Stop" : "Speak"}
          </button>
          <ChatForm
            chatHistory={chatHistory}
            setChatHistory={setChatHistory}
            generateBotResponse={generateBotResponse}
          />
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
