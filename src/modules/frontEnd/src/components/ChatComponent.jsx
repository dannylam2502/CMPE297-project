import React, { useState } from 'react';
import { Input, Button, Select } from 'antd';
import './ChatComponent.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5005';

const LLM_OPTIONS = [
  {
    label: 'GPT-4o Mini',
    value: 'gpt-4o-mini',
    provider: 'openai',
    description: 'OpenAI lightweight reasoning model',
  },
  {
    label: 'Llama 3.1',
    value: 'llama3.1',
    provider: 'ollama',
    description: 'Ollama on-prem deployment',
  },
];

const PROVIDER_BY_VALUE = LLM_OPTIONS.reduce((acc, option) => {
  acc[option.value] = option.provider;
  return acc;
}, {});

const ChatComponent = ({
  handleResp,
  addQuestion,
  isLoading,
  setIsLoading,
  selectedLLM,
  onModelChange = () => {},
}) => {
  const [question, setQuestion] = useState('');
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    // Check backend is ready
    try {
      await fetch('http://localhost:5005/health');
    } catch {
      alert('Backend not ready yet, please wait...');
      return;
    }
    const currentQuestion = question;
    setQuestion('');
    addQuestion(currentQuestion);
    setIsLoading(true);

    try {
      const response = await fetch(`http://localhost:5005/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          llm: selectedLLM,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Format the response to include all fact-check data
      const formattedAnswer = {
        claim: data.claim,
        verdict: data.verdict,
        score: data.score,
        explanation: data.explanation,
        citations: data.citations,
        features: data.features,
        formatted_text: data.formatted_text
      };

      handleResp(currentQuestion, formattedAnswer);
    } catch (error) {
      console.error('Error fetching response:', error);
      handleResp(currentQuestion, {
        error: true,
        message: `Error: ${error.message}`,
        verdict: 'Error',
        score: 0
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSelect = async (value) => {
    if (value === selectedLLM) return;
    onModelChange(value);
    const provider = PROVIDER_BY_VALUE[value];
    if (!provider) return;

    setIsSwitchingModel(true);
    try {
      const response = await fetch(`${API_URL}/set-llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ llm_provider: provider }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      await response.json();
    } catch (error) {
      console.error('Failed to set backend LLM provider:', error);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  return (
    <div className="chat-bar-wrapper">
      <div className="chat-bar">
        <div className="chat-select-group">
          <span className="chat-label">Model</span>
          <Select
            value={selectedLLM}
            onChange={handleModelSelect}
            disabled={isLoading || isSwitchingModel}
            options={LLM_OPTIONS}
            popupClassName="chat-select-dropdown"
            className="chat-model-select"
            loading={isSwitchingModel}
            dropdownStyle={{
              background: '#14151a',
            }}
          />
        </div>
        <div className="chat-input-wrapper">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onPressEnter={(e) => {
              e.preventDefault();
              handleSubmit();
            }}
            placeholder="Ask about any player, team, or game..."
            disabled={isLoading || isSwitchingModel}
            className="chat-input"
            bordered={false}
          />
        </div>
        <Button
          type="primary"
          className="chat-check-button"
          onClick={handleSubmit}
          disabled={isLoading || !question.trim() || isSwitchingModel}
          loading={isLoading}
          size="large"
        >
          Check
        </Button>
      </div>
    </div>
  );
};

export default ChatComponent;
