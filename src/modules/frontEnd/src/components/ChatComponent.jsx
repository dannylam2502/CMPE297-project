import React, { useState } from 'react';
import { Input, Button, Space } from 'antd';
import { SendOutlined } from '@ant-design/icons';

const { TextArea } = Input;

const ChatComponent = ({ handleResp, addQuestion, isLoading, setIsLoading }) => {
  const [question, setQuestion] = useState('');

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
        body: JSON.stringify({ question: currentQuestion }),
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Space.Compact style={{ width: '100%' }} size="large">
      <TextArea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Enter a claim to fact-check..."
        autoSize={{ minRows: 2, maxRows: 4 }}
        disabled={isLoading}
        style={{ fontSize: '16px' }}
      />
      <Button
        type="primary"
        icon={<SendOutlined />}
        onClick={handleSubmit}
        disabled={isLoading || !question.trim()}
        loading={isLoading}
        size="large"
        style={{ height: 'auto' }}
      >
        Check
      </Button>
    </Space.Compact>
  );
};

export default ChatComponent;
