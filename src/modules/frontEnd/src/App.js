import React, { useState } from "react";
import ChatComponent from "./components/ChatComponent";
import RenderQA from "./components/RenderQA";
import LLMSelector from "./components/LLMSelector";
import { Layout, Typography, Space, Button } from "antd";
import { GithubOutlined, ExperimentOutlined } from "@ant-design/icons";

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;

const chatComponentStyle = {
  position: "fixed",
  bottom: "0",
  width: "80%",
  left: "10%",
  marginBottom: "20px",
  backgroundColor: "white",
  padding: "16px",
  borderRadius: "8px",
  boxShadow: "0 -2px 8px rgba(0,0,0,0.1)",
  zIndex: 1000,
};

const renderQAStyle = {
  marginTop: "80px",
  marginBottom: "120px",
  minHeight: "calc(100vh - 250px)",
  overflowY: "auto",
};

const controlsStackStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 16,
  width: "100%",
};

const App = () => {
  const [conversation, setConversation] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [reasoningEnabled, setReasoningEnabled] = useState(true);

  // Default to OpenAI GPT-4o-Mini to stay consistent with backend `/set-llm`
  const [selectedLLM, setSelectedLLM] = useState("gpt-4o-mini");

  const handleResp = (question, answer) => {
    setConversation((prev) =>
      prev.map((entry) =>
        entry.question === question ? { ...entry, answer } : entry
      )
    );
  };

  const addQuestion = (question) => {
    setConversation((prev) => [...prev, { question, answer: null }]);
  };

  const toggleReasoning = async () => {
    try {
      const response = await fetch('http://localhost:5005/toggle-reasoning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enable: !reasoningEnabled }),
      });
      const data = await response.json();
      setReasoningEnabled(data.reasoning_enabled);
    } catch (error) {
      console.error('Error toggling reasoning:', error);
    }
  };

  return (
    <Layout style={{ minHeight: "100vh", backgroundColor: "white" }}>
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "fixed",
          width: "100%",
          zIndex: 1001,
          backgroundColor: "#001529",
        }}
      >
        <Space>
          <ExperimentOutlined style={{ fontSize: "24px", color: "white" }} />
          <Title level={3} style={{ color: "white", margin: 0 }}>
            AI Fact-Checking System
          </Title>
        </Space>
        <Space>
          <Button
            type={reasoningEnabled ? "primary" : "default"}
            onClick={toggleReasoning}
            size="small"
          >
            Reasoning: {reasoningEnabled ? "ON" : "OFF"}
          </Button>
          <Button
            type="link"
            href="https://github.com/yourusername/fact-checking"
            target="_blank"
            icon={<GithubOutlined />}
            style={{ color: "white" }}
          >
            GitHub
          </Button>
        </Space>
      </Header>

      <Content style={{ width: "80%", margin: "auto" }}>
        <div style={renderQAStyle}>
          <RenderQA conversation={conversation} isLoading={isLoading} />
        </div>
      </Content>

      <div style={chatComponentStyle}>
        <div style={controlsStackStyle}>
          <LLMSelector
            selectedLLM={selectedLLM}
            onChange={setSelectedLLM}
            disabled={isLoading}
          />
          <ChatComponent
            handleResp={handleResp}
            addQuestion={addQuestion}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            selectedLLM={selectedLLM}
          />
        </div>
      </div>

      <Footer
        style={{
          textAlign: "center",
          backgroundColor: "#f0f2f5",
          position: "fixed",
          bottom: 0,
          width: "100%",
          zIndex: 999,
        }}
      >
        <Text type="secondary">
          CMPE297 Fact-Checking System | Powered by LLM Reasoning
        </Text>
      </Footer>
    </Layout>
  );
};

export default App;
