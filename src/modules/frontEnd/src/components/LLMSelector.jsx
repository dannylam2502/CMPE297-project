import React from "react";
import { Select, Space, Typography } from "antd";

const { Text } = Typography;

// Keep this list in lockstep with the Flask /set-llm backend.
const LLM_OPTIONS = [
  {
    label: "OpenAI GPT-4o-Mini",
    value: "gpt-4o-mini",
    description: "OpenAI lightweight reasoning model.",
    provider: "openai",
  },
  {
    label: "Ollama Llama-3.1",
    value: "llama3.1",
    description: "Ollama on-prem deployment model.",
    provider: "ollama",
  },
];

const containerStyle = {
  width: "100%",
  padding: "8px 0",
};

const labelStyle = {
  fontSize: "14px",
  fontWeight: 500,
  color: "rgba(0, 0, 0, 0.85)",
};

const optionLabel = (option) => (
  <div style={{ display: "flex", flexDirection: "column" }}>
    <Text strong style={{ fontSize: "14px" }}>
      {option.label}
    </Text>
    <Text type="secondary" style={{ fontSize: "12px" }}>
      {option.description}
    </Text>
  </div>
);

const LLMSelector = ({ selectedLLM, onChange, disabled }) => {
  // Bridge dropdown selections to the Flask /set-llm endpoint.
  const handleSelectChange = async (value) => {
    onChange(value);
    const selectedOption = LLM_OPTIONS.find((option) => option.value === value);
    if (!selectedOption) return;

    try {
      const response = await fetch("http://localhost:5005/set-llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_provider: selectedOption.provider }),
      });
      const data = await response.json();
      console.log("Backend confirmed model switch:", data);
    } catch (error) {
      console.error("Failed to set backend LLM provider:", error);
    }
  };

  return (
    <div style={containerStyle}>
      <Space direction="vertical" size={8} style={{ width: "100%" }}>
        <Text style={labelStyle}>Model preference</Text>
        <Select
          size="large"
          value={selectedLLM}
          onChange={handleSelectChange}
          disabled={disabled}
          style={{ width: "100%" }}
          options={LLM_OPTIONS.map((option) => ({
            value: option.value,
            label: optionLabel(option),
          }))}
        />
      </Space>
    </div>
  );
};

export default LLMSelector;
