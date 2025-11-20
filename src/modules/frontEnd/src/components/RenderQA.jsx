import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card, Typography, Space, Divider, Spin, Badge, Collapse, List } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  QuestionCircleOutlined,
  ExclamationCircleOutlined,
  LinkOutlined
} from '@ant-design/icons';
import './RenderQA.css';

const { Text, Title, Paragraph } = Typography;
const { Panel } = Collapse;

const VerdictIcon = ({ verdict }) => {
  const iconProps = { style: { fontSize: '20px' } };

  switch (verdict) {
    case 'Supported':
      return <CheckCircleOutlined style={{ ...iconProps.style, color: '#52c41a' }} />;
    case 'Refuted':
      return <CloseCircleOutlined style={{ ...iconProps.style, color: '#ff4d4f' }} />;
    case 'Contested':
      return <ExclamationCircleOutlined style={{ ...iconProps.style, color: '#faad14' }} />;
    case 'Not enough evidence':
      return <QuestionCircleOutlined style={{ ...iconProps.style, color: '#8c8c8c' }} />;
    default:
      return <QuestionCircleOutlined style={iconProps} />;
  }
};

const VerdictBadge = ({ verdict, score }) => {
  let color = 'default';

  switch (verdict) {
    case 'Supported':
      color = 'success';
      break;
    case 'Refuted':
      color = 'error';
      break;
    case 'Contested':
      color = 'warning';
      break;
    case 'Not enough evidence':
      color = 'default';
      break;
    default:
      color = 'default';
  }

  return (
    <Badge
      count={`${score}/100`}
      style={{
        backgroundColor: color === 'success' ? '#52c41a' :
          color === 'error' ? '#ff4d4f' :
            color === 'warning' ? '#faad14' : '#8c8c8c'
      }}
    />
  );
};

const FeatureDisplay = ({ features }) => {
  if (!features) return null;

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="small">
      <Text strong>Evidence Analysis:</Text>
      <div style={{ paddingLeft: '16px' }}>
        <Text>â€¢ Max Entailment: <Text code>{features.entail_max?.toFixed(2) || 'N/A'}</Text></Text><br />
        <Text>â€¢ Max Contradiction: <Text code>{features.contradict_max?.toFixed(2) || 'N/A'}</Text></Text><br />
        <Text>â€¢ Agreeing Domains: <Text code>{features.agree_domain_count || 0}</Text></Text><br />
        <Text>â€¢ Avg Relevance: <Text code>{features.relevance_avg?.toFixed(2) || 'N/A'}</Text></Text>
      </div>
    </Space>
  );
};

const CitationList = ({ citations }) => {
  if (!citations || citations.length === 0) {
    return <Text type="secondary">No citations available</Text>;
  }

  return (
    <List
      size="small"
      dataSource={citations}
      renderItem={(item, index) => (
        <List.Item>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text strong>{index + 1}. {item.title}</Text>
            <a href={item.url} target="_blank" rel="noopener noreferrer">
              <LinkOutlined /> {item.url}
            </a>
            <Paragraph
              ellipsis={{ rows: 2, expandable: true, symbol: 'more' }}
              style={{ marginBottom: 0 }}
            >
              {item.snippet}
            </Paragraph>
          </Space>
        </List.Item>
      )}
    />
  );
};

const AnswerCard = ({ answer, className = '' }) => {
  // Handle loading state
  if (!answer) {
    return (
      <Card className={className}>
        <Spin tip="Checking facts..." />
      </Card>
    );
  }

  // Handle error state
  if (answer.error) {
    return (
      <Card className={`${className} ai-error-card`}>
        <Text type="danger">{answer.message || 'An error occurred'}</Text>
      </Card>
    );
  }

  // Extract fields safely
  const claim = answer.claim || 'No claim extracted';
  const verdict = answer.verdict || 'Unknown';
  const score = answer.score || 0;
  const explanation = answer.explanation;
  const citations = answer.citations || [];
  const features = answer.features || null;
  const explanationContent = typeof explanation === 'string' && explanation.trim().length > 0
    ? explanation
    : 'No explanation available';
  const llmResponse = typeof answer.llm_response === 'string' && answer.llm_response.trim().length > 0
    ? answer.llm_response
    : null;

  return (
    <Card
      className={`${className} ai-answer-card`}
      title={
        <Space>
          <VerdictIcon verdict={verdict} />
          <Text strong>{verdict}</Text>
          <VerdictBadge verdict={verdict} score={score} />
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Claim */}
        <div>
          <Text strong>Claim:</Text>
          <Paragraph style={{ marginTop: '8px', marginBottom: 0 }}>
            "{claim}"
          </Paragraph>
        </div>

        <Divider style={{ margin: '8px 0' }} />

        {/* Explanation */}
        <div>
          <Text strong>Explanation:</Text>
          <div className="markdown-body" style={{ marginTop: '8px' }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {explanationContent}
            </ReactMarkdown>
          </div>
        </div>

        {/* Model Response + Evidence panels (collapsible) */}
        {(llmResponse || features) && (
          <Collapse ghost style={{ marginBottom: '3rem' }}>
            {llmResponse && (
              <Panel header="Model Response" key="model-response">
                <div className="markdown-body" style={{ marginBottom: 0 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {llmResponse}
                  </ReactMarkdown>
                </div>
              </Panel>
            )}
            {features && (
              <Panel header="View Evidence Analysis" key="evidence-analysis">
                <FeatureDisplay features={features} />
              </Panel>
            )}
          </Collapse>
        )}

        {/* Citations */}
        {citations.length > 0 && (
          <div>
            <Text strong>Sources:</Text>
            <div style={{ marginTop: '8px' }}>
              <CitationList citations={citations} />
            </div>
          </div>
        )}
      </Space>
    </Card>
  );
};

// Wrapper component to render fact-check results inside the AI message container
const FactCheckResultRenderer = ({ answer }) => {
  return (
    <div className="ai-result-wrapper">
      <AnswerCard answer={answer} />
    </div>
  );
};

const RenderQA = ({ conversation, isLoading }) => {
  // Handle empty state
  if (!conversation || conversation.length === 0) {
    if (!isLoading) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#8c8c8c' }}>
          <QuestionCircleOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
          <Title level={4} type="secondary">Enter a claim to fact-check</Title>
          <Text type="secondary">
            The system will analyze the claim against the knowledge base and provide a verdict with supporting evidence.
          </Text>
        </div>
      );
    }
  }

  return (
    <div className="chat-thread">
      {conversation.map((entry, index) => (
        <div key={index} className="message-pair">
          <div className="message-row user-row">
            <div className="bubble-label">You</div>
            <div className="message-bubble user-bubble">
              <Paragraph style={{ margin: 0 }}>{entry.question}</Paragraph>
            </div>
          </div>

          <div className="message-row ai-row">
            <div className="bubble-label ai-label">
              <div className="ai-avatar">AI</div>
              <Text strong>NBA Insight AI</Text>
            </div>
            <FactCheckResultRenderer answer={entry.answer} />
          </div>
        </div>
      ))}
    </div>
  );
};

export default RenderQA;
