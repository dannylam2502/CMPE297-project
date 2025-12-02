import React from "react";
import { Card, Typography, Form, Input, Button } from "antd";
import { ArrowRightOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import "./EventRegistrationPage.css";

const { Title, Paragraph, Text } = Typography;

const EventRegistrationPage = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const handleFinish = (values) => {
    console.log("Quick registration submitted:", values);
    navigate("/");
  };

  return (
    <div className="nba-lite-page">
      <div className="nba-lite-shell">
        <div className="nba-lite-copy">
          <span className="nba-lite-eyebrow">Courtside Access</span>
          <Title level={2} className="nba-lite-title">
            Join the NBA Insight waitlist
          </Title>
          <Paragraph className="nba-lite-subtitle">
            A single signup puts you on the distro for real-time scouting briefs
            and fact-check drops before every slate.
          </Paragraph>
          <Text className="nba-lite-note">Fields marked * are required.</Text>
        </div>

        <Card className="nba-lite-card" bordered={false}>
          <Form
            layout="vertical"
            form={form}
            onFinish={handleFinish}
            className="nba-lite-form"
          >
            <Form.Item
              label="Full Name"
              name="fullName"
              rules={[{ required: true, message: "Please enter your full name" }]}
            >
              <Input placeholder="Enter your full name" />
            </Form.Item>

            <Form.Item
              label="Email"
              name="email"
              rules={[
                { required: true, message: "Please enter your email" },
                { type: "email", message: "Enter a valid email address" },
              ]}
            >
              <Input placeholder="you@team.com" />
            </Form.Item>

            <Form.Item label="Organization" name="organization">
              <Input placeholder="Optional" />
            </Form.Item>

            <Form.Item label="Phone Number" name="phone">
              <Input placeholder="Optional" />
            </Form.Item>

            <Button
              type="primary"
              htmlType="submit"
              className="nba-lite-button"
              icon={<ArrowRightOutlined />}
            >
              Continue
            </Button>
          </Form>
        </Card>
      </div>
    </div>
  );
};

export default EventRegistrationPage;
