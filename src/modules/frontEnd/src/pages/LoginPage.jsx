import React from "react";
import { Card, Typography, Form, Input, Button } from "antd";
import { useNavigate } from "react-router-dom";
import "./LoginPage.css";

const { Title, Paragraph } = Typography;

const LoginPage = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const handleFinish = (values) => {
    console.log("Signin payload:", values);
    navigate("/");
  };

  return (
    <div className="nba-auth-page">
      <div className="nba-auth-shell">
        <div className="nba-auth-copy">
          <span className="nba-auth-eyebrow">Control Room</span>
          <Title level={2} className="nba-auth-title">
            Welcome back to NBA Insight
          </Title>
          <Paragraph className="nba-auth-subtitle">
            Sign in to continue reviewing claims, scouting intel, and RAM
            breakdowns.
          </Paragraph>
        </div>

        <Card className="nba-auth-card" bordered={false}>
          <Form
            layout="vertical"
            form={form}
            onFinish={handleFinish}
            className="nba-auth-form"
          >
            <Form.Item
              label="Email"
              name="email"
              rules={[
                { required: true, message: "Please enter your email" },
                { type: "email", message: "Enter a valid email address" },
              ]}
            >
              <Input placeholder="you@team.com" autoComplete="email" />
            </Form.Item>

            <Form.Item
              label="Password"
              name="password"
              rules={[{ required: true, message: "Please enter your password" }]}
            >
              <Input.Password
                placeholder="Enter password"
                autoComplete="current-password"
              />
            </Form.Item>

            <Button
              type="primary"
              htmlType="submit"
              className="nba-auth-button"
            >
              Sign In
            </Button>
          </Form>
        </Card>
      </div>
    </div>
  );
};

export default LoginPage;
