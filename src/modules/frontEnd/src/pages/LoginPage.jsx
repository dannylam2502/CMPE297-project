import React, { useState } from "react";
import { Card, Typography, Form, Input, Button, message } from "antd";
import { useNavigate } from "react-router-dom";
import "./LoginPage.css";

const { Title, Paragraph } = Typography;

const LoginPage = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [submitting, setSubmitting] = useState(false);

  const handleFinish = async (values) => {
    setSubmitting(true);
    try {
      const response = await fetch("http://localhost:5005/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          email: values.email,
          password: values.password,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.success === false) {
        throw new Error(data.message || "Login failed. Please try again.");
      }
      try {
        localStorage.setItem(
          "nbaInsightUser",
          JSON.stringify({ email: values.email })
        );
      } catch (error) {
        console.warn("Unable to persist login:", error);
      }
      message.success(data.message || "Login successful");
      navigate("/");
    } catch (error) {
      message.error(error.message || "Login failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
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
              loading={submitting}
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
