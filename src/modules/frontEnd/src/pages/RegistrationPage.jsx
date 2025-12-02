import React, { useState } from "react";
import { Card, Typography, Form, Input, Button, message } from "antd";
import { ArrowRightOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import "./RegistrationPage.css";

const { Title, Paragraph, Text } = Typography;

const RegistrationPage = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [submitting, setSubmitting] = useState(false);

  const handleFinish = async (values) => {
    if (values.password !== values.confirmPassword) {
      message.error("Passwords do not match. Please confirm your password.");
      return;
    }

    const payload = {
      full_name: (values.fullName || "").trim(),
      email: (values.email || "").trim(),
      password: values.password,
    };

    setSubmitting(true);
    try {
      const response = await fetch("http://localhost:5005/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.success === false) {
        throw new Error(data.message || "Registration failed. Please try again.");
      }

      message.success(data.message || "User registered successfully");
      form.resetFields();
      navigate("/login");
    } catch (error) {
      message.error(error.message || "Registration failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="nba-lite-page">
      <div className="nba-lite-shell">
        <div className="nba-lite-copy">
          <span className="nba-lite-eyebrow">Account Access</span>
          <Title level={2} className="nba-lite-title">
            Create your NBA Insight account
          </Title>
          <Paragraph className="nba-lite-subtitle">
            Unlock the control room to fact-check claims, explore scouting briefs,
            and track real-time intel.
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

            <Form.Item
              label="Password"
              name="password"
              rules={[{ required: true, message: "Please enter a password" }]}
            >
              <Input.Password
                placeholder="Create a password"
                visibilityToggle={false}
                autoComplete="new-password"
              />
            </Form.Item>

            <Form.Item
              label="Confirm Password"
              name="confirmPassword"
              dependencies={["password"]}
              rules={[
                { required: true, message: "Please confirm your password" },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue("password") === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(
                      new Error("Passwords do not match. Please confirm again.")
                    );
                  },
                }),
              ]}
            >
              <Input.Password
                placeholder="Re-enter password"
                visibilityToggle={false}
                autoComplete="new-password"
              />
            </Form.Item>

            <Button
              type="primary"
              htmlType="submit"
              className="nba-lite-button"
              loading={submitting}
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

export default RegistrationPage;
