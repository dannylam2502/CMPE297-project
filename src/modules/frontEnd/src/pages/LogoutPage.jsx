import React from "react";
import { Card, Typography, Button } from "antd";
import { useNavigate } from "react-router-dom";
import "./LogoutPage.css";

const { Title, Paragraph } = Typography;

const LogoutPage = () => {
  const navigate = useNavigate();

  const handleSignOut = () => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch (error) {
      console.warn("Unable to clear storage:", error);
    }
    navigate("/login");
  };

  return (
    <div className="nba-signout-page">
      <div className="nba-signout-shell">
        <Card className="nba-signout-card" bordered={false}>
          <Title level={3} className="nba-signout-title">
            Ready to leave the Control Room?
          </Title>
          <Paragraph className="nba-signout-copy">
            Signing out clears your local session. Return any time to keep
            evaluating claims with NBA Insight AI.
          </Paragraph>
          <Button
            type="primary"
            className="nba-signout-button"
            onClick={handleSignOut}
          >
            Sign Out
          </Button>
        </Card>
      </div>
    </div>
  );
};

export default LogoutPage;
