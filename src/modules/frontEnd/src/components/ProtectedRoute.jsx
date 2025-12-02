import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const STORAGE_KEY = "nbaInsightUser";

const ProtectedRoute = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    try {
      return Boolean(localStorage.getItem(STORAGE_KEY));
    } catch (error) {
      console.warn("Unable to read auth state:", error);
      return false;
    }
  });

  useEffect(() => {
    const evaluateAuth = () => {
      try {
        setIsAuthenticated(Boolean(localStorage.getItem(STORAGE_KEY)));
      } catch (error) {
        console.warn("Unable to read auth state:", error);
        setIsAuthenticated(false);
      }
    };

    evaluateAuth();
    window.addEventListener("storage", evaluateAuth);
    window.addEventListener("auth-change", evaluateAuth);
    return () => {
      window.removeEventListener("storage", evaluateAuth);
      window.removeEventListener("auth-change", evaluateAuth);
    };
  }, []);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login", { replace: true, state: { from: location.pathname } });
    }
  }, [isAuthenticated, location.pathname, navigate]);

  if (!isAuthenticated) {
    return null;
  }

  return children;
};

export default ProtectedRoute;
