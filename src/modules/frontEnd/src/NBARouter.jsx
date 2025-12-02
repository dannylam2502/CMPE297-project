import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import RegistrationPage from "./pages/RegistrationPage";
import nbaPageRoutes from "./pages/nbaPageRoutes";

const NBARouter = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/registration" element={<RegistrationPage />} />
      {nbaPageRoutes.map(({ path, component: Component }) => (
        <Route key={path} path={path} element={<Component />} />
      ))}
      <Route path="*" element={<App />} />
    </Routes>
  </BrowserRouter>
);

export default NBARouter;
