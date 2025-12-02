import EventRegistrationPage from "./EventRegistrationPage";
import LoginPage from "./LoginPage";
import LogoutPage from "./LogoutPage";

const nbaPageRoutes = [
  {
    path: "/event-registration",
    label: "Event Registration",
    description: "Reserve seats for live NBA Insight sessions.",
    component: EventRegistrationPage,
  },
  {
    path: "/login",
    label: "Login",
    description: "Authenticate league and media partners.",
    component: LoginPage,
  },
  {
    path: "/logout",
    label: "Logout",
    description: "Confirm secure sign-out state.",
    component: LogoutPage,
  },
];

export default nbaPageRoutes;
export { EventRegistrationPage, LoginPage, LogoutPage };
