import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";

import Dashboard from "@/pages/Dashboard";
import RunsList from "@/pages/RunsList";
import RunDetail from "@/pages/RunDetail";
import NewRun from "@/pages/NewRun";
import DataManagement from "@/pages/DataManagement";
import Auth from "@/pages/Auth";
import Profile from "@/pages/Profile";
import NotFound from "@/pages/not-found";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/runs" component={RunsList} />
      <Route path="/runs/:id" component={RunDetail} />
      <Route path="/new-run" component={NewRun} />
      <Route path="/datasets" component={DataManagement} />
      <Route path="/auth" component={Auth} />
      <Route path="/profile" component={Profile} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router />
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
