import { useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";
import { ArrowLeft, Save, LogOut, User } from "lucide-react";
import { Link } from "wouter";

export default function Profile() {
  const [, setLocation] = useLocation();
  const { user, isLoading, isAuthenticated, signout, updateProfile } = useAuth();
  const { toast } = useToast();

  const [displayName, setDisplayName] = useState("");
  const [defaultExchange, setDefaultExchange] = useState("coinbase");
  const [baseCurrency, setBaseCurrency] = useState("USD");
  const [riskMode, setRiskMode] = useState("balanced");
  const [maxLeverage, setMaxLeverage] = useState("2.0");
  const [maxPositionPct, setMaxPositionPct] = useState("0.25");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    if (user) {
      setDisplayName(user.displayName || "");
      setDefaultExchange(user.defaultExchange || "coinbase");
      setBaseCurrency(user.baseCurrency || "USD");
      setRiskMode(user.riskMode || "balanced");
      setMaxLeverage(user.maxLeverage || "2.0");
      setMaxPositionPct(user.maxPositionPct || "0.25");
      setNotes(user.notes || "");
    }
  }, [user]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p>Loading...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    setLocation("/auth");
    return null;
  }

  const handleSave = () => {
    updateProfile.mutate(
      { displayName, defaultExchange, baseCurrency, riskMode, maxLeverage, maxPositionPct, notes },
      {
        onSuccess: () => {
          toast({ title: "Profile saved", description: "Your preferences have been updated." });
        },
        onError: () => {
          toast({ title: "Error", description: "Failed to save profile.", variant: "destructive" });
        },
      }
    );
  };

  const handleSignout = () => {
    signout.mutate(undefined, {
      onSuccess: () => {
        setLocation("/auth");
      },
    });
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center gap-4 mb-6">
          <Link href="/">
            <Button variant="ghost" size="icon" data-testid="button-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-2xl font-bold">Profile Settings</h1>
        </div>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Account
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Email</Label>
              <p className="text-muted-foreground">{user?.email}</p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="displayName">Display Name</Label>
              <Input
                id="displayName"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                data-testid="input-display-name"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Trading Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Default Exchange</Label>
                <Select value={defaultExchange} onValueChange={setDefaultExchange}>
                  <SelectTrigger data-testid="select-exchange">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="coinbase">Coinbase</SelectItem>
                    <SelectItem value="kraken">Kraken</SelectItem>
                    <SelectItem value="binance">Binance</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Base Currency</Label>
                <Select value={baseCurrency} onValueChange={setBaseCurrency}>
                  <SelectTrigger data-testid="select-currency">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="USD">USD</SelectItem>
                    <SelectItem value="EUR">EUR</SelectItem>
                    <SelectItem value="USDT">USDT</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Risk Mode</Label>
              <Select value={riskMode} onValueChange={setRiskMode}>
                <SelectTrigger data-testid="select-risk-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="conservative">Conservative</SelectItem>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="aggressive">Aggressive</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="maxLeverage">Max Leverage</Label>
                <Input
                  id="maxLeverage"
                  value={maxLeverage}
                  onChange={(e) => setMaxLeverage(e.target.value)}
                  data-testid="input-max-leverage"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxPositionPct">Max Position %</Label>
                <Input
                  id="maxPositionPct"
                  value={maxPositionPct}
                  onChange={(e) => setMaxPositionPct(e.target.value)}
                  data-testid="input-max-position"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes">Notes</Label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Personal notes about your trading preferences..."
                data-testid="input-notes"
              />
            </div>
          </CardContent>
        </Card>

        <div className="flex gap-4">
          <Button onClick={handleSave} disabled={updateProfile.isPending} data-testid="button-save">
            <Save className="h-4 w-4 mr-2" />
            {updateProfile.isPending ? "Saving..." : "Save Changes"}
          </Button>
          <Button variant="outline" onClick={handleSignout} data-testid="button-signout">
            <LogOut className="h-4 w-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </div>
    </div>
  );
}
