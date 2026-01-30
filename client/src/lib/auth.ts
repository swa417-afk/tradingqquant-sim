import { apiRequest } from "./queryClient";

export interface UserProfile {
  email: string;
  displayName: string;
  avatarUrl: string;
  defaultExchange: string;
  baseCurrency: string;
  riskMode: string;
  maxLeverage: string;
  maxPositionPct: string;
  notes: string;
}

export async function signup(email: string, password: string): Promise<void> {
  await apiRequest("POST", "/api/auth/signup", { email, password });
}

export async function signin(email: string, password: string): Promise<void> {
  await apiRequest("POST", "/api/auth/signin", { email, password });
}

export async function signout(): Promise<void> {
  await apiRequest("POST", "/api/auth/signout");
}

export async function getMe(): Promise<UserProfile> {
  const res = await fetch("/api/auth/me", { credentials: "include" });
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

export async function updateProfile(data: Partial<UserProfile>): Promise<UserProfile> {
  const res = await fetch("/api/auth/profile", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update profile");
  return res.json();
}
