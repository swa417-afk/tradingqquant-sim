import { Router, Request, Response, NextFunction } from "express";
import bcrypt from "bcrypt";
import crypto from "crypto";
import { db } from "./db";
import { users, sessions, profiles, signupSchema, profileUpdateSchema } from "@shared/schema";
import { eq, and, gt } from "drizzle-orm";

const SALT_ROUNDS = 10;
const SESSION_TTL_DAYS = 14;
const COOKIE_NAME = "qs_session";

export const authRouter = Router();

function nowUtc(): Date {
  return new Date();
}

function sessionExpiresAt(): Date {
  const d = new Date();
  d.setDate(d.getDate() + SESSION_TTL_DAYS);
  return d;
}

async function createSession(userId: number): Promise<string> {
  const token = crypto.randomBytes(48).toString("base64url");
  const expiresAt = sessionExpiresAt();
  await db.insert(sessions).values({ userId, token, expiresAt });
  return token;
}

function setAuthCookie(res: Response, token: string) {
  res.cookie(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: SESSION_TTL_DAYS * 24 * 60 * 60 * 1000,
    path: "/",
  });
}

function clearAuthCookie(res: Response) {
  res.clearCookie(COOKIE_NAME, { path: "/" });
}

authRouter.post("/signup", async (req: Request, res: Response) => {
  try {
    const parsed = signupSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.issues[0].message });
    }

    const { email, password } = parsed.data;
    const normalizedEmail = email.toLowerCase();

    const existing = await db.select().from(users).where(eq(users.email, normalizedEmail));
    if (existing.length > 0) {
      return res.status(409).json({ error: "Email already registered" });
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const [user] = await db.insert(users).values({ email: normalizedEmail, passwordHash }).returning();

    await db.insert(profiles).values({
      userId: user.id,
      displayName: email.split("@")[0],
    });

    const token = await createSession(user.id);
    setAuthCookie(res, token);

    return res.json({ ok: true });
  } catch (err) {
    console.error("Signup error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.post("/signin", async (req: Request, res: Response) => {
  try {
    const parsed = signupSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.issues[0].message });
    }

    const { email, password } = parsed.data;
    const normalizedEmail = email.toLowerCase();

    const [user] = await db.select().from(users).where(eq(users.email, normalizedEmail));
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const valid = await bcrypt.compare(password, user.passwordHash);
    if (!valid) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const token = await createSession(user.id);
    setAuthCookie(res, token);

    return res.json({ ok: true });
  } catch (err) {
    console.error("Signin error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.post("/signout", async (req: Request, res: Response) => {
  try {
    const token = req.cookies?.[COOKIE_NAME];
    if (token) {
      await db.delete(sessions).where(eq(sessions.token, token));
    }
    clearAuthCookie(res);
    return res.json({ ok: true });
  } catch (err) {
    console.error("Signout error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.get("/me", async (req: Request, res: Response) => {
  try {
    const token = req.cookies?.[COOKIE_NAME];
    if (!token) {
      return res.status(401).json({ error: "Not authenticated" });
    }

    const [session] = await db
      .select()
      .from(sessions)
      .where(and(eq(sessions.token, token), gt(sessions.expiresAt, nowUtc())));

    if (!session) {
      clearAuthCookie(res);
      return res.status(401).json({ error: "Session expired" });
    }

    const [user] = await db.select().from(users).where(eq(users.id, session.userId));
    if (!user) {
      return res.status(401).json({ error: "Invalid session" });
    }

    const [profile] = await db.select().from(profiles).where(eq(profiles.userId, user.id));

    return res.json({
      email: user.email,
      displayName: profile?.displayName || "",
      avatarUrl: profile?.avatarUrl || "",
      defaultExchange: profile?.defaultExchange || "coinbase",
      baseCurrency: profile?.baseCurrency || "USD",
      riskMode: profile?.riskMode || "balanced",
      maxLeverage: profile?.maxLeverage || "2.0",
      maxPositionPct: profile?.maxPositionPct || "0.25",
      notes: profile?.notes || "",
    });
  } catch (err) {
    console.error("Me error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

authRouter.patch("/profile", async (req: Request, res: Response) => {
  try {
    const token = req.cookies?.[COOKIE_NAME];
    if (!token) {
      return res.status(401).json({ error: "Not authenticated" });
    }

    const [session] = await db
      .select()
      .from(sessions)
      .where(and(eq(sessions.token, token), gt(sessions.expiresAt, nowUtc())));

    if (!session) {
      return res.status(401).json({ error: "Session expired" });
    }

    const parsed = profileUpdateSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.issues[0].message });
    }

    const [existingProfile] = await db.select().from(profiles).where(eq(profiles.userId, session.userId));

    if (!existingProfile) {
      await db.insert(profiles).values({ userId: session.userId, ...parsed.data });
    } else {
      await db.update(profiles).set(parsed.data).where(eq(profiles.userId, session.userId));
    }

    const [user] = await db.select().from(users).where(eq(users.id, session.userId));
    const [profile] = await db.select().from(profiles).where(eq(profiles.userId, session.userId));

    return res.json({
      email: user.email,
      displayName: profile?.displayName || "",
      avatarUrl: profile?.avatarUrl || "",
      defaultExchange: profile?.defaultExchange || "coinbase",
      baseCurrency: profile?.baseCurrency || "USD",
      riskMode: profile?.riskMode || "balanced",
      maxLeverage: profile?.maxLeverage || "2.0",
      maxPositionPct: profile?.maxPositionPct || "0.25",
      notes: profile?.notes || "",
    });
  } catch (err) {
    console.error("Profile update error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export async function getCurrentUser(req: Request): Promise<{ id: number; email: string } | null> {
  const token = req.cookies?.[COOKIE_NAME];
  if (!token) return null;

  const [session] = await db
    .select()
    .from(sessions)
    .where(and(eq(sessions.token, token), gt(sessions.expiresAt, nowUtc())));

  if (!session) return null;

  const [user] = await db.select().from(users).where(eq(users.id, session.userId));
  if (!user) return null;

  return { id: user.id, email: user.email };
}

export function requireAuth(req: Request, res: Response, next: NextFunction) {
  getCurrentUser(req).then((user) => {
    if (!user) {
      return res.status(401).json({ error: "Not authenticated" });
    }
    (req as any).user = user;
    next();
  });
}
