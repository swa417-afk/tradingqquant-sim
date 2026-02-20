import { pgTable, text, serial, timestamp, jsonb, integer, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// === AUTH TABLE DEFINITIONS ===

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  email: varchar("email", { length: 320 }).notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const sessions = pgTable("sessions", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  token: varchar("token", { length: 128 }).notNull().unique(),
  expiresAt: timestamp("expires_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const profiles = pgTable("profiles", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id, { onDelete: "cascade" }).unique(),
  displayName: varchar("display_name", { length: 120 }).default(""),
  avatarUrl: text("avatar_url").default(""),
  defaultExchange: varchar("default_exchange", { length: 64 }).default("coinbase"),
  baseCurrency: varchar("base_currency", { length: 16 }).default("USD"),
  riskMode: varchar("risk_mode", { length: 32 }).default("balanced"),
  maxLeverage: varchar("max_leverage", { length: 16 }).default("2.0"),
  maxPositionPct: varchar("max_position_pct", { length: 16 }).default("0.25"),
  notes: text("notes").default(""),
});

// === TRADING TABLE DEFINITIONS ===

export const runs = pgTable("runs", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id, { onDelete: "set null" }),
  runName: text("run_name").notNull(),
  type: text("type").notNull(),
  status: text("status").notNull().default("pending"),
  config: jsonb("config").notNull(),
  logs: text("logs"),
  artifactPath: text("artifact_path"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const datasets = pgTable("datasets", {
  id: serial("id").primaryKey(),
  filename: text("filename").notNull(),
  originalName: text("original_name").notNull(),
  filePath: text("file_path").notNull(),
  size: text("size").notNull(),
  uploadedAt: timestamp("uploaded_at").defaultNow(),
});

// === SCHEMAS ===

export const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
});

export const profileUpdateSchema = z.object({
  displayName: z.string().optional(),
  avatarUrl: z.string().optional(),
  defaultExchange: z.string().optional(),
  baseCurrency: z.string().optional(),
  riskMode: z.string().optional(),
  maxLeverage: z.string().optional(),
  maxPositionPct: z.string().optional(),
  notes: z.string().optional(),
});

export const insertRunSchema = createInsertSchema(runs).omit({ 
  id: true, 
  createdAt: true, 
  logs: true, 
  artifactPath: true, 
  status: true,
  userId: true,
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({ 
  id: true, 
  uploadedAt: true 
});

// === TYPES ===

export type User = typeof users.$inferSelect;
export type Session = typeof sessions.$inferSelect;
export type Profile = typeof profiles.$inferSelect;

export type Run = typeof runs.$inferSelect;
export type InsertRun = z.infer<typeof insertRunSchema>;

export type Dataset = typeof datasets.$inferSelect;
export type InsertDataset = z.infer<typeof insertDatasetSchema>;

export type SignupRequest = z.infer<typeof signupSchema>;
export type ProfileUpdateRequest = z.infer<typeof profileUpdateSchema>;

// API Types
export type CreateRunRequest = InsertRun;
export type RunResponse = Run;

export type FileUploadResponse = Dataset;

export type ConfigFile = {
  name: string;
  content: string;
};
