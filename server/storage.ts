import { db } from "./db";
import { runs, datasets, type InsertRun, type Run, type Dataset, type InsertDataset } from "@shared/schema";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  // Runs
  getRuns(): Promise<Run[]>;
  getRun(id: number): Promise<Run | undefined>;
  createRun(run: InsertRun): Promise<Run>;
  updateRunStatus(id: number, status: string, logs?: string, artifactPath?: string): Promise<Run>;
  
  // Datasets
  getDatasets(): Promise<Dataset[]>;
  getDataset(id: number): Promise<Dataset | undefined>;
  createDataset(dataset: InsertDataset): Promise<Dataset>;
  deleteDataset(id: number): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  // Runs
  async getRuns(): Promise<Run[]> {
    return await db.select().from(runs).orderBy(desc(runs.createdAt));
  }

  async getRun(id: number): Promise<Run | undefined> {
    const [run] = await db.select().from(runs).where(eq(runs.id, id));
    return run;
  }

  async createRun(run: InsertRun): Promise<Run> {
    const [newRun] = await db.insert(runs).values(run).returning();
    return newRun;
  }

  async updateRunStatus(id: number, status: string, logs?: string, artifactPath?: string): Promise<Run> {
    const updates: Partial<Run> = { status };
    if (logs) updates.logs = logs;
    if (artifactPath) updates.artifactPath = artifactPath;
    
    const [updated] = await db.update(runs)
      .set(updates)
      .where(eq(runs.id, id))
      .returning();
    return updated;
  }

  // Datasets
  async getDatasets(): Promise<Dataset[]> {
    return await db.select().from(datasets).orderBy(desc(datasets.uploadedAt));
  }
  
  async getDataset(id: number): Promise<Dataset | undefined> {
    const [dataset] = await db.select().from(datasets).where(eq(datasets.id, id));
    return dataset;
  }

  async createDataset(dataset: InsertDataset): Promise<Dataset> {
    const [newDataset] = await db.insert(datasets).values(dataset).returning();
    return newDataset;
  }

  async deleteDataset(id: number): Promise<void> {
    await db.delete(datasets).where(eq(datasets.id, id));
  }
}

export const storage = new DatabaseStorage();
