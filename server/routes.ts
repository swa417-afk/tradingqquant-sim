import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import multer from "multer";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import express from "express";
import { authRouter, getCurrentUser } from "./auth";

const upload = multer({ dest: "quant-sim/data/raw/" });

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  app.use('/api/auth', authRouter);
  
  app.use('/artifacts', express.static(path.join(process.cwd(), 'quant-sim/runs')));

  // === RUNS ===

  app.get(api.runs.list.path, async (req, res) => {
    const runs = await storage.getRuns();
    res.json(runs);
  });

  app.get(api.runs.get.path, async (req, res) => {
    const run = await storage.getRun(Number(req.params.id));
    if (!run) return res.status(404).json({ message: "Run not found" });
    res.json(run);
  });

  app.get(api.runs.logs.path, async (req, res) => {
      const run = await storage.getRun(Number(req.params.id));
      if (!run) return res.status(404).json({ message: "Run not found" });
      
      // If run is running, read from file. If completed, it might be in DB or file.
      // For now, always try to read from file if artifactPath exists or try to find it.
      // The artifactPath in DB is the run directory. Logs are at {run_dir}/logs.txt
      
      let logs = "";
      if (run.artifactPath) {
          const logPath = path.join(run.artifactPath, "logs.txt");
          if (fs.existsSync(logPath)) {
              logs = fs.readFileSync(logPath, "utf-8");
          }
      }
      res.json({ logs });
  });

  app.post(api.runs.create.path, async (req, res) => {
    try {
      const input = api.runs.create.input.parse(req.body);
      
      // 1. Create run in DB
      const run = await storage.createRun({
          ...input,
          status: "pending",
          config: input.config,
      });

      // 2. Spawn python process
      // We need to write the config to a temp file or use the one selected
      const configName = (input.config as any).name; // Assuming frontend sends { name: "backtest.yaml" }
      // Or if raw content provided, write to temp.
      // For MVP, let's assume input.config has { filename: string } and we use that from configs/
      // Actually, let's just accept the config filename in the input.
      
      const configFilename = (input.config as any).filename || "backtest.yaml";
      const configPath = path.join(process.cwd(), "quant-sim/configs", configFilename);

      if (!fs.existsSync(configPath)) {
          return res.status(400).json({ message: `Config file ${configFilename} not found` });
      }

      const cmd = input.type === 'backtest' ? 'backtest' : 'paper';
      
      // Run Python
      // quant-sim backtest --config configs/backtest.yaml
      const pythonProcess = spawn("python3", ["-m", "quant_sim.cli", cmd, "--config", configPath], {
          cwd: path.join(process.cwd(), "quant-sim"),
          env: { ...process.env, PYTHONPATH: path.join(process.cwd(), "quant-sim/src") }
      });

      console.log(`Started run ${run.id} with PID ${pythonProcess.pid}`);

      // Update status to running
      await storage.updateRunStatus(run.id, "running");

      let runDir = "";
      
      pythonProcess.stdout.on("data", async (data) => {
          const output = data.toString();
          console.log(`[Run ${run.id}] ${output}`);
          // Try to catch the artifacts dir from logs
          // "Artifacts: runs/..."
          const match = output.match(/Artifacts: (.*)/);
          if (match && match[1]) {
              runDir = match[1].trim();
              // Update DB with artifact path
              // Relative to quant-sim root, we want full path or relative to project root
              // The python script outputs relative path inside quant-sim cwd (e.g. "runs/...")
              const fullPath = path.join(process.cwd(), "quant-sim", runDir);
              await storage.updateRunStatus(run.id, "running", undefined, fullPath);
          }
      });

      pythonProcess.stderr.on("data", (data) => {
          console.error(`[Run ${run.id} ERR] ${data}`);
      });

      pythonProcess.on("close", async (code) => {
          console.log(`Run ${run.id} exited with code ${code}`);
          const status = code === 0 ? "completed" : "failed";
          await storage.updateRunStatus(run.id, status);
      });

      res.status(201).json(run);
    } catch (err) {
       console.error(err);
       res.status(500).json({ message: "Failed to start run" });
    }
  });

  // === DATASETS ===

  app.get(api.datasets.list.path, async (req, res) => {
    const datasets = await storage.getDatasets();
    res.json(datasets);
  });

  app.post(api.datasets.upload.path, upload.single("file"), async (req, res) => {
    if (!req.file) return res.status(400).json({ message: "No file uploaded" });
    
    // Move to processed dir (or keep in raw and let python handle it)
    // The python script expects CSVs in `data/processed/` or similar.
    // Let's put them in `quant-sim/data/processed/`
    const targetDir = path.join(process.cwd(), "quant-sim/data/processed");
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
    }
    
    const targetPath = path.join(targetDir, req.file.originalname);
    fs.renameSync(req.file.path, targetPath);

    const dataset = await storage.createDataset({
        filename: req.file.originalname,
        originalName: req.file.originalname,
        filePath: targetPath,
        size: String(req.file.size),
    });

    res.status(201).json(dataset);
  });

  app.delete(api.datasets.delete.path, async (req, res) => {
      const id = Number(req.params.id);
      const dataset = await storage.getDataset(id);
      if (dataset) {
          if (fs.existsSync(dataset.filePath)) {
              fs.unlinkSync(dataset.filePath);
          }
          await storage.deleteDataset(id);
      }
      res.status(204).send();
  });

  // === CONFIGS ===

  app.get(api.configs.list.path, (req, res) => {
      const configDir = path.join(process.cwd(), "quant-sim/configs");
      if (!fs.existsSync(configDir)) return res.json([]);
      const files = fs.readdirSync(configDir).filter(f => f.endsWith(".yaml") || f.endsWith(".yml"));
      res.json(files);
  });

  app.get(api.configs.get.path, (req, res) => {
      const name = req.params.name;
      const filePath = path.join(process.cwd(), "quant-sim/configs", name);
      if (!fs.existsSync(filePath)) return res.status(404).json({ message: "Config not found" });
      const content = fs.readFileSync(filePath, "utf-8");
      res.json({ content });
  });

  app.post(api.configs.save.path, (req, res) => {
      const name = req.params.name;
      const content = req.body.content;
      const filePath = path.join(process.cwd(), "quant-sim/configs", name);
      fs.writeFileSync(filePath, content, "utf-8");
      res.json({ success: true });
  });

  return httpServer;
}
