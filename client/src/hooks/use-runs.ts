import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, buildUrl, type CreateRunRequest } from "@shared/routes";

// GET /api/runs
export function useRuns() {
  return useQuery({
    queryKey: [api.runs.list.path],
    queryFn: async () => {
      const res = await fetch(api.runs.list.path, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch runs");
      return api.runs.list.responses[200].parse(await res.json());
    },
    refetchInterval: 5000, // Poll for updates
  });
}

// GET /api/runs/:id
export function useRun(id: number) {
  return useQuery({
    queryKey: [api.runs.get.path, id],
    queryFn: async () => {
      const url = buildUrl(api.runs.get.path, { id });
      const res = await fetch(url, { credentials: "include" });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("Failed to fetch run");
      return api.runs.get.responses[200].parse(await res.json());
    },
    refetchInterval: (query) => {
      const data = query.state.data;
      // Poll faster if running/pending
      return data && (data.status === 'running' || data.status === 'pending') ? 2000 : false;
    }
  });
}

// GET /api/runs/:id/logs
export function useRunLogs(id: number) {
  return useQuery({
    queryKey: [api.runs.logs.path, id],
    queryFn: async () => {
      const url = buildUrl(api.runs.logs.path, { id });
      const res = await fetch(url, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch logs");
      return api.runs.logs.responses[200].parse(await res.json());
    },
    refetchInterval: 3000, // Poll logs while viewing
  });
}

// POST /api/runs
export function useCreateRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: CreateRunRequest) => {
      const validated = api.runs.create.input.parse(data);
      const res = await fetch(api.runs.create.path, {
        method: api.runs.create.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validated),
        credentials: "include",
      });
      if (!res.ok) {
        if (res.status === 400) {
          const error = api.runs.create.responses[400].parse(await res.json());
          throw new Error(error.message);
        }
        throw new Error("Failed to create run");
      }
      return api.runs.create.responses[201].parse(await res.json());
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [api.runs.list.path] }),
  });
}
