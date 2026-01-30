import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, buildUrl } from "@shared/routes";

// GET /api/configs
export function useConfigs() {
  return useQuery({
    queryKey: [api.configs.list.path],
    queryFn: async () => {
      const res = await fetch(api.configs.list.path, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch configs");
      return api.configs.list.responses[200].parse(await res.json());
    },
  });
}

// GET /api/configs/:name
export function useConfig(name: string | null) {
  return useQuery({
    queryKey: [api.configs.get.path, name],
    queryFn: async () => {
      if (!name) return null;
      const url = buildUrl(api.configs.get.path, { name });
      const res = await fetch(url, { credentials: "include" });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("Failed to fetch config content");
      return api.configs.get.responses[200].parse(await res.json());
    },
    enabled: !!name,
  });
}

// POST /api/configs/:name
export function useSaveConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ name, content }: { name: string; content: string }) => {
      const url = buildUrl(api.configs.save.path, { name });
      const res = await fetch(url, {
        method: api.configs.save.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to save config");
      return api.configs.save.responses[200].parse(await res.json());
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [api.configs.list.path] }),
  });
}
