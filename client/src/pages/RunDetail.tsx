import { useRoute } from "wouter";
import { useRun, useRunLogs } from "@/hooks/use-runs";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Terminal as TerminalIcon, 
  FileJson, 
  Download, 
  RefreshCw 
} from "lucide-react";
import { format } from "date-fns";

export default function RunDetail() {
  const [match, params] = useRoute("/runs/:id");
  const id = parseInt(params?.id || "0");
  
  const { data: run, isLoading: runLoading } = useRun(id);
  const { data: logsData, isLoading: logsLoading } = useRunLogs(id);

  if (runLoading) return <RunSkeleton />;
  if (!run) return <div>Run not found</div>;

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title={`Run #${id}`} />
        
        <main className="flex-1 p-8 space-y-6 overflow-hidden flex flex-col h-full">
          {/* Top Info Bar */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card/40 p-6 rounded-xl border border-border/50 backdrop-blur-sm">
            <div>
              <h2 className="text-2xl font-bold mb-1">{run.runName}</h2>
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <span className="flex items-center gap-1 font-mono">
                  <FileJson size={14} />
                  {run.type}
                </span>
                <span className="text-border">|</span>
                <span>{run.createdAt && format(new Date(run.createdAt), "PPP p")}</span>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <StatusBadge status={run.status} className="text-sm px-4 py-1" />
              {run.artifactPath && (
                <Button variant="outline" size="sm" className="gap-2">
                  <Download size={14} /> Artifacts
                </Button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
            {/* Configuration Column */}
            <Card className="glass-panel flex flex-col h-full">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <FileJson size={18} className="text-primary" />
                  Configuration
                </CardTitle>
              </CardHeader>
              <Separator className="bg-border/50" />
              <CardContent className="flex-1 min-h-0 p-0">
                <ScrollArea className="h-full p-4">
                  <pre className="text-xs font-mono text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {JSON.stringify(run.config, null, 2)}
                  </pre>
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Logs Column */}
            <Card className="glass-panel lg:col-span-2 flex flex-col h-full border-primary/20 shadow-lg shadow-black/20">
              <CardHeader className="pb-3 bg-black/20 flex flex-row items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <TerminalIcon size={18} className="text-primary" />
                  System Logs
                </CardTitle>
                {run.status === 'running' && (
                  <div className="flex items-center gap-2 text-xs text-primary animate-pulse">
                    <RefreshCw size={12} className="animate-spin" />
                    Live Streaming
                  </div>
                )}
              </CardHeader>
              <Separator className="bg-border/50" />
              <CardContent className="flex-1 min-h-0 p-0 bg-[#0c0c0e]">
                <ScrollArea className="h-full p-4 font-mono text-sm">
                  {logsLoading ? (
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-3/4 bg-white/5" />
                      <Skeleton className="h-4 w-1/2 bg-white/5" />
                      <Skeleton className="h-4 w-2/3 bg-white/5" />
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap text-gray-300 leading-relaxed">
                      {logsData?.logs || run.logs || "No logs available."}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}

function RunSkeleton() {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex-1 p-8">
        <Skeleton className="h-16 w-full mb-8" />
        <Skeleton className="h-32 w-full mb-8" />
        <div className="grid grid-cols-3 gap-6 h-[500px]">
          <Skeleton className="h-full" />
          <Skeleton className="col-span-2 h-full" />
        </div>
      </div>
    </div>
  );
}
