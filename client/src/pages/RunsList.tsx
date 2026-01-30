import { useRuns } from "@/hooks/use-runs";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";
import { Eye, Search, Filter, Play } from "lucide-react";
import { useState } from "react";
import { format } from "date-fns";
import { Skeleton } from "@/components/ui/skeleton";

export default function RunsList() {
  const { data: runs, isLoading } = useRuns();
  const [search, setSearch] = useState("");

  const filteredRuns = runs?.filter(run => 
    run.runName.toLowerCase().includes(search.toLowerCase()) ||
    run.type.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="All Runs" />
        
        <main className="flex-1 p-8 space-y-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 flex-1 max-w-md">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input 
                  placeholder="Search runs..." 
                  className="pl-9 bg-card border-border"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
              <Button variant="outline" size="icon" className="shrink-0">
                <Filter className="h-4 w-4" />
              </Button>
            </div>
            
            <Link href="/new-run">
              <Button className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20">
                <Play className="h-4 w-4 mr-2" />
                New Run
              </Button>
            </Link>
          </div>

          <Card className="glass-panel overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-muted/50 border-b border-border text-muted-foreground uppercase text-xs font-semibold">
                  <tr>
                    <th className="px-6 py-4">ID</th>
                    <th className="px-6 py-4">Name</th>
                    <th className="px-6 py-4">Type</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4">Created At</th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {isLoading ? (
                    Array(5).fill(0).map((_, i) => (
                      <tr key={i}>
                        <td className="px-6 py-4"><Skeleton className="h-4 w-8" /></td>
                        <td className="px-6 py-4"><Skeleton className="h-4 w-32" /></td>
                        <td className="px-6 py-4"><Skeleton className="h-4 w-16" /></td>
                        <td className="px-6 py-4"><Skeleton className="h-4 w-20" /></td>
                        <td className="px-6 py-4"><Skeleton className="h-4 w-24" /></td>
                        <td className="px-6 py-4"><Skeleton className="h-8 w-8 ml-auto" /></td>
                      </tr>
                    ))
                  ) : filteredRuns?.map((run) => (
                    <tr key={run.id} className="hover:bg-muted/30 transition-colors">
                      <td className="px-6 py-4 font-mono text-muted-foreground">#{run.id}</td>
                      <td className="px-6 py-4 font-medium">{run.runName}</td>
                      <td className="px-6 py-4">
                        <span className="px-2 py-1 rounded bg-secondary text-secondary-foreground text-xs font-medium">
                          {run.type}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <StatusBadge status={run.status} />
                      </td>
                      <td className="px-6 py-4 text-muted-foreground font-mono text-xs">
                        {run.createdAt ? format(new Date(run.createdAt), "MMM d, HH:mm") : '-'}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <Link href={`/runs/${run.id}`}>
                          <Button size="sm" variant="ghost" className="hover:bg-primary/10 hover:text-primary">
                            <Eye className="h-4 w-4 mr-1" /> View
                          </Button>
                        </Link>
                      </td>
                    </tr>
                  ))}
                  
                  {!isLoading && filteredRuns?.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-6 py-12 text-center text-muted-foreground">
                        No runs found matching your search.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </main>
      </div>
    </div>
  );
}
