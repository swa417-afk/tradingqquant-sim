import { useRuns } from "@/hooks/use-runs";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";
import { Activity, TrendingUp, Wallet, Clock, ArrowRight } from "lucide-react";
import { Link } from "wouter";
import { format } from "date-fns";

// Dummy data for visualization since we don't have real time-series in schema
const chartData = [
  { name: 'Mon', pnl: 4000 },
  { name: 'Tue', pnl: 3000 },
  { name: 'Wed', pnl: -2000 },
  { name: 'Thu', pnl: 2780 },
  { name: 'Fri', pnl: 1890 },
  { name: 'Sat', pnl: 2390 },
  { name: 'Sun', pnl: 3490 },
];

export default function Dashboard() {
  const { data: runs, isLoading } = useRuns();

  const totalRuns = runs?.length || 0;
  const completedRuns = runs?.filter(r => r.status === 'completed').length || 0;
  const failedRuns = runs?.filter(r => r.status === 'failed').length || 0;
  const successRate = totalRuns > 0 ? Math.round((completedRuns / totalRuns) * 100) : 0;

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="Dashboard" />
        
        <main className="flex-1 p-8 space-y-8 overflow-y-auto">
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="glass-panel border-l-4 border-l-primary">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Total Runs</p>
                    <h3 className="text-2xl font-bold mt-2">{isLoading ? <Skeleton className="h-8 w-16" /> : totalRuns}</h3>
                  </div>
                  <div className="p-2 bg-primary/10 rounded-lg text-primary">
                    <Activity className="h-5 w-5" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel border-l-4 border-l-green-500">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                    <h3 className="text-2xl font-bold mt-2 text-green-500">{isLoading ? <Skeleton className="h-8 w-16" /> : `${successRate}%`}</h3>
                  </div>
                  <div className="p-2 bg-green-500/10 rounded-lg text-green-500">
                    <TrendingUp className="h-5 w-5" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel border-l-4 border-l-purple-500">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Total PnL</p>
                    <h3 className="text-2xl font-bold mt-2 text-purple-500">+$24,592</h3>
                  </div>
                  <div className="p-2 bg-purple-500/10 rounded-lg text-purple-500">
                    <Wallet className="h-5 w-5" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel border-l-4 border-l-orange-500">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Avg Duration</p>
                    <h3 className="text-2xl font-bold mt-2">12m 30s</h3>
                  </div>
                  <div className="p-2 bg-orange-500/10 rounded-lg text-orange-500">
                    <Clock className="h-5 w-5" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Chart Area */}
            <Card className="glass-panel lg:col-span-2">
              <CardHeader>
                <CardTitle>Performance Overview</CardTitle>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis 
                      dataKey="name" 
                      stroke="#888888" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false} 
                    />
                    <YAxis 
                      stroke="#888888" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false} 
                      tickFormatter={(value) => `$${value}`} 
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#09090b', border: '1px solid #27272a' }}
                      itemStyle={{ color: '#fff' }}
                      cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                    />
                    <Bar 
                      dataKey="pnl" 
                      fill="currentColor" 
                      radius={[4, 4, 0, 0]} 
                      className="fill-primary"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Recent Runs List */}
            <Card className="glass-panel">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recent Runs</CardTitle>
                <Link href="/runs" className="text-sm text-primary hover:underline">View All</Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {isLoading ? (
                    Array(5).fill(0).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)
                  ) : runs?.slice(0, 5).map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-3 rounded-lg border border-border/50 bg-card/30 hover:bg-accent/20 transition-colors">
                      <div className="flex flex-col gap-1">
                        <span className="font-medium text-sm text-foreground">{run.runName}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground font-mono">#{run.id}</span>
                          <span className="text-xs text-muted-foreground">• {run.type}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <StatusBadge status={run.status} />
                        <Link href={`/runs/${run.id}`}>
                          <Button size="icon" variant="ghost" className="h-8 w-8 text-muted-foreground hover:text-primary">
                            <ArrowRight size={16} />
                          </Button>
                        </Link>
                      </div>
                    </div>
                  ))}
                  {!isLoading && (!runs || runs.length === 0) && (
                    <div className="text-center text-muted-foreground py-8">
                      No runs yet
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
