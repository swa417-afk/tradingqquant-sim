import { Link, useLocation } from "wouter";
import { 
  LayoutDashboard, 
  PlayCircle, 
  Database, 
  Settings, 
  Terminal,
  Activity
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Active Runs', href: '/runs', icon: Activity },
  { name: 'New Run', href: '/new-run', icon: PlayCircle },
  { name: 'Datasets', href: '/datasets', icon: Database },
];

export function Sidebar() {
  const [location] = useLocation();

  return (
    <div className="flex flex-col w-64 border-r border-border bg-card/30 min-h-screen sticky top-0">
      <div className="p-6 flex items-center gap-3 border-b border-border/50">
        <div className="w-8 h-8 rounded bg-primary flex items-center justify-center text-primary-foreground">
          <Terminal size={18} />
        </div>
        <span className="font-bold text-lg tracking-tight">QuantTerm</span>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {navigation.map((item) => {
          const isActive = location === item.href;
          return (
            <Link 
              key={item.name} 
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200",
                isActive 
                  ? "bg-primary/10 text-primary border border-primary/20 shadow-sm" 
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
            >
              <item.icon size={18} />
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border/50">
        <div className="bg-accent/30 rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground mb-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            System Online
          </div>
          <div className="text-xs text-muted-foreground/60">
            v2.4.0-stable
          </div>
        </div>
      </div>
    </div>
  );
}
