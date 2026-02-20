import { cn } from "@/lib/utils";

type Status = "pending" | "running" | "completed" | "failed";

interface StatusBadgeProps {
  status: Status | string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const styles = {
    pending: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
    running: "bg-blue-500/10 text-blue-500 border-blue-500/20 animate-pulse",
    completed: "bg-green-500/10 text-green-500 border-green-500/20",
    failed: "bg-red-500/10 text-red-500 border-red-500/20",
  };

  const normalizedStatus = (status.toLowerCase() as Status);
  const activeStyle = styles[normalizedStatus] || "bg-gray-500/10 text-gray-500";

  return (
    <span className={cn(
      "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border",
      activeStyle,
      className
    )}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}
