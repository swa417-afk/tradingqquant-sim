import { useDatasets, useUploadDataset, useDeleteDataset } from "@/hooks/use-datasets";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { UploadCloud, Trash2, FileText, HardDrive } from "lucide-react";
import { format } from "date-fns";
import { useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";

export default function DataManagement() {
  const { data: datasets, isLoading } = useDatasets();
  const uploadDataset = useUploadDataset();
  const deleteDataset = useDeleteDataset();
  const { toast } = useToast();
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("originalName", file.name);
    
    // Approximate size in MB string for UI
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    formData.append("size", sizeMB);

    try {
      await uploadDataset.mutateAsync(formData);
      toast({
        title: "Success",
        description: `Uploaded ${file.name} successfully.`,
      });
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this dataset?")) return;
    try {
      await deleteDataset.mutateAsync(id);
      toast({
        title: "Deleted",
        description: "Dataset removed successfully.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete dataset",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="Datasets" />
        
        <main className="flex-1 p-8 space-y-8">
          
          {/* Upload Area */}
          <div 
            className={`
              relative border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200
              ${dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-card/50"}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Input 
              type="file" 
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" 
              onChange={handleChange}
              accept=".csv,.json,.parquet"
            />
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 rounded-full bg-primary/10 text-primary">
                <UploadCloud size={32} />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Drop files here or click to upload</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Supports CSV, JSON, Parquet (Max 50MB)
                </p>
              </div>
            </div>
          </div>

          {/* Dataset List */}
          <Card className="glass-panel">
            <CardHeader>
              <CardTitle>Available Datasets</CardTitle>
              <CardDescription>Manage your market data files and external datasets.</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="border-border/50 hover:bg-transparent">
                    <TableHead>Filename</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Uploaded At</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                     Array(3).fill(0).map((_, i) => (
                      <TableRow key={i} className="border-border/50">
                        <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-16" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 ml-auto" /></TableCell>
                      </TableRow>
                    ))
                  ) : datasets?.map((dataset) => (
                    <TableRow key={dataset.id} className="border-border/50 hover:bg-muted/30">
                      <TableCell className="font-medium flex items-center gap-2">
                        <FileText size={16} className="text-primary" />
                        {dataset.originalName}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{dataset.size}</TableCell>
                      <TableCell className="text-muted-foreground text-xs">
                        {dataset.uploadedAt ? format(new Date(dataset.uploadedAt), "MMM d, yyyy") : '-'}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="hover:text-destructive hover:bg-destructive/10"
                          onClick={() => handleDelete(dataset.id)}
                          disabled={deleteDataset.isPending}
                        >
                          <Trash2 size={16} />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {!isLoading && datasets?.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4} className="h-24 text-center text-muted-foreground">
                        No datasets uploaded yet.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}
