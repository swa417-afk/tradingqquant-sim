import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useLocation } from "wouter";
import { useCreateRun } from "@/hooks/use-runs";
import { useConfigs, useConfig } from "@/hooks/use-configs";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/hooks/use-toast";
import { PlayCircle, Settings, Code, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";

// Form Schema
const formSchema = z.object({
  runName: z.string().min(3, "Name must be at least 3 characters"),
  type: z.enum(["backtest", "paper"]),
  configContent: z.string().min(1, "Config content is required"),
});

export default function NewRun() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const createRun = useCreateRun();
  const { data: configFiles } = useConfigs();
  
  const [selectedConfigFile, setSelectedConfigFile] = useState<string | null>(null);
  const { data: configContent } = useConfig(selectedConfigFile);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      runName: "",
      type: "backtest",
      configContent: "",
    },
  });

  // Update textarea when a config file is selected
  useEffect(() => {
    if (configContent) {
      form.setValue("configContent", configContent.content);
    }
  }, [configContent, form]);

  async function onSubmit(values: z.infer<typeof formSchema>) {
    try {
      // Parse YAML/JSON content to object for the API
      let configObj;
      try {
        configObj = JSON.parse(values.configContent);
      } catch (e) {
        // Fallback: send as string wrapped in object if not JSON
        // In a real app, use a YAML parser here if YAML is expected
        configObj = { raw: values.configContent };
      }

      const run = await createRun.mutateAsync({
        runName: values.runName,
        type: values.type,
        config: configObj,
      });

      toast({
        title: "Run Started",
        description: `Run #${run.id} has been initialized successfully.`,
      });

      setLocation(`/runs/${run.id}`);
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to start run",
        variant: "destructive",
      });
    }
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="New Run" />
        
        <main className="flex-1 p-8">
          <div className="max-w-4xl mx-auto">
            <Card className="glass-panel border-t-4 border-t-primary">
              <CardHeader>
                <CardTitle>Configure Execution</CardTitle>
                <CardDescription>Set up parameters for a new backtest or paper trading session.</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Left Column: Basic Settings */}
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="runName">Run Name</Label>
                        <Input 
                          id="runName" 
                          placeholder="e.g. Momentum Strategy Q1" 
                          {...form.register("runName")}
                          className="bg-background"
                        />
                        {form.formState.errors.runName && (
                          <p className="text-red-500 text-xs">{form.formState.errors.runName.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <Label>Execution Type</Label>
                        <RadioGroup 
                          defaultValue="backtest" 
                          onValueChange={(val) => form.setValue("type", val as "backtest" | "paper")}
                          className="flex gap-4"
                        >
                          <div className="flex items-center space-x-2 border border-border p-4 rounded-lg bg-card/50 flex-1 hover:border-primary/50 transition-colors cursor-pointer">
                            <RadioGroupItem value="backtest" id="backtest" />
                            <Label htmlFor="backtest" className="cursor-pointer">Backtest</Label>
                          </div>
                          <div className="flex items-center space-x-2 border border-border p-4 rounded-lg bg-card/50 flex-1 hover:border-primary/50 transition-colors cursor-pointer">
                            <RadioGroupItem value="paper" id="paper" />
                            <Label htmlFor="paper" className="cursor-pointer">Paper Trading</Label>
                          </div>
                        </RadioGroup>
                      </div>

                      <div className="space-y-2">
                        <Label>Load Configuration Template</Label>
                        <Select onValueChange={setSelectedConfigFile}>
                          <SelectTrigger className="bg-background">
                            <SelectValue placeholder="Select a config file..." />
                          </SelectTrigger>
                          <SelectContent>
                            {configFiles?.map((name) => (
                              <SelectItem key={name} value={name}>{name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {/* Right Column: Config Editor */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="flex items-center gap-2">
                          <Code size={14} /> 
                          Configuration (JSON)
                        </Label>
                        <span className="text-xs text-muted-foreground">Editable</span>
                      </div>
                      <Textarea 
                        {...form.register("configContent")}
                        className="font-mono text-xs min-h-[300px] bg-black/40 border-border resize-none focus:ring-primary/20"
                        placeholder="{ 'strategy': 'momentum', 'params': { ... } }"
                      />
                      {form.formState.errors.configContent && (
                        <p className="text-red-500 text-xs">{form.formState.errors.configContent.message}</p>
                      )}
                    </div>
                  </div>

                  <div className="flex justify-end pt-4 border-t border-border/50">
                    <Button 
                      type="submit" 
                      size="lg"
                      className="bg-primary hover:bg-primary/90 shadow-lg shadow-primary/25 min-w-[150px]"
                      disabled={createRun.isPending}
                    >
                      {createRun.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Initializing...
                        </>
                      ) : (
                        <>
                          <PlayCircle className="mr-2 h-5 w-5" />
                          Start Run
                        </>
                      )}
                    </Button>
                  </div>

                </form>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
