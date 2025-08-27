"use client";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, CheckCircle, XCircle } from "lucide-react";
import { PanelSection } from "./panel-section";
import type { GuardrailCheck, OutputGuardrailCheck } from "@/lib/types";

interface GuardrailsProps {
  guardrails: GuardrailCheck[];
  inputGuardrails: string[];
  outputGuardrails: OutputGuardrailCheck[];
  outputGuardrailNames: string[];
}

export function Guardrails({ guardrails, inputGuardrails, outputGuardrails, outputGuardrailNames }: GuardrailsProps) {
  const guardrailNameMap: Record<string, string> = {
    relevance_guardrail: "Relevance Guardrail",
    jailbreak_guardrail: "Jailbreak Guardrail",
    tov_guardrail: "Tone of Voice Guardrail",
  };

  const guardrailDescriptionMap: Record<string, string> = {
    "Relevance Guardrail": "Ensure messages are relevant to airline support",
    "Jailbreak Guardrail":
      "Detect and block attempts to bypass or override system instructions",
    "Tone of Voice Guardrail": "Format output according to brand tone of voice guidelines",
  };

  const extractGuardrailName = (rawName: string): string =>
    guardrailNameMap[rawName] ?? rawName;

  const guardrailsToShow: GuardrailCheck[] = inputGuardrails.map((rawName) => {
    const existing = guardrails.find((gr) => gr.name === rawName);
    if (existing) {
      return existing;
    }
    return {
      id: rawName,
      name: rawName,
      input: "",
      reasoning: "",
      passed: false,
      timestamp: new Date(),
    };
  });

  const outputGuardrailsToShow: OutputGuardrailCheck[] = outputGuardrailNames.map((rawName) => {
    const existing = outputGuardrails.find((gr) => gr.name === rawName);
    if (existing) {
      return existing;
    }
    return {
      id: rawName,
      name: rawName,
      input_text: "",
      output: "",
      reasoning: "",
      final_text: "",
      tripwire_triggered: false,
      timestamp: new Date(),
    };
  });

  return (
    <PanelSection
      title="Guardrails"
      icon={<Shield className="h-4 w-4 text-blue-600" />}
    >
      {/* Input Guardrails Section */}
      {guardrailsToShow.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Input Guardrails</h3>
          <div className="grid grid-cols-3 gap-3">
            {guardrailsToShow.map((gr) => (
              <Card
                key={gr.id}
                className={`bg-white border-gray-200 transition-all ${
                  !gr.input ? "opacity-60" : ""
                }`}
              >
                <CardHeader className="p-3 pb-1">
                  <CardTitle className="text-sm flex items-center text-zinc-900">
                    {extractGuardrailName(gr.name)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3 pt-1">
                  <p className="text-xs font-light text-zinc-500 mb-1">
                    {(() => {
                      const title = extractGuardrailName(gr.name);
                      return guardrailDescriptionMap[title] ?? gr.input;
                    })()}
                  </p>
                  <div className="flex text-xs">
                    {!gr.input || gr.passed ? (
                      <Badge className="mt-2 px-2 py-1 bg-emerald-500 hover:bg-emerald-600 flex items-center text-white">
                        <CheckCircle className="h-4 w-4 mr-1 text-white" />
                        Passed
                      </Badge>
                    ) : (
                      <Badge className="mt-2 px-2 py-1 bg-red-500 hover:bg-red-600 flex items-center text-white">
                        <XCircle className="h-4 w-4 mr-1 text-white" />
                        Failed
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Output Guardrails Section */}
      {outputGuardrailsToShow.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">Output Guardrails</h3>
          <div className="grid grid-cols-3 gap-3">
            {outputGuardrailsToShow.map((gr) => (
              <Card
                key={gr.id}
                className={`bg-white border-gray-200 transition-all ${
                  !gr.input_text ? "opacity-60" : ""
                }`}
              >
                <CardHeader className="p-3 pb-1">
                  <CardTitle className="text-sm flex items-center text-zinc-900">
                    {extractGuardrailName(gr.name)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3 pt-1">
                  <p className="text-xs font-light text-zinc-500 mb-1">
                    {(() => {
                      const title = extractGuardrailName(gr.name);
                      return guardrailDescriptionMap[title] ?? gr.reasoning;
                    })()}
                  </p>
                  <div className="flex text-xs">
                    {!gr.input_text || !gr.tripwire_triggered ? (
                      <Badge className="mt-2 px-2 py-1 bg-emerald-500 hover:bg-emerald-600 flex items-center text-white">
                        <CheckCircle className="h-4 w-4 mr-1 text-white" />
                        Applied
                      </Badge>
                    ) : (
                      <Badge className="mt-2 px-2 py-1 bg-red-500 hover:bg-red-600 flex items-center text-white">
                        <XCircle className="h-4 w-4 mr-1 text-white" />
                        Blocked
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </PanelSection>
  );
}
