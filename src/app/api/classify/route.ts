import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    if (!body || typeof body !== "object") {
      return NextResponse.json(
        { error: "Request body must be JSON" },
        { status: 400 }
      );
    }

    // Write features to temp file
    const tmpFile = path.join(os.tmpdir(), `ir_ais_input_${Date.now()}.json`);
    fs.writeFileSync(tmpFile, JSON.stringify(body));

    // Call Python prediction script
    const scriptPath = path.join(process.cwd(), "ml-service").replace(/\\/g, "/");
    const tmpFilePy = tmpFile.replace(/\\/g, "/");
    const pythonScript = `
import json, sys
sys.path.insert(0, '${scriptPath}')
from predict import classify, predict_auxiliary
features = json.load(open('${tmpFilePy}'))
result_cls = classify(features)
result_aux = predict_auxiliary(features)
print(json.dumps({"classification": result_cls, "auxiliary": result_aux}))
`;

    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    let result: string;
    try {
      result = execFileSync(pythonCmd, ["-c", pythonScript], {
        timeout: 30000,
        encoding: "utf-8",
      });
    } finally {
      try { fs.unlinkSync(tmpFile); } catch {}
    }

    const output = JSON.parse(result.trim());
    const prediction = output.classification;
    const auxiliary = output.auxiliary;

    // Read best model name dynamically
    let modelName = "Best Classifier";
    try {
      const bestModels = JSON.parse(
        fs.readFileSync(
          path.join(process.cwd(), "ml-service/models/best_models.json"),
          "utf-8"
        )
      );
      modelName = bestModels.best_classifier_name || modelName;
    } catch {
      // Fall back to default name if file doesn't exist
    }

    return NextResponse.json({
      severity: prediction.prediction,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      model: modelName,
      auxiliary: auxiliary,
      input_features: body,
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "Unknown error occurred";
    console.error("Classification error:", error);
    return NextResponse.json(
      { error: "Prediction failed", details: message },
      { status: 500 }
    );
  }
}
