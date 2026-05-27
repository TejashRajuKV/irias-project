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
    const tmpFile = path.join(
      os.tmpdir(),
      `ir_ais_regress_${Date.now()}.json`
    );
    fs.writeFileSync(tmpFile, JSON.stringify(body));

    // Call Python prediction script
    const scriptPath = path.join(process.cwd(), "ml-service").replace(/\\/g, "/");
    const tmpFilePy = tmpFile.replace(/\\/g, "/");
    const pythonScript = `
import json, sys
sys.path.insert(0, '${scriptPath}')
from predict import regress
features = json.load(open('${tmpFilePy}'))
result = regress(features)
print(json.dumps(result))
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

    const prediction = JSON.parse(result.trim());

    // Read best model name dynamically
    let modelName = "Best Regressor";
    try {
      const bestModels = JSON.parse(
        fs.readFileSync(
          path.join(process.cwd(), "ml-service/models/best_models.json"),
          "utf-8"
        )
      );
      modelName = bestModels.best_regressor_name || modelName;
    } catch {
      // Fall back to default name if file doesn't exist
    }

    return NextResponse.json({
      predicted_casualties: prediction.prediction_rounded,
      predicted_casualties_float: prediction.prediction,
      model: modelName,
      input_features: body,
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "Unknown error occurred";
    console.error("Regression error:", error);
    return NextResponse.json(
      { error: "Regression prediction failed", details: message },
      { status: 500 }
    );
  }
}
