import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

interface RawClassMetrics {
  accuracy: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_weighted: number;
  approach: string;
  roc_auc_ovr?: number;
  confusion_matrix?: number[][];
  best_params?: Record<string, unknown>;
}

interface RawRegMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  approach: string;
  best_params?: Record<string, unknown>;
}

function readJsonSafe(filePath: string): Record<string, unknown> {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return {};
  }
}

export async function GET() {
  try {
    const classMetrics = readJsonSafe(
      path.join(process.cwd(), "ml-service/models/classification_metrics.json")
    ) as Record<string, RawClassMetrics>;

    const classMetricsPca = readJsonSafe(
      path.join(process.cwd(), "ml-service/models/classification_metrics_pca.json")
    ) as Record<string, RawClassMetrics>;

    const regMetrics = readJsonSafe(
      path.join(process.cwd(), "ml-service/models/regression_metrics.json")
    ) as Record<string, RawRegMetrics>;

    const regMetricsPca = readJsonSafe(
      path.join(process.cwd(), "ml-service/models/regression_metrics_pca.json")
    ) as Record<string, RawRegMetrics>;

    const auxiliaryMetrics = readJsonSafe(
      path.join(process.cwd(), "ml-service/models/auxiliary_metrics.json")
    );

    const classificationModels = [
      ...Object.entries(classMetrics).map(([name, m]) => ({
        name,
        accuracy: m.accuracy,
        precision: m.precision_weighted,
        recall: m.recall_weighted,
        f1Score: m.f1_weighted,
        approach: m.approach,
        rocAuc: m.roc_auc_ovr ?? null,
        confusionMatrix: m.confusion_matrix ?? [],
        bestParams: m.best_params ?? null,
      })),
      ...Object.entries(classMetricsPca).map(([name, m]) => ({
        name: `${name} (PCA)`,
        accuracy: m.accuracy,
        precision: m.precision_weighted,
        recall: m.recall_weighted,
        f1Score: m.f1_weighted,
        approach: m.approach,
        rocAuc: m.roc_auc_ovr ?? null,
        confusionMatrix: m.confusion_matrix ?? [],
        bestParams: m.best_params ?? null,
      })),
    ];

    const regressionModels = [
      ...Object.entries(regMetrics).map(([name, m]) => ({
        name,
        mae: m.mae,
        mse: m.mse,
        rmse: m.rmse,
        r2: m.r2,
        approach: m.approach,
        bestParams: m.best_params ?? null,
      })),
      ...Object.entries(regMetricsPca).map(([name, m]) => ({
        name: `${name} (PCA)`,
        mae: m.mae,
        mse: m.mse,
        rmse: m.rmse,
        r2: m.r2,
        approach: m.approach,
        bestParams: m.best_params ?? null,
      })),
    ];

    return NextResponse.json({
      classification: classificationModels,
      regression: regressionModels,
      auxiliary: auxiliaryMetrics,
    });
  } catch (error) {
    console.error("Model comparison error:", error);
    return NextResponse.json(
      { error: "Failed to load model data" },
      { status: 500 }
    );
  }
}
