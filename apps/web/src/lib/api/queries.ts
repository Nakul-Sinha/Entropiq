/** TanStack Query keys and option builders for every read endpoint. */
import { queryOptions } from "@tanstack/react-query";

import {
  getAdvice,
  getAnalysis,
  getFile,
  getGraph,
  getPlan,
  getTreemap,
  getTrend,
  listFiles,
} from "./endpoints";
import type { FileListQuery } from "./types";

export const queryKeys = {
  analysis: (id: string) => ["analysis", id] as const,
  files: (id: string, query: FileListQuery = {}) => ["analysis", id, "files", query] as const,
  file: (id: string, fileId: number) => ["analysis", id, "file", fileId] as const,
  graph: (id: string) => ["analysis", id, "graph"] as const,
  treemap: (id: string) => ["analysis", id, "treemap"] as const,
  plan: (id: string, budgetHours: number) => ["analysis", id, "plan", budgetHours] as const,
  advice: (fileId: number) => ["advice", fileId] as const,
  trend: (repoId: string) => ["trend", repoId] as const,
};

export const analysisQuery = (id: string) =>
  queryOptions({ queryKey: queryKeys.analysis(id), queryFn: () => getAnalysis(id) });

export const filesQuery = (id: string, query: FileListQuery = {}) =>
  queryOptions({ queryKey: queryKeys.files(id, query), queryFn: () => listFiles(id, query) });

export const fileQuery = (id: string, fileId: number) =>
  queryOptions({ queryKey: queryKeys.file(id, fileId), queryFn: () => getFile(id, fileId) });

export const graphQuery = (id: string) =>
  queryOptions({ queryKey: queryKeys.graph(id), queryFn: () => getGraph(id) });

export const treemapQuery = (id: string) =>
  queryOptions({ queryKey: queryKeys.treemap(id), queryFn: () => getTreemap(id) });

export const planQuery = (id: string, budgetHours: number) =>
  queryOptions({
    queryKey: queryKeys.plan(id, budgetHours),
    queryFn: () => getPlan(id, budgetHours),
  });

export const adviceQuery = (fileId: number) =>
  queryOptions({ queryKey: queryKeys.advice(fileId), queryFn: () => getAdvice(fileId) });

export const trendQuery = (repoId: string) =>
  queryOptions({ queryKey: queryKeys.trend(repoId), queryFn: () => getTrend(repoId) });
