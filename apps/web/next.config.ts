import type { NextConfig } from "next";

const apiUrl = (process.env.API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  output: "standalone",
  // Gzip in the Next server would buffer the SSE progress stream behind the rewrite; Caddy
  // compresses responses in production.
  compress: false,
  poweredByHeader: false,
  reactStrictMode: true,
  async rewrites() {
    // Rewrites are resolved at build time, so Docker builds must pass API_URL as a build argument.
    return [{ source: "/api/v1/:path*", destination: `${apiUrl}/api/v1/:path*` }];
  },
};

export default nextConfig;
