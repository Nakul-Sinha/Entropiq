import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/site-header";

import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: { default: "Entropiq", template: "%s | Entropiq" },
  description:
    "Entropiq ranks the files of a Git repository by the maintenance pain they are likely to cause.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Providers>
          <SiteHeader />
          <main className="mx-auto w-full max-w-7xl px-4 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
