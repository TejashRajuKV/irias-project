import type { Metadata } from "next";
import { Playfair_Display, Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/frontend/components/ui/toaster";
import { Navbar } from "@/frontend/components/Navbar";
import { Suspense } from "react";

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "IR-AIS | India Road Accident Intelligence System",
  description:
    "Advanced ML-powered road accident severity prediction and analysis system for India. Leveraging 6+ machine learning models on 12,316 accident records with 32 features.",
  keywords: [
    "IR-AIS",
    "India",
    "Road Accident",
    "Machine Learning",
    "Severity Prediction",
    "Accident Analysis",
    "Traffic Safety",
  ],
  authors: [{ name: "IR-AIS Team" }],
  icons: {
    icon: "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🛣️</text></svg>",
  },
  openGraph: {
    title: "IR-AIS | India Road Accident Intelligence System",
    description:
      "ML-powered road accident severity prediction and analysis for India",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${playfair.variable} ${inter.variable} antialiased bg-background text-foreground min-h-screen font-sans`}
      >
        <Suspense fallback={<div className="h-[67px] border-b border-black/10 bg-background" />}>
          <Navbar />
        </Suspense>
        <div className="min-h-[calc(100vh-64px)]">{children}</div>
        <Toaster />
      </body>
    </html>
  );
}
