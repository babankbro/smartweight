import type { Metadata } from "next";
import { Kanit, Outfit } from "next/font/google";
import "./globals.css";

const kanit = Kanit({
  subsets: ["latin", "thai"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-kanit",
});

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  variable: "--font-outfit",
});

export const metadata: Metadata = {
  title: "Smart CattleWeight AI",
  description: "AI-powered cattle weight estimation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* Browser extensions (e.g. Grammarly) inject attributes into <body>
          before hydration. This only silences attribute mismatches on this
          one element - mismatches in children are still reported. */}
      <body
        suppressHydrationWarning
        className={`${kanit.variable} ${outfit.variable} font-sans antialiased bg-[#e8e4dc] text-[#1c1c1c] selection:bg-[#1c1c1c] selection:text-white`}
      >
        <div className="w-full max-w-5xl min-h-screen bg-[#f3f0ea] md:shadow-2xl relative mx-auto overflow-x-hidden md:border-x md:border-[#e3dfd6]">
          {children}
        </div>
      </body>
    </html>
  );
}
