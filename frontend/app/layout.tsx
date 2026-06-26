import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WorldMap 3D — Viewer",
  description: "地図で道沿いを選んで3D化し、歩いて回れるビューワー",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
