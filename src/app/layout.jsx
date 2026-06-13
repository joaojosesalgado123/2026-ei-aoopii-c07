import "./globals.css";

export const metadata = {
  title: "AI Detector",
  description: "Detete imagens falsas geradas por IA em segundos."
};

export default function RootLayout({ children }) {
  return (
    <html lang="pt">
      <body>{children}</body>
    </html>
  );
}
