import './globals.css';

export const metadata = {
  title: 'NeuroRoute Console',
  description: 'Adaptive LLM inference orchestration — routing, execution graphs, confidence and cost per request.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
