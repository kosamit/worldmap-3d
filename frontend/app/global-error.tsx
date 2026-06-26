"use client";

// アプリ全体のフォールバックエラー画面。
// 既定の builtin global-error を自前のものに置き換える。

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="ja">
      <body
        style={{
          margin: 0,
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "12px",
          background: "#101418",
          color: "#e6e9ed",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <h2>エラーが発生しました</h2>
        <p style={{ color: "#9fb3c8", fontSize: "0.85rem" }}>{error.message}</p>
        <button
          onClick={() => reset()}
          style={{
            padding: "8px 16px",
            borderRadius: 6,
            border: "none",
            background: "#2f6fed",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          再試行
        </button>
      </body>
    </html>
  );
}
