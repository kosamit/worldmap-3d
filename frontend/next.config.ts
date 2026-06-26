import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 親ディレクトリの別 lockfile を誤検出しないよう、このフォルダをルートに固定。
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
