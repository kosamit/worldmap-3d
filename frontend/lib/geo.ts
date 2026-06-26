// 緯度経度をローカル平面メートルに近似し、道沿いの点列をサンプリングする。
// バックエンドの geo.py と同じ近似（数km程度なら十分）。

export interface LatLng {
  lat: number;
  lng: number;
}

const METERS_PER_DEG_LAT = 110540;
const METERS_PER_DEG_LNG = 111320;

// 始点・終点間の距離（メートル）。
export function distanceMeters(a: LatLng, b: LatLng): number {
  const latMid = ((a.lat + b.lat) / 2) * (Math.PI / 180);
  const north = (b.lat - a.lat) * METERS_PER_DEG_LAT;
  const east = (b.lng - a.lng) * METERS_PER_DEG_LNG * Math.cos(latMid);
  return Math.hypot(north, east);
}

// 始点〜終点を stepMeters 間隔でサンプリングした点列を返す（最大 maxPoints 点）。
export function sampleLine(
  start: LatLng,
  end: LatLng,
  stepMeters: number,
  maxPoints: number,
): LatLng[] {
  const distance = distanceMeters(start, end);
  const count = Math.max(
    2,
    Math.min(maxPoints, Math.round(distance / stepMeters) + 1),
  );
  const points: LatLng[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    points.push({
      lat: start.lat + (end.lat - start.lat) * t,
      lng: start.lng + (end.lng - start.lng) * t,
    });
  }
  return points;
}
