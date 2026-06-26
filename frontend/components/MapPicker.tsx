"use client";

// Google マップを表示し、クリックで始点→終点を選んで道沿いの点列を返す。
// 地図は @vis.gl/react-google-maps、ポリラインは google.maps.Polyline を直接使う。

import {
  APIProvider,
  Map,
  Marker,
  useMap,
  useMapsLibrary,
} from "@vis.gl/react-google-maps";
import { useEffect, useRef, useState } from "react";
import { sampleLine, type LatLng } from "@/lib/geo";

interface MapPickerProps {
  apiKey: string;
  maxPoints: number;
  stepMeters: number;
  resetSignal: number;
  onPointsChange: (points: LatLng[]) => void;
}

const TOKYO: LatLng = { lat: 35.6595, lng: 139.7005 };

// 始点・終点を結ぶポリラインを描画する補助コンポーネント。
function RoutePolyline({ path }: { path: LatLng[] }) {
  const map = useMap();
  const mapsLib = useMapsLibrary("maps");
  const lineRef = useRef<google.maps.Polyline | null>(null);

  useEffect(() => {
    if (!map || !mapsLib) return;
    if (!lineRef.current) {
      lineRef.current = new mapsLib.Polyline({
        strokeColor: "#2f6fed",
        strokeWeight: 4,
        strokeOpacity: 0.9,
      });
      lineRef.current.setMap(map);
    }
    lineRef.current.setPath(path);
  }, [map, mapsLib, path]);

  useEffect(() => () => lineRef.current?.setMap(null), []);
  return null;
}

function PickerInner({
  maxPoints,
  stepMeters,
  resetSignal,
  onPointsChange,
}: Omit<MapPickerProps, "apiKey">) {
  const map = useMap();
  const [start, setStart] = useState<LatLng | null>(null);
  const [end, setEnd] = useState<LatLng | null>(null);
  const [points, setPoints] = useState<LatLng[]>([]);

  // クリック: 1回目=始点、2回目=終点、3回目以降=リセットして新しい始点。
  useEffect(() => {
    if (!map) return;
    const listener = map.addListener(
      "click",
      (e: google.maps.MapMouseEvent) => {
        if (!e.latLng) return;
        const ll: LatLng = { lat: e.latLng.lat(), lng: e.latLng.lng() };
        if (!start || end) {
          setStart(ll);
          setEnd(null);
          setPoints([]);
          onPointsChange([]);
        } else {
          const pts = sampleLine(start, ll, stepMeters, maxPoints);
          setEnd(ll);
          setPoints(pts);
          onPointsChange(pts);
        }
      },
    );
    return () => listener.remove();
  }, [map, start, end, stepMeters, maxPoints, onPointsChange]);

  // 間隔(m) 変更を即反映する。
  useEffect(() => {
    if (start && end) {
      const pts = sampleLine(start, end, stepMeters, maxPoints);
      setPoints(pts);
      onPointsChange(pts);
    }
    // start/end 変化時は上のクリック処理で更新するため stepMeters のみを見る。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepMeters]);

  // 親からのリセット要求。
  useEffect(() => {
    if (resetSignal === 0) return;
    setStart(null);
    setEnd(null);
    setPoints([]);
    onPointsChange([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetSignal]);

  return (
    <>
      {start && <Marker position={start} label="始" />}
      {end && <Marker position={end} label="終" />}
      {start && end && <RoutePolyline path={points} />}
    </>
  );
}

export default function MapPicker({ apiKey, ...rest }: MapPickerProps) {
  return (
    <APIProvider apiKey={apiKey}>
      <Map
        defaultCenter={TOKYO}
        defaultZoom={18}
        gestureHandling="greedy"
        disableDefaultUI={false}
        streetViewControl={false}
        mapTypeControl={false}
        fullscreenControl={false}
        style={{ width: "100%", height: "100%" }}
      >
        <PickerInner {...rest} />
      </Map>
    </APIProvider>
  );
}
