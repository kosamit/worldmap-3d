"use client";

// Google マップ。地点検索＋1クリックで降り立つ地点を選ぶ。
// 現在地はピン1つで表示し、ツアー移動に追従する。経路探索はしない
// （隣接ノードの取得は親側が StreetViewService で都度行う）。

import {
  APIProvider,
  Map,
  Marker,
  useApiIsLoaded,
  useMap,
  useMapsLibrary,
} from "@vis.gl/react-google-maps";
import { useEffect, useRef } from "react";
import type { LatLng } from "@/lib/geo";

interface MapPickerProps {
  apiKey: string;
  current: LatLng | null;
  onPick: (point: LatLng) => void;
  onReady?: () => void;
}

const TOKYO: LatLng = { lat: 35.6595, lng: 139.7005 };

// 地点検索ボックス（Places Autocomplete）。選択地点へ地図を移動する。
function SearchBox() {
  const map = useMap();
  const placesLib = useMapsLibrary("places");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!placesLib || !map || !inputRef.current) return;
    const autocomplete = new placesLib.Autocomplete(inputRef.current, {
      fields: ["geometry"],
    });
    const listener = autocomplete.addListener("place_changed", () => {
      const location = autocomplete.getPlace().geometry?.location;
      if (location) {
        map.panTo(location);
        map.setZoom(18);
      }
    });
    return () => listener.remove();
  }, [placesLib, map]);

  return (
    <input
      ref={inputRef}
      className="mapSearch"
      type="text"
      placeholder="地点を検索（例: 渋谷駅）"
    />
  );
}

function PickerInner({
  current,
  onPick,
  onReady,
}: Omit<MapPickerProps, "apiKey">) {
  const map = useMap();
  const apiLoaded = useApiIsLoaded();

  useEffect(() => {
    if (apiLoaded) onReady?.();
  }, [apiLoaded, onReady]);

  useEffect(() => {
    if (map && current) map.panTo(current);
  }, [map, current]);

  useEffect(() => {
    if (!map) return;
    const listener = map.addListener("click", (e: google.maps.MapMouseEvent) => {
      if (e.latLng) onPick({ lat: e.latLng.lat(), lng: e.latLng.lng() });
    });
    return () => listener.remove();
  }, [map, onPick]);

  return current ? <Marker position={current} /> : null;
}

export default function MapPicker({ apiKey, ...rest }: MapPickerProps) {
  return (
    <APIProvider apiKey={apiKey}>
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <SearchBox />
        <Map
          defaultCenter={rest.current ?? TOKYO}
          defaultZoom={18}
          gestureHandling="greedy"
          streetViewControl={false}
          mapTypeControl={false}
          fullscreenControl={false}
          style={{ width: "100%", height: "100%" }}
        >
          <PickerInner {...rest} />
        </Map>
      </div>
    </APIProvider>
  );
}
