import { useEffect, useRef, useState } from "react";
import type { Selection } from "./api";

export type Page =
  | "skills"
  | "memories"
  | "algorithm-visualization"
  | "graphs"
  | "settings"
  | "service-settings";
export type Route = { page: Page; selection?: Selection; tab?: string };
type Entry = { index: number; parent?: number };

export function readRoute(): Route {
  const params = new URLSearchParams(location.hash.slice(1));
  const page = params.get("page");
  const route: Route = {
    page:
      page === "memories" ||
      page === "graphs" ||
      page === "algorithm-visualization" ||
      page === "settings" ||
      page === "service-settings"
        ? page
        : "skills",
  };
  const id = params.get("skill");
  if (id && (route.page === "skills" || route.page === "graphs")) {
    route.selection = {
      id,
      source: params.get("source") === "graph" ? "graph" : "local",
      name: params.get("name") || id,
      version: params.get("version") || undefined,
    };
    const tab = params.get("tab");
    route.tab =
      tab === "diff" ||
      tab === "versions" ||
      tab === "content" ||
      tab === "trajectories"
        ? tab
        : route.selection.version
          ? "content"
          : "versions";
  }
  return route;
}

export function routeHash(route: Route): string {
  const params = new URLSearchParams({ page: route.page });
  if (route.selection) {
    const s = route.selection;
    params.set("skill", s.id);
    params.set("source", s.source);
    params.set("name", s.name);
    if (s.version) params.set("version", s.version);
    params.set("tab", route.tab || (s.version ? "content" : "versions"));
  }
  return "#" + params;
}

export function useNavigation() {
  const [route, setRoute] = useState(readRoute);
  const dirty = useRef(false);
  const current = useRef<Entry>(
    history.state?.mindmemosNavigation || { index: 0 },
  );
  const currentHash = useRef(routeHash(route));
  const restoring = useRef(false);

  useEffect(() => {
    history.replaceState(
      { ...history.state, mindmemosNavigation: current.current },
      "",
      currentHash.current,
    );
    const pop = () => {
      if (restoring.current) {
        restoring.current = false;
        return;
      }
      const target: Entry = history.state?.mindmemosNavigation || {
        index: current.current.index + 1,
      };
      if (dirty.current && !window.confirm("Discard unsaved changes?")) {
        const delta = current.current.index - target.index;
        // popstate arrives after the browser has moved. Restore the original
        // entry on cancel so both the draft and the forward stack survive.
        if (delta) {
          restoring.current = true;
          history.go(delta);
        }
        return;
      }
      dirty.current = false;
      current.current = target;
      const next = readRoute();
      currentHash.current = routeHash(next);
      history.replaceState(
        { ...history.state, mindmemosNavigation: target },
        "",
        currentHash.current,
      );
      setRoute(next);
    };
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty.current) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("popstate", pop);
    window.addEventListener("beforeunload", warn);
    return () => {
      window.removeEventListener("popstate", pop);
      window.removeEventListener("beforeunload", warn);
    };
  }, []);

  function navigate(next: Route, replace = false) {
    const hash = routeHash(next);
    if (restoring.current || hash === currentHash.current) return;
    if (dirty.current && !window.confirm("Discard unsaved changes?")) return;
    dirty.current = false;
    const parent = next.selection
      ? route.selection
        ? current.current.parent
        : current.current.index
      : undefined;
    current.current = {
      index: current.current.index + (replace ? 0 : 1),
      parent,
    };
    currentHash.current = hash;
    history[replace ? "replaceState" : "pushState"](
      { mindmemosNavigation: current.current },
      "",
      hash,
    );
    setRoute(next);
  }

  function back() {
    if (restoring.current) return;
    const parent = current.current.parent;
    if (parent !== undefined && parent < current.current.index) {
      history.go(parent - current.current.index);
    } else {
      navigate({ page: route.page }, true);
    }
  }
  return {
    route,
    navigate,
    back,
    setDirty: (value: boolean) => {
      dirty.current = value;
    },
  };
}
