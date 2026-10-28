/**
 * A stand-in for next/navigation in component tests:
 *
 *   vi.mock("next/navigation", () => import("@/test/next-navigation"));
 *
 * Set the current location with setPathname (or by editing `navigation`), and read router calls
 * from `navigation.pushed` and `navigation.replaced`. vitest.setup.ts resets it after each test.
 */

export const navigation = {
  pathname: "/",
  searchParams: new URLSearchParams(),
  params: {} as Record<string, string>,
  pushed: [] as string[],
  replaced: [] as string[],
};

/** Sets the pathname, and the search params when `pathname` has a query string. */
export function setPathname(pathname: string): void {
  const [path, query = ""] = pathname.split("?");
  navigation.pathname = path;
  navigation.searchParams = new URLSearchParams(query);
}

export function resetNavigation(): void {
  navigation.pathname = "/";
  navigation.searchParams = new URLSearchParams();
  navigation.params = {};
  navigation.pushed = [];
  navigation.replaced = [];
}

export function usePathname(): string {
  return navigation.pathname;
}

export function useSearchParams(): URLSearchParams {
  return navigation.searchParams;
}

export function useParams(): Record<string, string> {
  return navigation.params;
}

const router = {
  push: (href: string) => {
    navigation.pushed.push(href);
  },
  replace: (href: string) => {
    navigation.replaced.push(href);
  },
  back: () => undefined,
  forward: () => undefined,
  refresh: () => undefined,
  prefetch: () => undefined,
};

export function useRouter(): typeof router {
  return router;
}

export function notFound(): never {
  throw new Error("NEXT_NOT_FOUND");
}

export function redirect(url: string): never {
  throw new Error(`NEXT_REDIRECT ${url}`);
}
