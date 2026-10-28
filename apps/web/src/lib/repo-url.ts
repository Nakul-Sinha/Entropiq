/**
 * Client-side repository URL checks that mirror tdp_core.security.validate_repo_url (without the
 * DNS step), so the form and the mock backend reject exactly what the API rejects.
 */

export const SUPPORTED_GIT_HOSTS = ["github.com", "gitlab.com"] as const;

const MAX_URL_LENGTH = 2048;
const PART = /^[A-Za-z0-9_.-]{1,100}$/;
const EXAMPLE = "https://github.com/owner/name";

type Parsed =
  { ok: true; host: string; owner: string; name: string } | { ok: false; error: string };

function parse(input: string): Parsed {
  const url = input.trim();
  const fail = (error: string): Parsed => ({ ok: false, error });
  if (url === "") return fail("Enter a repository URL.");
  if (url.length > MAX_URL_LENGTH) return fail("The URL is too long.");
  if (!/^[\x21-\x7e]+$/.test(url))
    return fail("The URL contains spaces or characters that are not allowed.");
  const scheme = /^([A-Za-z][A-Za-z0-9+.-]*):\/\/(.*)$/.exec(url);
  if (!scheme || scheme[1].toLowerCase() !== "https") return fail("Use an https URL.");
  const rest = scheme[2];
  if (rest.includes("?") || rest.includes("#")) {
    return fail("Remove the query string or fragment from the URL.");
  }
  const slash = rest.indexOf("/");
  const authority = slash === -1 ? rest : rest.slice(0, slash);
  const path = slash === -1 ? "" : rest.slice(slash);
  if (authority.includes("@")) return fail("Remove the user name or password from the URL.");
  const hostPort = /^([A-Za-z0-9.-]+)(?::(\d*))?$/.exec(authority);
  if (!hostPort) return fail("The host name is not valid.");
  if (hostPort[2] !== undefined && hostPort[2] !== "443")
    return fail("Use the default https port.");
  const host = hostPort[1].toLowerCase();
  if (!(SUPPORTED_GIT_HOSTS as readonly string[]).includes(host)) {
    return fail("Only GitHub and GitLab repositories are supported.");
  }
  const parts = (path.endsWith("/") ? path.slice(0, -1) : path).split("/");
  if (parts.length !== 3 || parts[0] !== "")
    return fail(`Use the repository URL, like ${EXAMPLE}.`);
  let [, owner, name] = parts;
  if (name.toLowerCase().endsWith(".git")) name = name.slice(0, -4);
  for (const part of [owner, name]) {
    if (!PART.test(part) || /^\.+$/.test(part)) {
      return fail("The owner and repository names may only use letters, digits, '.', '_' and '-'.");
    }
  }
  if (name.toLowerCase().endsWith(".git")) return fail(`Use the repository URL, like ${EXAMPLE}.`);
  if (host === "github.com") {
    owner = owner.toLowerCase();
    name = name.toLowerCase();
  }
  return { ok: true, host, owner, name };
}

/**
 * Null when `input` is an analysable repository URL: https, no user info, port absent or 443,
 * host github.com or gitlab.com, path /owner/name with an optional .git and trailing slash, and no
 * query or fragment. Otherwise a sentence saying what to fix.
 */
export function repoUrlError(input: string): string | null {
  const parsed = parse(input);
  return parsed.ok ? null : parsed.error;
}

/**
 * The canonical form "https://<host>/<owner>/<name>" (no .git, no trailing slash; lower-cased on
 * github.com, which ignores case). Expects a valid URL and throws an Error otherwise.
 */
export function canonicalRepoUrl(input: string): string {
  const parsed = parse(input);
  if (!parsed.ok) throw new Error(parsed.error);
  return `https://${parsed.host}/${parsed.owner}/${parsed.name}`;
}
